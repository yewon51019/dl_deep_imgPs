# =====================================================================
#  split_exp300.py  —  절취선·분할 각인 처리 실험 (개발용만)
# =====================================================================
#
#  [문제]  'SAMJIN OSTEMIN', 'JW5 JW5', 'SCD 37.5' 처럼 각인이 띄어쓰기·절취선으로 나뉜 약은
#          CRAFT 가 박스를 여러 개 찾고, 박스를 이어 붙이는 순서가 틀리면('AINREP' ← REP AIN) 매칭이 흔들려요.
#  [DB]   make_search_db.py 가 만든 search_db_parts.csv 의 front_parts / back_parts (조각 목록, '|' 로 이음)
#
#  [비교]  박스가 2개 이상인 방향에서만 새 규칙이 달라지고, 나머지는 기준과 같아요.
#   dbscore 쪽 (dbscore_exp300.py 의 teacher forcing 점수, 합계(sum)·평균(mean) 둘 다)
#     D0  : 박스를 읽는 순서대로 이어 붙인 이미지 vs 전체 각인            (= dbscore concat, 기준)
#     Da  : (a) 박스를 여러 순서로 이어 붙인 이미지 중 최고 vs 전체 각인
#     Db  : (b) 박스별 점수 vs 조각별 점수 — 조각마다 서로 다른 박스를 하나씩 짝지어(최적 짝짓기) 합산.
#              조각이 1개인 후보나 박스가 조각보다 적으면 '박스별 최고 점수'(dbscore max)로 계산
#   자유 읽기 쪽 (지금 파이프라인: 자유롭게 읽은 글자 → 편집거리)
#     F0      : 12방향 다수결 (지금 방식)            Fp     : 다수결로 고른 방향에서 박스 글자를 여러 순서로 붙여 DB 와 가장 비슷한 것
#     V3      : 보정 V3 (모든 방향 후보)             V3p    : V3 + 모든 방향에서 박스 순서 바꿔 붙이기
#   순서 바꾸기: 박스 4개 이하면 모든 순서, 5개 이상이면 원래 순서와 거꾸로만
#
#  [출력]  --out 폴더:  split_dev.csv (사진별·방식별), split_metrics.csv
#          분할 각인 사진 수, 그중 방식별로 좋아진(틀림→맞음) / 나빠진(맞음→틀림) 사진 수
#
#  [실행 예]  (저장소 맨 위 폴더에서)
#     python exp300\split_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_parts.csv ^
#            --vote exp300\results\pred_dev_parseq_P0_align_vote.csv --out exp300\results\split
# =====================================================================

import argparse
import csv
import itertools
import os
import re
import time
from collections import Counter

import torch

import numpy as np

import correct_exp300 as corr
import run_exp300 as base
from dbscore_exp300 import DBScorer, hconcat
from evaluate_exp300 import drug_metrics
from make_search_db import parts as split_parts

NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]
MAX_ALL_PERM = 4


def load_cands(path):
    """후보 [{drug_N, face, text, parts}] (빈 각인 제외)"""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for face, col, pcol in (("front", "print_front", "front_parts"), ("back", "print_back", "back_parts")):
                t = base.norm(r.get(col))
                if t:
                    out.append({"drug_N": r["drug_N"], "face": face, "text": t,
                                "parts": [p for p in (r.get(pcol) or "").split("|") if p] or [t]})
    return out


def perms(n):
    if n <= MAX_ALL_PERM:
        return list(itertools.permutations(range(n)))
    return [tuple(range(n)), tuple(reversed(range(n)))]


def best_assign(S, k_idx, n):
    """S: (박스 n, 전체 문자열 목록) 점수 행렬, k_idx: 조각들의 문자열 번호 → 서로 다른 박스에 조각을 하나씩 짝지은 최대 합"""
    k = len(k_idx)
    best = -np.inf
    for boxes in itertools.permutations(range(n), k):           # n ≤ 6 정도라 다 세어도 빠름
        v = sum(S[b, j] for b, j in zip(boxes, k_idx))
        best = max(best, v)
    return best


def identify_edit(text, db):
    return base.identify(text, "", "", db)[1] if text else ""


def judge_drug(pred, truth):
    return pred if pred else NONE_LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True, help="search_db_parts.csv (조각 열 포함)")
    ap.add_argument("--vote", required=True, help="run_exp300.py --align vote 개발용 결과 (기준선 확인용)")
    ap.add_argument("--out", default="split")
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                 # 개발용만
    if a.limit:
        items = items[: a.limit]
    cands = load_cands(a.db)
    strings = list(dict.fromkeys([c["text"] for c in cands] + [p for c in cands for p in c["parts"]]))
    sid = {s: i for i, s in enumerate(strings)}
    parseq = base.PARSeq()
    scorer = DBScorer(parseq, strings)
    db = base.load_db(a.db)
    cdb = corr.load_db(a.db)
    vote_csv = {r["file_name"]: r for r in csv.DictReader(open(a.vote, encoding="utf-8-sig"))}
    tok = np.array([float(scorer.n_tok[i]) for i in range(len(strings))])
    full_i = np.array([sid[c["text"]] for c in cands])
    print(f"사진 {len(items)}장 (dev), 후보 {len(cands)}개, 점수 매길 문자열 {len(strings)}개 (전체 각인 + 조각)")

    rows = []
    for k, it in enumerate(items, 1):
        fn = it["file_name"]
        gray, aspect, color, ok = base.crop_pill(base.imread_any(it["image_path"]))
        g0 = base.PREPS["P0"][1](gray)
        shape = base.shape_group(aspect)
        truth = base.norm(it["truth"])
        tparts = split_parts(it["truth"])
        t0 = time.time()
        angles = [0, 180] if aspect > 1.3 else list(range(0, 360, a.step))
        best = {f"{v}_{nm}": np.full(len(cands), -np.inf) for v in ("D0", "Da", "Db") for nm in ("sum", "mean")}
        free = []                                                  # 방향별 (이어 붙인 글자, 신뢰도, 박스 글자 목록, 각도)
        n_multi = 0
        for ang in angles:
            r = base.rotate(g0, ang)
            text, conf, boxes, texts = parseq.read_boxes(r)
            boxes_real = base.craft_boxes(r)
            free.append((base.norm(text), conf, [(b, t) for b, t in zip(boxes, texts) if t], ang))
            crops = [r[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes_real] or [r]
            n = len(crops)
            order = sorted(range(n), key=lambda i: (round(boxes_real[i][1] / max(10, (max(b[1] for b in boxes_real) - min(b[1] for b in boxes_real)) / 3 if n > 1 else 10)), boxes_real[i][0])) if boxes_real else [0]
            imgs = list(crops)
            pl = perms(n) if n > 1 else [tuple(order)]
            if n > 1:
                n_multi += 1
                imgs += [hconcat([crops[i] for i in p]) for p in pl]
                imgs.append(hconcat([crops[i] for i in order]))
            s_sum, _ = scorer.score(imgs)                          # (이미지, 문자열) 합계 로그확률
            box_s = s_sum[:n]
            conc0 = s_sum[-1] if n > 1 else s_sum[0]
            concP = s_sum[n:n + len(pl)].max(0) if n > 1 else s_sum[0]
            for nm, f in (("sum", lambda v, t: v), ("mean", lambda v, t: v / t)):
                d0 = f(conc0[full_i], tok[full_i])
                da = f(concP[full_i], tok[full_i])
                bmax = f(box_s[:, full_i].max(0), tok[full_i])      # 조각 1개·박스 부족 → 박스별 최고
                db_ = bmax.copy()
                for ci, c in enumerate(cands):
                    kp = len(c["parts"])
                    if kp >= 2 and n >= kp:
                        ki = [sid[p] for p in c["parts"]]
                        v = best_assign(box_s, ki, n)
                        db_[ci] = max(bmax[ci], f(v, tok[ki].sum() - (kp - 1)))   # [E] 는 조각마다 붙으므로 1개만 셈
                for key, v in (("D0", d0), ("Da", da), ("Db", db_)):
                    cur = best[f"{key}_{nm}"]
                    np.maximum(cur, v, out=cur)
        common = {"file_name": fn, "shape": shape, "truth_drug": it["drug_N"], "truth_imprint": truth,
                  "truth_parts": "|".join(tparts), "is_split": int(len(tparts) >= 2), "n_multi_box_angles": n_multi,
                  "camera_la": it.get("camera_la", "")}
        for key, sc in best.items():
            i1 = int(np.argmax(sc))
            rows.append({**common, "method": f"db_{key}", "pred_drug": cands[i1]["drug_N"], "pred_imprint": cands[i1]["text"],
                         "imprint_ok": int(cands[i1]["text"] == truth)})
        # 자유 읽기: 다수결 (run_exp300 과 같은 규칙) + 순서 바꾸기
        votes = Counter(t for t, *_ in free if 1 <= len(t) <= 12)
        pick = max(free, key=lambda c: (votes.get(c[0], 0) if c[0] else -1, c[1]))
        f0 = identify_edit(pick[0], db)
        rows.append({**common, "method": "F0_vote", "pred_drug": f0 or NONE_LABEL, "pred_imprint": pick[0],
                     "imprint_ok": int(pick[0] == truth)})
        if vote_csv[fn]["pred_imprint"] != pick[0]:
            print(f"    [확인] 다수결 결과가 기존 CSV 와 다름: {vote_csv[fn]['pred_imprint']} vs {pick[0]}")

        def perm_texts(bt):
            if len(bt) < 2:
                return [base.norm("".join(t for _, t in bt))] if bt else []
            return [base.norm("".join(bt[i][1] for i in p)) for p in perms(len(bt))]

        def best_by_sim(texts):
            sc = []
            for t in texts:
                d, s = (base.identify(t, "", "", db)[1], base.identify(t, "", "", db)[3]) if t else ("", 0)
                if d:
                    sc.append((s, t, d))
            return max(sc) if sc else (0, "", "")

        s, t, d = best_by_sim(perm_texts(pick[2]) or [pick[0]])
        rows.append({**common, "method": "Fp_vote_perm", "pred_drug": d or NONE_LABEL, "pred_imprint": t,
                     "imprint_ok": int(t == truth)})
        v3 = corr.predict([vote_csv[fn]], cdb, "V3", 1.0)
        rows.append({**common, "method": "V3", "pred_drug": v3 or NONE_LABEL, "pred_imprint": "", "imprint_ok": ""})
        allp = [x for fr in free for x in (perm_texts(fr[2]) or [fr[0]]) if 1 <= len(x) <= 12]
        bestm = {}
        opt = (True, True, 1.0)
        for c in allp:
            for dd, vv in corr.drug_scores(c, cdb, opt).items():
                bestm[dd] = max(bestm.get(dd, 0.0), vv)
        v3p = corr.decide(bestm)
        rows.append({**common, "method": "V3p_perm", "pred_drug": v3p or NONE_LABEL, "pred_imprint": "", "imprint_ok": ""})
        d = {x["method"]: x for x in rows if x["file_name"] == fn}
        mark = lambda m: "OK" if d[m]["pred_drug"] == it["drug_N"] else "NG"
        print(f"  {k:2d}/{len(items)} {shape} {'분할' if len(tparts) >= 2 else '    '} {truth:<14} "
              f"F0={d['F0_vote']['pred_imprint'] or '-'}({mark('F0_vote')}) Fp={d['Fp_vote_perm']['pred_imprint'] or '-'}({mark('Fp_vote_perm')}) "
              f"V3({mark('V3')}) V3p({mark('V3p_perm')}) | mean: D0({mark('db_D0_mean')}) Da({mark('db_Da_mean')}) Db({mark('db_Db_mean')})"
              f" sum: D0({mark('db_D0_sum')}) Da({mark('db_Da_sum')}) Db({mark('db_Db_sum')})  ({time.time() - t0:.1f}s)")

    with open(os.path.join(a.out, "split_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    methods = [("F0_vote", "F0 다수결 (지금)", None), ("Fp_vote_perm", "Fp 다수결 + 박스 순서 바꾸기", "F0_vote"),
               ("V3", "V3 보정", None), ("V3p_perm", "V3p V3 + 박스 순서 바꾸기", "V3"),
               ("db_D0_sum", "dbscore D0 이어붙임(순서대로) sum", None), ("db_Da_sum", "dbscore Da (a)여러 순서 sum", "db_D0_sum"),
               ("db_Db_sum", "dbscore Db (b)조각 짝짓기 sum", "db_D0_sum"),
               ("db_D0_mean", "dbscore D0 이어붙임(순서대로) mean", None), ("db_Da_mean", "dbscore Da (a)여러 순서 mean", "db_D0_mean"),
               ("db_Db_mean", "dbscore Db (b)조각 짝짓기 mean", "db_D0_mean")]
    met = []
    files = sorted({x["file_name"] for x in rows})
    split_files = {x["file_name"] for x in rows if x["is_split"]}
    for gname, cond in (("전체", lambda x: True), ("둥근", lambda x: x["shape"] == "둥근"),
                        ("길쭉", lambda x: x["shape"] == "길쭉"), ("분할 각인만", lambda x: x["is_split"] == 1)):
        n = len({x["file_name"] for x in rows if cond(x)})
        if not n:
            continue
        print(f"\n[{gname} {n}장]  (1장 = {100 / n:.1f}%p)")
        print(f"  {'방식':<34} " + " ".join(f"{m:>8}" for m in METRICS))
        for m, label, _ in methods:
            sub = [x for x in rows if x["method"] == m and cond(x)]
            acc, p, r, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
            imp = np.mean([x["imprint_ok"] for x in sub]) if sub[0]["imprint_ok"] != "" else float("nan")
            v = [imp, acc, p, r, f1]
            print(f"  {label:<34} " + " ".join("     -  " if np.isnan(s) else f"{s * 100:7.1f}%" for s in v))
            met.append({"group": gname, "n": n, "method": m, **{k2: ("" if np.isnan(s) else round(s * 100, 1))
                                                                for k2, s in zip(METRICS, v)}})
    by = {(x["file_name"], x["method"]): x for x in rows}
    print(f"\n[분할 각인] 개발용 {len(files)}장 중 {len(split_files)}장 (정답 각인이 띄어쓰기·구분 기호로 2조각 이상)")
    print("  방식 (기준 대비)                         분할 사진에서 좋아짐 / 나빠짐   |  전체에서 좋아짐 / 나빠짐")
    for m, label, ref in methods:
        if not ref:
            continue
        ok = lambda f, mm: by[(f, mm)]["pred_drug"] == by[(f, mm)]["truth_drug"]
        up_s = sum(1 for f in split_files if ok(f, m) and not ok(f, ref))
        dn_s = sum(1 for f in split_files if not ok(f, m) and ok(f, ref))
        up = sum(1 for f in files if ok(f, m) and not ok(f, ref))
        dn = sum(1 for f in files if not ok(f, m) and ok(f, ref))
        print(f"  {label:<36} {up_s:3d} / {dn_s:<3d}                       {up:3d} / {dn}")
        met.append({"group": "좋아짐/나빠짐", "method": m, "split_up": up_s, "split_down": dn_s, "all_up": up, "all_down": dn})
    with open(os.path.join(a.out, "split_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k2 for r in met for k2 in r)))
        w.writeheader()
        w.writerows(met)
    print(f"\n저장 완료: {a.out}  (split_dev.csv, split_metrics.csv)")


if __name__ == "__main__":
    main()
