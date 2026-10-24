# =====================================================================
#  combined_exp300.py  —  색·크기 동점 처리 + '합친 방식' + 남은 오류 분류 (개발용만)
# =====================================================================
#
#  [1] 색·크기 동점 처리 (tiebreak)
#      글자 점수 1·2등이 같거나 차이가 작을 때만, 사진의 색(필요하면 크기 비율)으로 약을 고름.
#      · 사진 색: 원본 사진에서 알약 부분(segment_pill)의 Lab 색공간 a·b 중앙값 (밝기는 빼고 '색'만)
#      · 약의 색: search_db_v3.csv 의 color1·color2 (두 색 캡슐은 두 색 대표값의 평균과 비교 — 사진 한 장의 중앙값은 두 색이 섞이므로)
#      · dbscore 에서 글자가 똑같은 후보(ARICEPT 5mg/10mg 앞면)는 같은 점수로 맞춤 (GPU 계산 오차로 동점이 안 잡히던 문제)
#      · 라벨 색 이름 → 실제로 찍히는 색: 개발용 사진으로 색 이름별 대표 a·b 를 구함.
#        자기 사진을 빼고 나머지로 구함(leave-one-out) → 개발용 점수가 부풀려지지 않게
#        (예: 이 촬영 조명에서 흰 알약은 푸르스름하게 찍혀 crop_pill 은 '파랑'으로 판정 → 이름 비교는 못 씀)
#      · 색 이름이 같으면 크기 비율(잘라낸 알약의 긴변/짧은변 vs 라벨 장축/단축)이 더 가까운 쪽
#  [2] 합친 방식
#      C1 : 길 A 의 모든 후보(모든 방향 × 박스 순서 바꾸기)를 V3 규칙(부분일치+헷갈리는 글자)으로 DB 와 비교
#           → 동점·판정없음이면 그 약들(판정없음이면 전체) 중 dbscore 최고 → dbscore 차이도 δ 이하면 색
#      C2 : dbscore 최고 → 1·2등 차이가 δ 이하면 그 약들 중 C1 의 글자 점수 → 그래도 동점이면 색
#      C3 : 길 A 모든 후보 글자(V3 규칙) → 동점이면 색 먼저 → 그래도 못 정하면 dbscore
#           (C1 은 dbscore 가 뒷면 후보('10') 점수로 ARICEPT 동점을 깨 버려서 색까지 못 감 → 색을 먼저)
#      C4 : C3 와 같되 박스 순서 바꾸기 없이 보정 V3 후보(방향별 이어 붙인 글자)만
#  [3] 남은 오류 분류 (길 A, dbscore, 합친 방식에서 틀린 사진)
#      각인 같은 약 > 마크뿐 > 라벨 앞뒤 뒤바뀜 > 각인 일부만 보임 > 박스 못 찾음 > 잘못 읽음 (앞 칸이 먼저)
#      '마크뿐·라벨 뒤바뀜·일부만 보임'은 사진을 눈으로 봐야 알 수 있어서 --notes CSV(사람이 본 기록)를 씀
#
#  [저장]  --out 폴더: combined_dev.csv(사진별·방식별), combined_metrics.csv, error_table.csv, color_check.csv
#  [실행 예]  python exp300\combined_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#             --viz exp300\results\viz_align2\align_results.csv --notes exp300\results\combined_cv410\error_notes_dev.csv ^
#             --out exp300\results\combined_cv410
# =====================================================================

import argparse
import csv
import itertools
import os
from collections import defaultdict

import torch

import cv2
import numpy as np

import correct_exp300 as corr
import dbscore_exp300 as dbs
import run_exp300 as base
from evaluate_exp300 import drug_metrics
from viz_align_exp300 import Recorder, perm_best
from viz_boxes_exp300 import judge

NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]
DELTAS = [0.0, 0.05, 0.1, 0.2]          # dbscore '차이가 작다' 기준 (글자당 평균 로그확률)
V3_OPT = (True, True, 1.0)              # 부분일치 + 헷갈리는 글자 (보정 V3 와 같음)
ERROR_ORDER = ["각인 같은 약", "마크뿐", "라벨 앞뒤 뒤바뀜", "각인 일부만 보임", "박스 못 찾음", "잘못 읽음"]


# ----------------------------- 색·크기 -----------------------------
def pill_ab(img):
    """원본 사진의 알약 부분 Lab a·b 중앙값 (OpenCV 8비트 Lab: 128 이 무채색)"""
    m = base.segment_pill(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    sel = m > 0 if m is not None else np.ones(img.shape[:2], bool)
    return np.array([np.median(lab[..., 1][sel]), np.median(lab[..., 2][sel])], np.float32)


def single_color(row):
    """한 가지 색 알약이면 그 색 이름, 아니면 None ('초록, 투명' 같은 것도 None)"""
    c1, c2 = (row.get("color1") or "").strip(), (row.get("color2") or "").strip()
    return c1 if c1 and not c2 and "," not in c1 else None


def expected_ab(row, protos):
    """약의 예상 a·b: 한 가지 색이면 그 대표값, 두 색 캡슐이면 두 대표값의 평균 (사진 한 장의 중앙값은 두 색이 섞이므로).
       대표값이 없는 색이 있으면 None"""
    c1, c2 = (row.get("color1") or "").strip(), (row.get("color2") or "").strip()
    cols = [c for c in (c1, c2) if c]
    if not cols or any("," in c or c not in protos for c in cols):
        return None
    return np.mean([protos[c] for c in cols], 0)


def color_protos(items, ab, dbrow, leave_out=None):
    """색 이름 → 개발용 사진에서 실제로 찍힌 a·b 평균 (leave_out 사진은 뺌)"""
    acc = defaultdict(list)
    for it in items:
        if it["file_name"] == leave_out:
            continue
        c = single_color(dbrow[it["drug_N"]])
        if c:
            acc[c].append(ab[it["file_name"]])
    return {c: np.mean(v, 0) for c, v in acc.items()}


def tiebreak(cands, obs_ab, aspect, dbrow, protos):
    """동점 후보 약들 중 색(→ 크기 비율)으로 하나. 정할 수 없으면 None. 돌려주는 값: (약, 근거)"""
    cands = list(dict.fromkeys(cands))
    if len(cands) < 2:
        return (cands[0], "단일") if cands else (None, "")
    exp = [expected_ab(dbrow[d], protos) for d in cands]
    if all(e is not None for e in exp):
        dist = [float(np.linalg.norm(obs_ab - e)) for e in exp]
        best = int(np.argmin(dist))
        if sorted(dist)[1] - dist[best] > 1e-6:
            return cands[best], f"색({dbrow[cands[best]]['color1']})"
    ratios = [float(dbrow[d]["size_ratio"] or 0) for d in cands]          # 색으로 못 가르면 크기 비율
    if all(ratios):
        diff = [abs(aspect - r) for r in ratios]
        order = np.argsort(diff)
        if diff[order[1]] - diff[order[0]] >= 0.1:
            return cands[int(order[0])], "크기"
    return None, ""


# ----------------------------- 점수 -----------------------------
def plain_scores(texts, db):
    """길 A 와 같은 규칙(run_exp300.identify 의 편집거리 유사도, 0.3 이상) → {약: 최고 유사도}. 동점 찾기용"""
    best = {}
    for t in texts:
        if not t:
            continue
        for d in db:
            s = max(base.similarity(t, d["front"]), base.similarity(t, d["back"]))
            if s >= base.MIN_SIM:
                best[d["drug_N"]] = max(best.get(d["drug_N"], 0.0), s)
    return best


def edit_scores(texts, db_rows):
    """글자들(길 A 후보 등) → {약: V3 규칙 최고 유사도}"""
    best = {}
    for t in texts:
        for d, v in corr.drug_scores(t, db_rows, V3_OPT).items():
            best[d] = max(best.get(d, 0.0), v)
    return best


def top_set(score_map, delta=1e-9):
    """1등과 delta 이하로 차이 나는 약들 (1등 포함). 비어 있으면 []"""
    if not score_map:
        return []
    top = max(score_map.values())
    return [d for d, v in sorted(score_map.items(), key=lambda kv: -kv[1]) if top - v <= delta]


def perm_texts(bt):
    texts = [t for _, t in bt]
    if len(texts) < 2:
        return [base.norm("".join(texts))] if texts else []
    order = itertools.permutations(range(len(texts))) if len(texts) <= 4 else [tuple(range(len(texts))),
                                                                              tuple(reversed(range(len(texts))))]
    return [base.norm("".join(texts[i] for i in p)) for p in order]


# ----------------------------- 실행 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True, help="search_db_v3.csv (색·크기 열 포함)")
    ap.add_argument("--viz", required=True, help="viz_align2/align_results.csv (기존 13가지, 같은 표에 넣기)")
    ap.add_argument("--notes", required=True, help="사람이 사진을 보고 적은 오류 원인 기록 (file_name, visual, note)")
    ap.add_argument("--out", default="combined")
    ap.add_argument("--step", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                          # 개발용만
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    db = base.load_db(a.db)
    cdb = corr.load_db(a.db)
    cands = dbs.db_candidates(a.db)
    rec = Recorder(base.PARSeq())
    scorer = dbs.DBScorer(rec.m, [c["text"] for c in cands])
    notes = {r["file_name"]: r for r in csv.DictReader(open(a.notes, encoding="utf-8-sig"))}
    viz = defaultdict(dict)
    for r in csv.DictReader(open(a.viz, encoding="utf-8-sig")):
        viz[r["file_name"]][r["method"]] = r

    # 1) 사진마다 한 번만 계산해 둘 것: 색, 길 A, 모든 후보, dbscore
    ab, per = {}, {}
    for k, it in enumerate(items, 1):
        fn = it["file_name"]
        img = base.imread_any(it["image_path"])
        ab[fn] = pill_ab(img)
        gray, aspect, color, ok = base.crop_pill(img)
        g0 = base.PREPS["P0"][1](gray)
        rec.log = []
        _, _, c12, _ = base.read_with_rotation(g0, aspect, rec, a.step)
        log = list(rec.log)
        bc = base.pick_by_conf(c12)                                         # ② 방향
        gA, boxesA, textsA = log[[x[2] for x in c12].index(bc[2])]
        tA, dA, _ = perm_best([(b, t) for b, t in zip(boxesA, textsA) if t], db)
        if not tA:
            tA, dA = bc[0], base.identify(bc[0], "", "", db)[1]
        nboxA = 0 if (len(boxesA) == 1 and tuple(boxesA[0]) == (0, 0, gA.shape[1], gA.shape[0])) else len(boxesA)
        textsA_all = perm_texts([(b, t) for b, t in zip(boxesA, textsA) if t]) or [bc[0]]
        all_texts = [x for (_, bx, tx) in log for x in (perm_texts([(b, t) for b, t in zip(bx, tx) if t]) or [])]
        all_texts = [t for t in all_texts + [c[0] for c in c12] if 1 <= len(t) <= 12]
        sc, ang_db, nb_db = dbs.score_image(scorer, g0, aspect, a.step)["concat_mean"]
        # 같은 글자(예: 5mg·10mg 앞면 'ARICEPT')는 GPU 계산의 아주 작은 오차로 점수가 미세하게 달라질 수 있음
        # → 글자가 같으면 같은 점수로 맞춰야 '동점'이 제대로 잡힘
        tbest = {}
        for c, s in zip(cands, sc):
            tbest[c["text"]] = max(tbest.get(c["text"], -np.inf), float(s))
        dmap = {}
        for c in cands:
            dmap[c["drug_N"]] = max(dmap.get(c["drug_N"], -np.inf), tbest[c["text"]])
        i1 = int(np.argmax(sc))
        per[fn] = {"it": it, "aspect": aspect, "shape": base.shape_group(aspect), "tA": tA, "dA": dA, "nboxA": nboxA,
                   "editA": plain_scores(textsA_all, db), "v3": edit_scores([c[0] for c in c12 if 1 <= len(c[0]) <= 12], cdb),
                   "v3p": edit_scores(all_texts, cdb), "dmap": dmap, "db_text": cands[i1]["text"],
                   "db_nbox": int(nb_db[i1])}
        chk = viz[fn]["conf_perm"]
        if chk["read"] != tA:
            print(f"  [확인] 길 A 결과가 viz_align2 와 다름: {fn[:24]} {chk['read']} vs {tA}")
        print(f"  {k:2d}/{len(items)} {per[fn]['shape']} {base.norm(it['truth']):<14} A={tA or '-'}  db={per[fn]['db_text']}")

    # 2) 색 대표값이 ARICEPT 를 가르는지 (leave-one-out)
    color_rows = []
    for it in items:
        protos = color_protos(items, ab, dbrow, leave_out=it["file_name"])
        c = single_color(dbrow[it["drug_N"]])
        near = min(protos, key=lambda k: np.linalg.norm(ab[it["file_name"]] - protos[k])) if protos else ""
        color_rows.append({"file_name": it["file_name"], "drug_N": it["drug_N"], "label_color": c or "(두 색/투명)",
                           "a": round(float(ab[it["file_name"]][0]), 1), "b": round(float(ab[it["file_name"]][1]), 1),
                           "nearest_color_LOO": near, "match": int(c == near) if c else ""})

    # 3) 방식별 예측
    def decide_with_color(fn, tied, base_pred, exact=None):
        """tied 중 색(→크기)으로 하나. 묶음 전체로 못 정하면 점수가 완전히 같은 약들(exact)끼리만 다시 시도"""
        p = per[fn]
        protos = color_protos(items, ab, dbrow, leave_out=fn)
        for group in (tied, exact):
            if group and len(group) >= 2:
                d, why = tiebreak(group, ab[fn], p["aspect"], dbrow, protos)
                if d:
                    return d, why
        return base_pred, ""

    def text_then_color_then_db(fn, smap, dl):
        """글자 점수(smap) → 동점·판정없음이면 색 → 그래도 못 정하면 dbscore (C3·C4)"""
        p = per[fn]
        e = corr.decide(smap)
        if e:
            return e, "글자"
        tied = top_set(smap)
        if len(tied) >= 2:
            d, why = decide_with_color(fn, tied, "")
            if d:
                return d, why
        pool = tied or list(p["dmap"])
        sub = {d: p["dmap"][d] for d in pool}
        return max(sub, key=sub.get), "dbscore"

    rows = []
    for fn, p in per.items():
        it = p["it"]
        truth = base.norm(it["truth"])
        out = {}
        out["A"] = (p["dA"], "")
        tied = top_set(p["editA"]) if not p["dA"] else []
        out["A+색"] = decide_with_color(fn, tied, p["dA"]) if tied else (p["dA"], "")
        v3 = corr.decide(p["v3"])
        out["V3"] = (v3, "")
        out["V3+색"] = decide_with_color(fn, top_set(p["v3"]), v3) if not v3 else (v3, "")
        dtop = max(p["dmap"], key=p["dmap"].get)
        for dl in DELTAS:
            tied = top_set(p["dmap"], dl)
            out[f"db"] = (dtop, "")
            out[f"db+색 δ={dl}"] = decide_with_color(fn, tied, dtop, top_set(p["dmap"])) if len(tied) > 1 else (dtop, "")
            # C1: 글자(길 A 모든 후보, V3 규칙) → dbscore → 색
            e = corr.decide(p["v3p"])
            if e:
                out[f"C1 δ={dl}"] = (e, "글자")
            else:
                pool = top_set(p["v3p"]) or list(p["dmap"])
                sub = {d: p["dmap"][d] for d in pool}
                dsub = max(sub, key=sub.get)
                t2 = top_set(sub, dl)
                out[f"C1 δ={dl}"] = decide_with_color(fn, t2, dsub, top_set(sub)) if len(t2) > 1 else (dsub, "dbscore")
            # C2: dbscore → (차이 작으면) 글자 → 색
            if len(tied) == 1:
                out[f"C2 δ={dl}"] = (dtop, "dbscore")
            else:
                sub = {d: p["v3p"].get(d, 0.0) for d in tied}
                t2 = top_set(sub) if max(sub.values()) > 0 else tied
                out[f"C2 δ={dl}"] = (t2[0], "글자") if len(t2) == 1 else decide_with_color(fn, t2, dtop)
        out["C3"] = text_then_color_then_db(fn, p["v3p"], None)          # 길 A 모든 후보(순서 바꾸기 포함) → 색 → dbscore
        out["C4"] = text_then_color_then_db(fn, p["v3"], None)           # 보정 V3 후보 → 색 → dbscore
        for m, (d, why) in out.items():
            rows.append({"file_name": fn, "shape": p["shape"], "camera_la": it.get("camera_la", ""), "method": m,
                         "truth_drug": it["drug_N"], "truth_imprint": truth, "pred_drug": d or NONE_LABEL,
                         "drug_ok": int(d == it["drug_N"]), "decided_by": why,
                         "read": p["tA"] if m.startswith("A") else (p["db_text"] if m.startswith("db") else ""),
                         "judge": judge(p["tA"], truth) if m.startswith("A") else (judge(p["db_text"], truth) if m.startswith("db") else "-")})
    with open(os.path.join(a.out, "combined_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "color_check.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(color_rows[0]))
        w.writeheader()
        w.writerows(color_rows)

    # 4) 기존 13가지 + 새 방식 한 표 (기존은 viz_align2 의 pred_drug 그대로)
    old = ["vote", "conf", "box", "pbox", "pbox_cls", "stroke", "polar", "polar_conf", "cnn", "tps", "conf_perm", "two_conf", "two_db"]
    table_rows = []
    for fn in per:
        for m in old:
            r = viz[fn][m]
            table_rows.append({"file_name": fn, "shape": r["shape"], "method": m, "truth_drug": r["truth_drug"],
                               "pred_drug": r["pred_drug"], "drug_ok": int(r["drug_ok"]), "judge": r["judge"]})
    table_rows += [{k: r[k] for k in ("file_name", "shape", "method", "truth_drug", "pred_drug", "drug_ok", "judge")} for r in rows]
    new = (["A", "A+색", "V3", "V3+색", "db"] + [f"db+색 δ={d}" for d in DELTAS] + [f"C1 δ={d}" for d in DELTAS]
           + [f"C2 δ={d}" for d in DELTAS] + ["C3", "C4"])
    by = {(r["file_name"], r["method"]): r for r in table_rows}
    met = []
    for gname, cond in (("전체", lambda s: True), ("둥근", lambda s: s == "둥근"), ("길쭉", lambda s: s == "길쭉")):
        fs = [fn for fn in per if cond(per[fn]["shape"])]
        print(f"\n[{gname} {len(fs)}장]  (1장 = {100 / len(fs):.1f}%p)   좋아짐/나빠짐: 길 A(conf_perm) 대비 | 다수결 대비")
        print(f"  {'방식':<16} " + " ".join(f"{m:>8}" for m in METRICS) + "    vs길A   vs다수결")
        for m in old + new:
            sub = [by[(fn, m)] for fn in fs]
            acc, pr, rc, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
            imp = np.mean([x["judge"] == "O" for x in sub]) if sub[0]["judge"] != "-" else float("nan")
            ud = lambda ref: (sum(by[(fn, m)]["drug_ok"] and not by[(fn, ref)]["drug_ok"] for fn in fs),
                              sum(not by[(fn, m)]["drug_ok"] and by[(fn, ref)]["drug_ok"] for fn in fs))
            uA, uV = ud("conf_perm"), ud("vote")
            v = [imp, acc, pr, rc, f1]
            print(f"  {m:<16} " + " ".join("     -  " if np.isnan(s) else f"{s * 100:7.1f}%" for s in v) +
                  f"   +{uA[0]}/-{uA[1]}   +{uV[0]}/-{uV[1]}")
            met.append({"group": gname, "n": len(fs), "method": m, **{k: ("" if np.isnan(s) else round(s * 100, 1))
                                                                    for k, s in zip(METRICS, v)},
                        "up_vs_A": uA[0], "down_vs_A": uA[1], "up_vs_vote": uV[0], "down_vs_vote": uV[1]})
    with open(os.path.join(a.out, "combined_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(met[0]))
        w.writeheader()
        w.writerows(met)

    # 5) 남은 오류 분류
    imprint_owner = defaultdict(set)
    for d, r in dbrow.items():
        for col in ("front_norm", "back_norm"):
            t = base.norm(r[col])
            if t:
                imprint_owner[t].add(d)

    def cause(fn, nbox):
        it = per[fn]["it"]
        t = base.norm(it["truth"])
        if len(imprint_owner.get(t, ())) > 1:
            return "각인 같은 약"
        v = notes.get(fn, {}).get("visual", "")
        if v in ("마크뿐", "라벨 앞뒤 뒤바뀜", "각인 일부만 보임"):
            return v
        return "박스 못 찾음" if nbox == 0 else "잘못 읽음"

    err = []
    targets = [("길 A", "conf_perm", lambda fn: per[fn]["nboxA"]), ("dbscore", "db", lambda fn: per[fn]["db_nbox"]),
               ("합친 C1 δ=0.05", "C1 δ=0.05", lambda fn: per[fn]["nboxA"]),
               ("합친 C2 δ=0.05", "C2 δ=0.05", lambda fn: per[fn]["nboxA"]),
               ("합친 C3", "C3", lambda fn: per[fn]["nboxA"]), ("합친 C4", "C4", lambda fn: per[fn]["nboxA"]),
               ("V3+색", "V3+색", lambda fn: per[fn]["nboxA"])]
    print("\n[남은 오류 분류] 개발용 51장에서 틀린 사진 (원인은 앞 칸이 먼저)")
    print(f"  {'방식':<16} " + " ".join(f"{c:>10}" for c in ERROR_ORDER) + "   합계")
    for label, m, nb in targets:
        cnt = defaultdict(int)
        for fn in per:
            if not by[(fn, m)]["drug_ok"]:
                c = cause(fn, nb(fn))
                cnt[c] += 1
                err.append({"method": label, "file_name": fn, "cause": c, "truth_imprint": base.norm(per[fn]["it"]["truth"]),
                            "pred_drug": by[(fn, m)]["pred_drug"], "note": notes.get(fn, {}).get("note", "")})
        print(f"  {label:<16} " + " ".join(f"{cnt[c]:>10}" for c in ERROR_ORDER) + f"   {sum(cnt.values()):4d}")
    with open(os.path.join(a.out, "error_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(err[0]))
        w.writeheader()
        w.writerows(err)

    ari = [r for r in color_rows if r["drug_N"] in ("K-023336", "K-023337")]
    print(f"\n[ARICEPT 색 확인, leave-one-out] 가장 가까운 색 이름이 라벨 색과 같은 사진: "
          f"{sum(r['match'] == 1 for r in ari)}/{len(ari)}장   (a·b: " +
          ", ".join(f"{r['label_color']}({r['a']},{r['b']})" for r in ari) + ")")
    single = [r for r in color_rows if r["match"] != ""]
    print(f"[한 가지 색 알약 전체, leave-one-out] 색 이름 맞힘 {sum(r['match'] for r in single)}/{len(single)}장")
    print(f"\n저장 완료: {a.out}  (combined_dev.csv, combined_metrics.csv, error_table.csv, color_check.csv)")


if __name__ == "__main__":
    main()
