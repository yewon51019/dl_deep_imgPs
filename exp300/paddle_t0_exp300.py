# =====================================================================
#  paddle_t0_exp300.py  —  PaddleOCR 을 PARSeq 최종 방식(보정 V3 + 색 동점 깨기)과 같은 절차로 개발용 51장에서만 비교
# =====================================================================
#
#  [사전 기록]  CLAUDE.md 6장: PaddleOCR 대표 설정 = paddleocr_t0 (임계값 3개 전부 0.0), 기본 설정은 비교용. 평가용은 돌리지 않음.
#  [절차]  read   : 둥근 알약 12방향(30°) / 길쭉 0°·180° 로 읽은 후보를 전부 저장 (run_exp300.read_with_rotation 그대로)
#                   · paddleocr_t0 × P0, P1   · paddleocr(기본) × P0  (확인용: 방향마다 박스 수·읽은 글자 수도 기록)
#          report : 저장된 후보에 보정 V3 + 색 동점 깨기를 PARSeq 때와 같은 코드(final_exp300.v3_color, 같은 색 대표값)로 적용
#                   → PARSeq 최종 방식과 나란히 표 (정답률·정확도·정밀도·민감도·F1, 전체/둥근/길쭉, 판정없음, 시간), 좋아짐·나빠짐
#  [색 대표값]  개발용 채점은 자기 사진을 뺀 개발용(leave-one-out), 즉 PARSeq 개발용 결과와 같은 규칙. 평가용 사진은 안 씀.
#  [저장]  --out: cands_{설정}_{전처리}.csv (방향별 후보·박스 수), results_dev.csv, metrics.csv, compare_parseq.csv,
#                 updown.csv (좋아진·나빠진 사진 파일 이름), box_check.csv (박스 수·읽은 글자 수 요약)
#          viz    : PaddleOCR 검출 박스 그림(dt). 각 설정에서 보정 V3 + 색이 고른 방향으로 한 번 더 읽어 박스·글자를 그림
#                   → --viz-out 폴더 (원본 사진 포함 → GitHub 금지), paddle_boxes_dev.csv (박스 좌표·글자)
#  [이어하기]  read 는 이미 저장된 사진은 건너뜀
#  [실행 예]  python exp300\paddle_t0_exp300.py read   --manifest ... --out exp300\results\paddle_t0_cv410
#             python exp300\paddle_t0_exp300.py report --manifest ... --db exp300\results\search_db_v3.csv ... --out exp300\results\paddle_t0_cv410
# =====================================================================

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

import torch                                         # Paddle 보다 먼저 (Windows DLL 충돌 방지)

import cv2
import numpy as np

import combined_exp300 as cb
import correct_exp300 as corr
import final_exp300 as fin
import run_exp300 as base
import run_exp300_paddle as rp
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

RUNS = [("t0", "P0"), ("t0", "P1"), ("default", "P0")]       # (설정, 전처리)
MODEL = {"t0": rp.PaddleOCRModelT0, "default": rp.PaddleOCRModel}
NAMES = {"t0": "paddleocr_t0", "default": "paddleocr(기본)"}
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]
STEP = 30


class Counting:
    """PaddleOCR 모델을 감싸서 호출마다 (검출 박스 수, 읽은 조각 수, 글자가 있는 조각 수)를 기록. 읽는 결과는 그대로"""
    def __init__(self, model):
        self.m, self.log = model, []
        orig = model.ocr.predict

        def spy(*a, **k):
            out = list(orig(*a, **k))
            spy.last = out
            return out
        spy.last = []
        self.spy = spy
        model.ocr.predict = spy

    def __call__(self, g):
        self.spy.last = []
        t, c = self.m(g)
        n_dt = n_rec = n_al = 0
        for res in self.spy.last:
            dt = res.get("dt_polys")
            n_dt += 0 if dt is None else len(dt)
            rt = list(res.get("rec_texts") or [])
            n_rec += len(rt)
            n_al += sum(any(ch.isalnum() for ch in str(x)) for x in rt)
        self.log.append((n_dt, n_rec, n_al))
        return t, c


def cands_path(out, setting, prep):
    return os.path.join(out, f"cands_{setting}_{prep}.csv")


def do_read(a):
    os.makedirs(a.out, exist_ok=True)
    items = base.load_manifest(a.manifest, "dev")                         # 개발용만
    if a.limit:
        items = items[: a.limit]
    crops = {it["file_name"]: base.crop_pill(base.imread_any(it["image_path"])) for it in items}
    cols = ["setting", "prep", "file_name", "split", "shape", "aspect", "ms", "vote_pred", "vote_conf", "vote_angle", "cands"]
    for setting, prep in RUNS:
        path = cands_path(a.out, setting, prep)
        done = set()
        if os.path.exists(path):
            done = {r["file_name"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
        todo = [it for it in items if it["file_name"] not in done]
        if not todo:
            print(f"[{NAMES[setting]} × {prep}] 이미 완료 ({len(done)}장)")
            continue
        t0 = time.time()
        cnt = Counting(MODEL[setting]())
        print(f"[{NAMES[setting]} × {prep}] 모델 불러옴 ({time.time() - t0:.0f}s), {len(todo)}장 읽는 중")
        new = not os.path.exists(path)
        f = open(path, "a", newline="", encoding="utf-8-sig")
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for k, it in enumerate(todo, 1):
            gray, aspect, color, ok = crops[it["file_name"]]
            g = base.PREPS[prep][1](gray)
            cnt.log = []
            t1 = time.time()
            pred, conf, cands, ang = base.read_with_rotation(g, aspect, cnt, STEP)
            ms = (time.time() - t1) * 1000
            w.writerow({"setting": setting, "prep": prep, "file_name": it["file_name"], "split": it["split"],
                        "shape": base.shape_group(aspect), "aspect": round(aspect, 4), "ms": round(ms), "vote_pred": pred,
                        "vote_conf": round(conf, 1), "vote_angle": ang,
                        "cands": json.dumps([[c[0], round(c[1], 1), c[2], *n] for c, n in zip(cands, cnt.log)], ensure_ascii=False)})
            f.flush()
            if k % 10 == 0 or k == len(todo):
                print(f"   {k}/{len(todo)}  ({time.time() - t0:.0f}s)  최근 '{pred}' / 정답 '{base.norm(it['truth'])}'")
        f.close()
    print("읽기 완료 →", a.out)


# ----------------------------- 채점 -----------------------------
def scores(rows):
    acc, p, r, f = drug_metrics([x["truth_drug"] for x in rows], [x["pred_drug"] for x in rows])
    return [float(np.mean([x["judge"] == "O" for x in rows])), acc, p, r, f]


def prepare(a):
    """채점·그림에 같이 쓰는 준비물: 개발용 목록, DB, 색(자기 사진 뺀 대표값), 보정 V3 + 색 결정 함수"""
    items = base.load_manifest(a.manifest, "dev")
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    db, cdb = base.load_db(a.db), corr.load_db(a.db)
    ab = {it["file_name"]: cb.pill_ab(base.imread_any(it["image_path"])) for it in items}
    protos = {it["file_name"]: cb.color_protos(items, ab, dbrow, leave_out=it["file_name"]) for it in items}
    info = {it["file_name"]: it for it in items}

    def decide(fn, aspect, texts):
        v3map = cb.edit_scores([t for t in texts if 1 <= len(t) <= 12], cdb)
        return fin.v3_color(v3map, ab[fn], aspect, dbrow, protos[fn])
    return items, dbrow, db, cdb, info, decide


def chosen(cl, d, cdb):
    """보정 V3 + 색이 고른 방향: 정한 약에 최고점을 준 후보의 방향, 판정 없음이면 확신도 최고 방향 → (읽은 글자, 각도)"""
    bc = fin.best_candidate(cl, d, cdb) if d else None
    if bc:
        return bc
    best = base.pick_by_conf(cl)
    return "", best[2]


def do_viz(a):
    """PaddleOCR 검출 박스 그림 (dt): 각 설정이 고른 방향에서 한 번 더 읽어 박스·글자를 그림. 개발용만"""
    from viz_align_exp300 import cell, fit
    from viz_boxes_exp300 import JCOL, ascii_only, paddle_detect
    items, dbrow, db, cdb, info, decide = prepare(a)
    parseq = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    vdir = a.viz_out
    os.makedirs(vdir, exist_ok=True)
    crops = {it["file_name"]: base.crop_pill(base.imread_any(it["image_path"])) for it in items}
    order = [it["file_name"] for it in items]
    cells = defaultdict(dict)
    box_rows = []
    color = {"t0": (220, 90, 30), "default": (30, 140, 240)}
    for setting, prep in RUNS:
        rows = {r["file_name"]: r for r in csv.DictReader(open(cands_path(a.out, setting, prep), encoding="utf-8-sig"))}
        model = MODEL[setting]()
        t0 = time.time()
        for fn in order:
            r, it = rows[fn], info[fn]
            cl = [(c[0], c[1], c[2]) for c in json.loads(r["cands"])]
            d, why = decide(fn, float(r["aspect"]), [c[0] for c in cl])
            read, ang = chosen(cl, d, cdb)
            g = base.rotate(base.PREPS[prep][1](crops[fn][0]), ang)
            boxes, texts, polys, pinfo = paddle_detect(model, g)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            col = color[setting]
            for p in polys:
                cv2.polylines(img, [np.round(p).astype(np.int32)], True, col, 2)
            for bx, t in zip(boxes, texts):
                cv2.putText(img, ascii_only(t)[:12], (int(bx[0]), max(14, int(bx[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, 2, cv2.LINE_AA)
            truth = base.norm(it["truth"])
            j = judge(read, truth)
            ok = d == it["drug_N"]
            head = f"{NAMES[setting].replace('(기본)', '(default)')} x {prep}  ang={ang:g}  boxes={pinfo['n_dt']}"
            cells[fn][(setting, prep)] = (cell(fit(img), head, col,
                                               [(f"[{j}] read: {ascii_only(read) or '(none)'}", JCOL[j]),
                                                (f"drug {ascii_only(d) or '(none)'} {'OK' if ok else 'NG'}",
                                                 (0, 150, 0) if ok else (0, 0, 200))]), d, ok, pinfo["n_dt"], j)
            box_rows.append({"setting": NAMES[setting], "prep": prep, "file_name": fn, "angle": ang, "n_dt": pinfo["n_dt"],
                             "n_text": pinfo["n_text"], "texts": "|".join(texts), "read": read, "pred_drug": d or NONE_LABEL,
                             "drug_ok": int(ok), "boxes": json.dumps([[round(v) for v in b] for b in boxes])})
        print(f"[{NAMES[setting]} × {prep}] 박스 그림 {len(order)}장 ({time.time() - t0:.0f}s)")
    files = []
    for k, fn in enumerate(order, 1):
        it = info[fn]
        g0 = base.PREPS["P0"][1](crops[fn][0])
        first = cell(fit(cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)), "cropped pill (P0, 0deg)", (60, 60, 60),
                     [(f"truth: {ascii_only(base.norm(it['truth']))}", (60, 60, 60)), (it["drug_N"], (90, 90, 90))])
        grid = np.hstack([first] + [cells[fn][rp_][0] for rp_ in RUNS])
        summ = np.full((grid.shape[0], 380, 3), 255, np.uint8)
        cv2.rectangle(summ, (0, 0), (379, grid.shape[0] - 1), (190, 190, 190), 1)
        lines = [("SUMMARY (V3 + color tie)", (0, 0, 0)), (f"truth: {ascii_only(base.norm(it['truth']))} ({it['drug_N']})", (60, 60, 60))]
        p = parseq[fn]
        lines.append((f"PARSeq (final): {'OK' if p['drug_ok'] == '1' else 'NG'}  read {ascii_only(p['read']) or '-'}",
                      (0, 150, 0) if p["drug_ok"] == "1" else (0, 0, 200)))
        for rp_ in RUNS:
            _, d, ok, nd, j = cells[fn][rp_]
            lines.append((f"{NAMES[rp_[0]].replace('(기본)', '(default)')} x {rp_[1]}: {'OK' if ok else 'NG'}  boxes {nd}",
                          (0, 150, 0) if ok else (0, 0, 200)))
        for i, (t, c) in enumerate(lines):
            cv2.putText(summ, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2 if i == 0 else 1, cv2.LINE_AA)
        grid = np.hstack([grid, summ])
        bar = np.full((34, grid.shape[1], 3), 245, np.uint8)
        cv2.putText(bar, f"{k:02d} dev {it['drug_N']} {'round' if crops[fn][1] <= 1.3 else 'long'} rot={it.get('rot_deg', '')} "
                         f"cam={it.get('camera_la', '')}  PaddleOCR boxes at the angle chosen by V3 + color tie",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        path = os.path.join(vdir, f"paddle_{k:02d}_{it['drug_N']}_rot{it.get('rot_deg', '')}_cam{it.get('camera_la', '')}.png")
        cv2.imencode(".png", np.vstack([bar, grid]))[1].tofile(path)
        files.append(path)
    from viz_two_final_exp300 import sheets
    sheets(files, vdir, "paddle")
    with open(os.path.join(a.out, "paddle_boxes_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(box_rows[0]))
        w.writeheader()
        w.writerows(box_rows)
    print(f"저장 완료: {vdir} (사진별 PNG {len(files)}장, sheet_paddle_*.png), {os.path.join(a.out, 'paddle_boxes_dev.csv')}")


def do_report(a):
    items, dbrow, db, cdb, info, decide = prepare(a)

    # PARSeq 최종 방식 (저장된 개발용 결과) + 같은 코드로 다시 계산해 일치하는지 확인
    parseq = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    pv = {r["file_name"]: r for r in csv.DictReader(open(a.parseq_vote, encoding="utf-8-sig"))}
    mism = 0
    for fn, it in info.items():
        aspect = base.crop_pill(base.imread_any(it["image_path"]))[1]
        d, _ = decide(fn, aspect, json.loads(pv[fn]["candidates"]))
        mism += (d or NONE_LABEL) != parseq[fn]["pred_drug"]
    print(f"[확인] 같은 코드를 PARSeq 후보에 적용 → 저장된 PARSeq V3+색 결과와 다른 사진 {mism}/{len(info)}장")

    res, runs_meta = [], {}
    for fn, r in parseq.items():
        res.append({"setting": "PARSeq", "prep": "P0", "method": "V3+색", "file_name": fn, "shape": r["shape"],
                    "truth_drug": r["truth_drug"], "pred_drug": r["pred_drug"], "read": r["read"], "judge": r["judge"],
                    "decided_by": r["decided_by"], "drug_ok": r["drug_ok"], "ms": ""})
    pms = {r["file_name"]: float(r["ms"]) for r in csv.DictReader(open(a.parseq_ms, encoding="utf-8-sig")) if r["method"] == "vote"}
    for setting, prep in RUNS:
        p = cands_path(a.out, setting, prep)
        if not os.path.exists(p):
            print("  (없음)", p)
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        for r in rows:
            fn, aspect = r["file_name"], float(r["aspect"])
            cl = [(c[0], c[1], c[2]) for c in json.loads(r["cands"])]
            it = info[fn]
            truth = base.norm(it["truth"])
            d, why = decide(fn, aspect, [c[0] for c in cl])
            bc = fin.best_candidate(cl, d, cdb) if d else None
            read = bc[0] if bc else ""
            # 다수결 (Paddle 자체의 지금 방식)
            dv = base.identify(r["vote_pred"], "", "", db)[1]
            common = {"setting": NAMES[setting], "prep": prep, "file_name": fn, "shape": r["shape"], "truth_drug": it["drug_N"],
                      "ms": float(r["ms"])}
            res.append({**common, "method": "V3+색", "pred_drug": d or NONE_LABEL, "read": read, "judge": judge(read, truth),
                        "decided_by": why, "drug_ok": int(d == it["drug_N"])})
            res.append({**common, "method": "다수결", "pred_drug": dv or NONE_LABEL, "read": r["vote_pred"],
                        "judge": judge(r["vote_pred"], truth), "decided_by": "다수결", "drug_ok": int(dv == it["drug_N"])})
        nd = [json.loads(r["cands"]) for r in rows]
        dirs = [c for x in nd for c in x]
        runs_meta[(setting, prep)] = {
            "setting": NAMES[setting], "prep": prep, "photos": len(rows), "reads": len(dirs),
            "reads_0_box": sum(c[3] == 0 for c in dirs), "reads_0_text": sum(c[5] == 0 for c in dirs),
            "mean_boxes": round(float(np.mean([c[3] for c in dirs])), 2), "mean_texts": round(float(np.mean([c[5] for c in dirs])), 2),
            "photos_all_0_box": sum(all(c[3] == 0 for c in x) for x in nd), "photos_all_0_text": sum(all(c[5] == 0 for c in x) for x in nd),
            "photos_vote_empty": sum(r["vote_pred"] == "" for r in rows)}

    def write(name, rows):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)

    write("results_dev.csv", res)
    write("box_check.csv", list(runs_meta.values()))

    # 방식별 지표 표
    groups = [("전체", lambda s: True), ("둥근", lambda s: s == "둥근"), ("길쭉", lambda s: s == "길쭉")]
    keys = [("PARSeq", "P0", "V3+색"), ("paddleocr_t0", "P0", "V3+색"), ("paddleocr_t0", "P1", "V3+색"),
            ("paddleocr(기본)", "P0", "V3+색"), ("PARSeq", "P0", "다수결"), ("paddleocr_t0", "P0", "다수결"),
            ("paddleocr_t0", "P1", "다수결"), ("paddleocr(기본)", "P0", "다수결")]
    pv_rows = []
    for fn, r in pv.items():                                                  # PARSeq 다수결 (저장된 개발용 결과)
        pv_rows.append({"setting": "PARSeq", "prep": "P0", "method": "다수결", "file_name": fn, "shape": "둥근" if r["det_shape"] == "둥근" else "길쭉",
                        "truth_drug": r["truth_drug"], "pred_drug": r["pred_edit"] or NONE_LABEL, "read": r["pred_imprint"],
                        "judge": judge(r["pred_imprint"], r["truth_imprint"]), "drug_ok": int((r["pred_edit"] or "") == r["truth_drug"]),
                        "decided_by": "다수결", "ms": ""})
    allr = res + pv_rows
    met = []
    print("\n[개발용 51장: PARSeq 최종 방식 vs PaddleOCR]  (보정 V3 + 색 동점 깨기, 같은 코드·같은 색 대표값)")
    for gname, cond in groups:
        n = len({x["file_name"] for x in allr if cond(x["shape"])})
        print(f"\n  ({gname} {n}장, 1장 = {100 / n:.1f}%p)")
        print(f"    {'설정 × 전처리 × 방식':<34} " + " ".join(f"{k:>8}" for k in METRICS) + "  판정없음  읽기시간/장")
        for s, pr, m in keys:
            sub = [x for x in allr if x["setting"] == s and x["prep"] == pr and x["method"] == m and cond(x["shape"])]
            if not sub:
                continue
            v = scores(sub)
            none = sum(x["pred_drug"] == NONE_LABEL for x in sub)
            ms = [float(x["ms"]) for x in sub if x["ms"] != ""] if s != "PARSeq" else [pms[x["file_name"]] for x in sub]
            sec = np.mean(ms) / 1000
            print(f"    {s + ' × ' + pr + ' × ' + m:<34} " + " ".join(f"{q * 100:7.1f}%" for q in v) + f"  {none:6d}   {sec:6.2f}초")
            met.append({"group": gname, "n": n, "setting": s, "prep": pr, "method": m,
                        **{k: round(q * 100, 1) for k, q in zip(METRICS, v)}, "판정없음": none, "읽기_초_장": round(float(sec), 2)})
    write("metrics.csv", met)

    # 좋아짐/나빠짐 (PARSeq 최종 방식 대비)
    ud = []
    base_ok = {x["file_name"]: int(x["drug_ok"]) for x in res if x["setting"] == "PARSeq"}
    print("\n[PARSeq 최종 방식(V3+색) 대비 좋아짐 / 나빠짐 사진 수]")
    for s, pr, m in keys[1:4]:
        sub = {x["file_name"]: x for x in res if x["setting"] == s and x["prep"] == pr and x["method"] == m}
        up = sorted(f for f, x in sub.items() if x["drug_ok"] == 1 and not base_ok[f])
        dn = sorted(f for f, x in sub.items() if x["drug_ok"] == 0 and base_ok[f])
        print(f"  {s} × {pr}: 좋아짐 {len(up)}장 / 나빠짐 {len(dn)}장")
        for f in up:
            print(f"      + {f}  ({info[f]['drug_N']}, 정답 {base.norm(info[f]['truth'])}) Paddle '{sub[f]['read'] or '-'}'")
        for f in dn:
            print(f"      - {f}  ({info[f]['drug_N']}, 정답 {base.norm(info[f]['truth'])}) Paddle '{sub[f]['read'] or '-'}' → {sub[f]['pred_drug']}")
        ud += [{"setting": s, "prep": pr, "change": "좋아짐", "file_name": f, "truth_drug": info[f]["drug_N"],
                "paddle_read": sub[f]["read"], "paddle_pred": sub[f]["pred_drug"]} for f in up]
        ud += [{"setting": s, "prep": pr, "change": "나빠짐", "file_name": f, "truth_drug": info[f]["drug_N"],
                "paddle_read": sub[f]["read"], "paddle_pred": sub[f]["pred_drug"]} for f in dn]
    write("updown.csv", ud)

    print("\n[박스 수·읽은 글자 수 확인 (방향별 읽기 기준)]")
    print(f"  {'설정 × 전처리':<22} {'읽기':>5} {'박스 0개':>8} {'글자 0개':>8} {'평균 박스':>8} {'평균 글자 조각':>11} {'모든 방향 박스 0개 사진':>20} {'다수결 빈 결과':>12}")
    for m in runs_meta.values():
        print(f"  {m['setting'] + ' × ' + m['prep']:<22} {m['reads']:5d} {m['reads_0_box']:8d} {m['reads_0_text']:8d} {m['mean_boxes']:8.2f} "
              f"{m['mean_texts']:11.2f} {m['photos_all_0_box']:14d}/{m['photos']:<5d} {m['photos_vote_empty']:8d}/{m['photos']}")
    print(f"\n저장 완료: {a.out}  (results_dev.csv, metrics.csv, updown.csv, box_check.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "report", "viz"])
    ap.add_argument("--viz-out", default="", help="viz: 박스 그림 폴더 (원본 사진 포함, GitHub 금지)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", default="")
    ap.add_argument("--parseq", default="", help="test_final/final_dev.csv (PARSeq 최종 방식 개발용 결과)")
    ap.add_argument("--parseq-vote", default="", help="pred_dev_parseq_P0_align_vote_cv410.csv (후보 텍스트·다수결)")
    ap.add_argument("--parseq-ms", default="", help="viz_align2/align_results.csv (PARSeq 읽기 시간)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="시험용(read): 앞에서 N장만")
    a = ap.parse_args()
    {"read": do_read, "report": do_report, "viz": do_viz}[a.stage](a)


if __name__ == "__main__":
    main()
