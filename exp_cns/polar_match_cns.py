# =====================================================================
#  polar_match_cns.py  —  기존 점수가 불확실한 사진에만 극좌표 후보를 더해 약 판별 (사후 보조 실험)
# =====================================================================
#
#  [흐름]  base 약 판별 (match2 방식, λ 0.95, 길이 ≤ 30, w 0.02) → 불확실하면 극좌표 후보를 더해 같은 계산을 다시
#  [실행 전에 고정한 규칙, 결과를 보고 바꾸지 않음 (2026-11-02)]
#     불확실 = 판정 없음 또는 1등 점수 < 0.95 또는 1등 − 2등 < 0.1
#     극좌표 후보 = tps_cns.polar_candidates 중 3~30 글자, 각인 점수에 0.95 를 곱함 (base 의 '다른 방향' 후보와 같은 값)
#     확실한 사진은 답을 바꾸지 않음
#  [대상]  개발용 468장 (불확실한 사진만 극좌표로 읽음) + 싱귤레어 K-011307 SINGULAIR 면 12장 (검증 부분, 개발용과 합치지 않고 따로)
#          싱귤레어는 극좌표 읽기 결과를 이미 본 사진이라 참고 숫자. 평가용 사진은 읽지 않음
#  [보고]  개발용은 전체 / K-000147 (사진·라벨 불일치, 사용자 결정으로 제외) 뺀 값을 함께
#  [저장]  exp_cns\results\tps_v1\polar_match\  pred_dev.csv, pred_singulair.csv, summary.csv,
#          dt\ (극좌표로 읽은 사진마다 한 장. 파일 이름 앞에 better_ / worse_ / changed_ / same_. 사진이 들어 있어 .gitignore)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\polar_match_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv
import json
import time

import tps_cns                                   # torch 를 먼저 불러옴. polar_candidates, base(run_exp300)
import cv2
import numpy as np

import match2_cns as m2
import match_cns as m1
from correct_exp300 import sim_v

cc = tps_cns.cc
base = tps_cns.base
OUT = tps_cns.OUT / "polar_match"
LAM, MAX_LEN, W = 0.95, 30, 0.02                 # 사전 기록한 주 방식 값
SURE_TOP, SURE_GAP = 0.95, 0.1                   # 확실 기준 (실행 전 고정)
POLAR_LAM, POLAR_MIN = 0.95, 3                   # 극좌표 후보 점수 곱, 최소 글자 수 (실행 전 고정)
SAVED = {"dev": ("best_wA_0.02_pred", cc.RESULTS / "match_v3" / "base" / "pred_dev.csv"),
         "val": ("L0.95_wA_0.02_pred", cc.RESULTS / "match_v3" / "base" / "pred_val.csv")}


def load(part, keep=lambda r: True):
    """(OCR 줄, 겉모습 확률) 목록. match2.predict_all 과 같은 파일."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = [r for r in csv.DictReader(open(m1.PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs and keep(r)]
    return sorted(((r, probs[r["file_name"]]) for r in ocr), key=lambda x: x[0]["file_name"])


def final_scores(T, A):
    return {d: T.get(d, 0.0) + W * A[d] for d in A}


def save_dt(path, img, strip, r, base_pred, new_pred, base_top, new_top, used):
    """DT 한 장: 사진 | 펼친 띠 + 정답·읽기·답·후보 (그림 글자는 영문만, 약은 번호로)."""
    canvas = np.full((400, 1400, 3), 255, np.uint8)
    canvas[50:350, 0:300] = cv2.resize(img, (300, 300))
    canvas[10:70, 320:1390] = cv2.cvtColor(cv2.resize(strip, (1070, 60)), cv2.COLOR_GRAY2BGR)
    fmt = lambda top: "  ".join(f"{d}:{v:.3f}" for d, v in top)
    color = lambda p: (0, 140, 0) if p == r["drug_N"] else (0, 0, 220)
    lines = [(f"{r['file_name']}", (0, 0, 0)),
             (f"answer drug {r['drug_N']}  label imprint {r['truth']}  base read {r['read']}", (0, 0, 0)),
             (f"BASE pred {base_pred or 'none'}", color(base_pred)), (f"  top3 {fmt(base_top)}", (90, 90, 90)),
             (f"POLAR pred {new_pred or 'none'}", color(new_pred)), (f"  top3 {fmt(new_top)}", (90, 90, 90)),
             (f"polar text behind pred: {used or '-'}", (0, 0, 0))]
    for k, (s, c) in enumerate(lines):
        cv2.putText(canvas, s, (320, 100 + k * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 1, cv2.LINE_AA)
    cc.imwrite(str(path), canvas)


def run(items, part, db, paths, reader, dt_dir):
    col, saved_path = SAVED[part]
    saved = {r["file_name"]: r[col] for r in csv.DictReader(open(saved_path, encoding="utf-8-sig"))}
    rows, t0 = [], time.time()
    for k, (r, prob) in enumerate(items, 1):
        fn = r["file_name"]
        A = {d["drug_N"]: m1.attr_score(prob, d) for d in db}
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM)
        pred, top = m1.decide(final_scores(T, A))
        assert pred == saved[fn], (fn, pred, saved[fn])       # base 저장 결과를 그대로 재현하는지 사진마다 확인
        ranked = m1.rank(final_scores(T, A))
        s1, gap = ranked[0][1], ranked[0][1] - ranked[1][1]
        unsure = pred == "" or s1 < SURE_TOP or gap < SURE_GAP
        new_pred, new_top, polar_c, used = pred, top, [], ""
        if unsure:
            img = cc.imread(paths[fn])
            cands, strip = tps_cns.polar_candidates(img, reader)
            polar_c = sorted({c[0] for c in cands if POLAR_MIN <= len(c[0]) <= MAX_LEN})
            T2 = m2.combine(T, m2.text_scores(polar_c, db, False), POLAR_LAM)
            new_pred, new_top = m1.decide(final_scores(T2, A))
            if new_pred:
                imps = next(d["imprints"] for d in db if d["drug_N"] == new_pred)
                sc = [(max((sim_v(c, i, *m1.OPT) for i in imps), default=0.0), c) for c in polar_c]
                if sc:
                    used = max(sc)[1]
        ok0, ok1 = pred == r["drug_N"], new_pred == r["drug_N"]
        kind = "better" if ok1 and not ok0 else "worse" if ok0 and not ok1 else "changed" if new_pred != pred else "same"
        rows.append({"file_name": fn, "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"], "aligned": r["aligned"], "camera_la": r["camera_la"],
                     "top1": round(s1, 4), "gap": round(gap, 4), "unsure": int(unsure), "base_pred": pred, "base_ok": int(ok0),
                     "base_top3": " ".join(f"{d}:{v:.3f}" for d, v in top), "polar_pred": new_pred, "polar_ok": int(ok1),
                     "polar_top3": " ".join(f"{d}:{v:.3f}" for d, v in new_top), "kind": kind, "polar_used": used, "polar_cands": json.dumps(polar_c)})
        if unsure:
            save_dt(dt_dir / f"{kind}_{fn}", img, strip, r, pred, new_pred, top, new_top, used)
        if k % 50 == 0:
            print(f"  {part} {k}/{len(items)}  {time.time() - t0:.0f}초", flush=True)
    return rows


def summarize(name, rs):
    n = len(rs)
    pct = lambda v: round(100 * v / max(n, 1), 1)
    in3 = lambda r, key: r["drug_N"] in r[key]
    s = {"group": name, "n": n, "unsure": sum(r["unsure"] for r in rs),
         "base_correct": sum(r["base_ok"] for r in rs), "base_acc": pct(sum(r["base_ok"] for r in rs)),
         "polar_correct": sum(r["polar_ok"] for r in rs), "polar_acc": pct(sum(r["polar_ok"] for r in rs)),
         "better": sum(r["kind"] == "better" for r in rs), "worse": sum(r["kind"] == "worse" for r in rs),
         "changed_still_wrong": sum(r["kind"] == "changed" for r in rs),
         "base_none": sum(r["base_pred"] == "" for r in rs), "polar_none": sum(r["polar_pred"] == "" for r in rs),
         "base_top3": pct(sum(in3(r, "base_top3") for r in rs)), "polar_top3": pct(sum(in3(r, "polar_top3") for r in rs))}
    return s


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    dt_dir = OUT / "dt"
    dt_dir.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    paths = {r["file_name"]: r["image_path"] for r in csv.DictReader(open(m1.ATTR / "labels.csv", encoding="utf-8-sig"))}
    reader = base.PARSeq()
    dev = run(load("dev"), "dev", db, paths, reader, dt_dir)
    write_csv(OUT / "pred_dev.csv", dev)
    sing = run(load("val", lambda r: r["drug_N"] == "K-011307" and r["truth"] == "SINGULAIR"), "val", db, paths, reader, dt_dir)
    write_csv(OUT / "pred_singulair.csv", sing)
    rest = [r for r in dev if r["drug_N"] != "K-000147"]
    table = [summarize("dev 전체", dev), summarize("dev K-000147 제외", rest),
             summarize("dev 긴 알약", [r for r in dev if r["aligned"] == "1"]),
             summarize("dev 그 외", [r for r in dev if r["aligned"] != "1"]),
             summarize("싱귤레어 SINGULAIR 면 (참고)", sing)]
    write_csv(OUT / "summary.csv", table)
    for s in table:
        print(s)
    for r in dev + sing:
        if r["kind"] != "same":
            print(r["kind"], r["file_name"], r["truth"], "| base", r["base_pred"], "→", r["polar_pred"], "| 후보", r["polar_used"])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
