# =====================================================================
#  match4_cns.py  —  약 판별 v2 + '고른 방향이 큰 상자면 그 후보 점수 × μ'
# =====================================================================
#
#  [왜]  큰 상자(알약 전체, 넓이 ≥ 400×400 의 절반)를 받은 PARSeq 는 'PHOTOGRAPHY' 같은 단어를 지어내는데, 확신도가 높아 고른 방향이 됨
#        (방향 고르기에서 깎는 방법은 검증·개발 모두 손해 → big_pick_cns.py, P = 0). 그래서 약 판별에서만 그 후보를 덜 믿음
#  [점수]  T = max(μ_사진 · T_고른 방향, λ · T_다른 방향)  (μ_사진 = μ if 고른 방향 상자 중 큰 상자 있음 else 1), 점수 = T + w·A
#  [고르기]  μ × λ × w 를 검증(trainval 828장) 약 정확도로 (같으면 μ 큰 쪽 → λ 큰 쪽 → w 작은 쪽), 개발용 468장은 고른 조합 하나로 측정만
#  [저장]  exp_cns\results\match_v4\  grid_val.csv, pred_val.csv, pred_dev.csv, summary.csv
#  [실행]  python exp_cns\match4_cns.py
# =====================================================================

import csv
import json
import os

import match_cns as m1
import match2_cns as m2

OCR = m1.oc.OCR_OUT
OUT = m1.cc.RESULTS / "match_v4"
MUS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]
BIG = 0.5 * 400 * 400
PRED = {"dev": OCR / "dev" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv",
        "val": OCR / "trainval" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"}


def predict(part, db, configs):
    """configs = [(이름, λ, μ)] → [(행, {이름_wA_w: (1등, 1~3등)})]"""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs)
    out = []
    for k, (fn, r) in enumerate(sorted(ocr.items()), 1):
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        big = any((b[2] - b[0]) * (b[3] - b[1]) >= BIG for b in json.loads(r["best_boxes"] or "[]"))
        cm, co = m2.split_candidates(r, m2.MAX_LEN)
        tm, to = m2.text_scores(cm, db, False), m2.text_scores(co, db, False)
        res = {}
        for name, lam, mu in configs:
            T = m2.combine({d: v * (mu if big else 1) for d, v in tm.items() if v * (mu if big else 1) >= m1.MIN_SIM}, to, lam)
            for w in m1.W_GRID:
                res[f"{name}_wA_{w:g}"] = m1.decide({d: T.get(d, 0.0) + w * A[d] for d in A})
        r["big"] = int(big)
        out.append((r, res))
        if k % 200 == 0:
            print(f"  {part} {k}/{len(ocr)}")
    return out


def main():
    db = m1.load_db()
    m1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    grid = [(f"M{mu:g}_L{lam:g}", lam, mu) for mu in MUS for lam in m2.LAMBDAS]
    val = predict("val", db, grid)
    rows = [dict(mu=mu, lam=lam, w=w, acc=m1.metrics(val, f"{n}_wA_{w:g}")["acc"]) for n, lam, mu in grid for w in m1.W_GRID]
    with open(OUT / "grid_val.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    best = max(rows, key=lambda x: (x["acc"], x["mu"], x["lam"], -x["w"]))
    mu, lam, w = best["mu"], best["lam"], best["w"]
    print(f"val 에서 고름: μ = {mu:g}, λ = {lam:g}, w = {w:g} (val {best['acc']}) / μ = 1 최고 "
          f"{max(x['acc'] for x in rows if x['mu'] == 1.0)}")
    print("  μ 별 val 최고:", {f"{m:g}": max(x["acc"] for x in rows if x["mu"] == m) for m in MUS})
    print(f"  큰 상자 사진: val {sum(r['big'] for r, _ in val)}장")
    dev = predict("dev", db, [("best", lam, mu), ("base", 0.95, 1.0)])
    print(f"  큰 상자 사진: dev {sum(r['big'] for r, _ in dev)}장")
    out_rows = []
    for part, o, keys in (("val", val, [f"M{mu:g}_L{lam:g}_wA_{w:g}", "M1_L0.95_wA_0.02"]),
                          ("dev", dev, [f"best_wA_{w:g}", "base_wA_0.02"])):
        m1.save_pred(part, o, keys)
        for k in keys:
            mm = m1.metrics(o, k)
            out_rows.append(dict(part=part, method=k, **mm))
            print(f"  {part:3s} {k:22s} 정확도 {mm['acc']:5.1f}  3등안 {mm['top3']:5.1f}  판정없음 {mm['none']:3d}  약단위 {mm['drug_acc']:5.1f}  "
                  f"긴 {mm['long']:5.1f} 그외 {mm['other']:5.1f}")
        a, b = keys
        print(f"    좋아짐 {sum(r['drug_N'] == x[a][0] and r['drug_N'] != x[b][0] for r, x in o)} / "
              f"나빠짐 {sum(r['drug_N'] != x[a][0] and r['drug_N'] == x[b][0] for r, x in o)}")
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)


if __name__ == "__main__":
    main()
