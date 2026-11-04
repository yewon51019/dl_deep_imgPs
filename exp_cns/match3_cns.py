# =====================================================================
#  match3_cns.py  —  약 판별 v2 방식(match2_cns.py)을 다른 읽기 결과에 그대로 적용 (1·2단계 비교용)
# =====================================================================
#
#  [방식]  match2 와 같음: T = max(T_고른 방향, λ·T_다른 방향), 길이 제한 30, 반대 방향 부분일치 끔, 점수 = T + w·A
#          λ(LAMBDAS) × w(W_GRID) 를 검증 부분(trainval 828장)에서 정확도로 고름 (같으면 λ 큰 쪽·w 작은 쪽), 개발용 468장은 측정만
#  [읽기]  --tag 마다 ocr_v1 의 dev / trainval pred.csv  (base 는 v2 와 같은 파일: 개발용 기존 결과 + trainall 결과)
#  [저장]  exp_cns\results\match_v3\<tag>\  pred_val.csv, pred_dev.csv, summary.csv
#          exp_cns\results\match_v3\compare.csv  (tag 마다 한 줄: 읽기 정답률·약 정확도 등, 실행할 때마다 덧붙임)
#  [실행]  python exp_cns\match3_cns.py --tag pft1_porient
# =====================================================================

import argparse
import csv
import os

import match_cns as m1
import match2_cns as m2

OCR = m1.oc.OCR_OUT
TAGS = {   # tag → (개발용 결과 폴더 끝, 검증 결과 split·폴더 끝)
    "base": ("_line", ("trainall", "_line")),
    "base_tv": ("_line", ("trainval", "_line")),
    "pft1": ("_line_pft1", ("trainval", "_line_pft1")),
    "porient": ("_line_porient", ("trainval", "_line_porient")),
    "pft1_porient": ("_line_pft1_porient", ("trainval", "_line_pft1_porient")),
    "pgate": ("_line_pgate", ("trainval", "_line_pgate")),                 # orient_gate_cns.py (δ 는 검증에서 고름)
    "pft1_pgate": ("_line_pft1_pgate", ("trainval", "_line_pft1_pgate")),
    "pft1_pad": ("_line_pft1_pad", ("trainval", "_line_pft1_pad")),        # ocr_cns.py --pad (정사각형에 가까운 조각을 늘리지 않음)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(TAGS))
    a = ap.parse_args()
    dev_tail, (vsplit, vtail) = TAGS[a.tag]
    m1.PRED = {"dev": OCR / "dev" / "paddle_ft1_u2.0" / f"orig_alignlong{dev_tail}" / "pred.csv",
               "val": OCR / vsplit / "paddle_ft1_u2.0" / f"orig_alignlong{vtail}" / "pred.csv"}
    out = m2.OUT.parent / "match_v3" / a.tag
    os.makedirs(out, exist_ok=True)
    db = m1.load_db()                                                # DB = match_v1\search_db_cns.csv (m1.OUT 를 바꾸기 전에 읽음)
    m1.OUT = out                                                     # save_pred 는 이 폴더로
    grid = [(f"L{lam:g}", lam, False, m2.MAX_LEN) for lam in m2.LAMBDAS]
    val = m2.predict_all("val", db, grid)
    best = max(((lam, w, m1.metrics(val, f"L{lam:g}_wA_{w:g}")["acc"]) for _, lam, _, _ in grid for w in m1.W_GRID),
               key=lambda x: (x[2], x[0], -x[1]))
    lam, w, vacc = best
    print(f"[{a.tag}] val 에서 고름: λ = {lam:g}, w = {w:g} (val 정확도 {vacc})")
    dev = m2.predict_all("dev", db, [("best", lam, False, m2.MAX_LEN)])
    rows = []
    for part, o, key in (("val", val, f"L{lam:g}_wA_{w:g}"), ("dev", dev, f"best_wA_{w:g}")):
        m1.save_pred(part, o, [key])
        mm = m1.metrics(o, key)
        rows.append(dict(tag=a.tag, part=part, lam=lam, w=w, **mm))
        print(f"  {part:3s} 정확도 {mm['acc']:5.1f}  3등안 {mm['top3']:5.1f}  판정없음 {mm['none']:3d}  약단위 {mm['drug_acc']:5.1f}  "
              f"긴 {mm['long']:5.1f} 그외 {mm['other']:5.1f}  75° {mm['cam75']:5.1f} 90° {mm['cam90']:5.1f}  (읽기 완전일치 {mm['ocr_exact']})")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    cmp_f = out.parent / "compare.csv"
    new = not cmp_f.exists()
    with open(cmp_f, "a", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
