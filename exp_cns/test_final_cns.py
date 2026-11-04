# =====================================================================
#  test_final_cns.py  —  평가용 최종 채점 (CLAUDE.md 10장 4-0 사전 기록 그대로, 한 번만)
#  [평가용]  exp300test = exp300/data 의 AI Hub 테스트셋 두 폴더 중 정답 각인이 있는 288장·26종 (마크뿐 12장 제외)
#            (pill_image_testset 은 쓰지 않음 — 사용자 지정, 2026-11-02). 카메라는 60° / 90° (출력의 cam75 칸 = 60°)
# =====================================================================
#
#  [주 방식]  읽기 = ocr_v1\test\paddle_ft1_u2.0\orig_alignlong_line (ft1 검출 u2.0 + 축 정렬 + 줄 묶기 + 사전학습 PARSeq, 확신도 방향)
#            약 판별 = match2 방식 λ 0.95, 길이 ≤ 30, 반대 방향 부분일치 끔, T + 0.02·A, DB match_v1\search_db_cns.csv
#  [비교용]  각인만(v1 text: λ 1, 길이 ≤ 14) · 각인 + CNN 동점 깨기(v1 text_tie) · v1 (λ 1, 길이 ≤ 14, w 0.02)
#            · 파인튜닝 PARSeq 읽기(orig_alignlong_line_pft1) + match2 (검증에서 고른 λ 0.9, w 0.02)
#            → 숫자는 모두 미리 정한 값, 평가용으로 고르는 것 없음
#  [지표]  읽기 완전일치, 약 정확도, 3등 안, 판정없음, 약 단위, macro 정밀도·재현율·F1 (평가용 36종 기준, 판정없음 = 예측 없음),
#          전체 / 긴 알약 / 그 외 / 카메라 75·90, 개발용(같은 방식)과 나란히
#  [저장]  exp_cns\results\test_final\  pred_test.csv, summary.csv, per_drug.csv
#  [실행]  python exp_cns\test_final_cns.py
# =====================================================================

import collections
import csv
import os

import match_cns as m1
import match2_cns as m2

OCR = m1.oc.OCR_OUT
SPLIT = "exp300test"
OUT = m1.cc.RESULTS / "test_final_exp300"
READ = {"base": OCR / SPLIT / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv",
        "pft1": OCR / SPLIT / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv"}
DEV = {"main": (m1.cc.RESULTS / "match_v3" / "base" / "pred_dev.csv", "best_wA_0.02"),
       "pft1": (m1.cc.RESULTS / "match_v3" / "pft1" / "pred_dev.csv", "best_wA_0.02"),
       "v1_text": (m1.cc.RESULTS / "match_v1" / "pred_dev.csv", "text"),
       "v1_tie": (m1.cc.RESULTS / "match_v1" / "pred_dev.csv", "text_tie"),
       "v1": (m1.cc.RESULTS / "match_v1" / "pred_dev.csv", "wA_0.02")}
# 이름 → (읽기, 설정 이름, 열 이름)
METHODS = {"main": ("base", "main", "main_wA_0.02"),
           "v1_text": ("base", "v1", "v1_text"),
           "v1_tie": ("base", "v1", "v1_tie"),
           "v1": ("base", "v1", "v1_wA_0.02"),
           "pft1": ("pft1", "pft1", "pft1_wA_0.02")}
NAMES = {"main": "주 방식 (v2 base)", "v1_text": "각인만", "v1_tie": "각인 + CNN 동점 깨기", "v1": "v1 (λ 1, 길이 ≤ 14)",
         "pft1": "파인튜닝 PARSeq + v2"}


def macro(pairs, classes):
    """pairs = [(정답, 예측)] → macro 정밀도·재현율·F1 (%) — 정답 약 classes 마다, 예측 없는 약의 정밀도는 0."""
    P, R, F = [], [], []
    for c in classes:
        tp = sum(t == c and p == c for t, p in pairs)
        npred = sum(p == c for _, p in pairs)
        ntrue = sum(t == c for t, _ in pairs)
        pr = tp / npred if npred else 0.0
        rc = tp / ntrue if ntrue else 0.0
        P.append(pr)
        R.append(rc)
        F.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    n = len(classes)
    return round(100 * sum(P) / n, 1), round(100 * sum(R) / n, 1), round(100 * sum(F) / n, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    db = m1.load_db()                                                # match_v1\search_db_cns.csv (평가용 36종은 xlsx 줄로 들어 있음)
    m1.OUT = OUT
    res = {}
    for read, cfgs in (("base", [("main", 0.95, False, 30), ("v1", 1.0, False, 14)]), ("pft1", [("pft1", 0.9, False, 30)])):
        m1.PRED[SPLIT] = READ[read]
        res[read] = m2.predict_all(SPLIT, db, cfgs)
    rows, per = [], []
    test_drugs = sorted({r["drug_N"] for r, _ in res["base"]})
    assert all(d in {x["drug_N"] for x in db} for d in test_drugs), "평가용 약이 DB 에 없음"
    for k, (read, _, key) in METHODS.items():
        out = res[read]
        mm = m1.metrics(out, key)
        c60 = [r["drug_N"] == x[key][0] for r, x in out if r["camera_la"] == "60"]
        mm["cam75"] = round(100 * sum(c60) / max(len(c60), 1), 1)    # 이 세트는 60° / 90° → cam75 칸에 60° 를 넣음
        mp, mr, mf = macro([(r["drug_N"], x[key][0]) for r, x in out], test_drugs)
        rows.append(dict(part=SPLIT, method=NAMES[k], **mm, macro_P=mp, macro_R=mr, macro_F1=mf))
        f, dk = DEV[k]
        dv = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        dpairs = [(r["drug_N"], r[f"{dk}_pred"]) for r in dv]
        dp, dr, df = macro(dpairs, sorted({r["drug_N"] for r in dv}))
        rows.append(dict(part="dev", method=NAMES[k], acc=round(100 * sum(r[f"{dk}_ok"] == "1" for r in dv) / len(dv), 1),
                         none=sum(r[f"{dk}_pred"] == "" for r in dv), macro_P=dp, macro_R=dr, macro_F1=df, n=len(dv)))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        keys = list(rows[0])
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['part']:4s} {r['method']:22s} 정확도 {r['acc']:5.1f}  판정없음 {r['none']:3d}  "
              f"P/R/F1 {r['macro_P']:5.1f}/{r['macro_R']:5.1f}/{r['macro_F1']:5.1f}"
              + (f"  3등안 {r['top3']:5.1f}  약단위 {r['drug_acc']:5.1f}  긴 {r['long']:5.1f} 그외 {r['other']:5.1f}  "
                 f"60° {r['cam75']:5.1f} 90° {r['cam90']:5.1f}  읽기 {r['ocr_exact']:5.1f}" if r["part"] == SPLIT else ""))
    # 사진마다 (주 방식 + 비교용 예측)
    m1.save_pred(SPLIT, [(r, {**x, **y}) for (r, x), (_, y) in zip(res["base"], res["pft1"])],
                 ["main_wA_0.02", "v1_text", "v1_tie", "v1_wA_0.02", "pft1_wA_0.02"])
    # 약별 (주 방식)
    name = {d["drug_N"]: d["name"] for d in db}
    by = collections.defaultdict(list)
    for r, x in res["base"]:
        by[r["drug_N"]].append((r, x["main_wA_0.02"][0]))
    with open(OUT / "per_drug.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["drug_N", "name", "photos", "correct", "none", "wrong", "ocr_exact", "long", "truths", "wrong_preds"])
        for d in test_drugs:
            v = by[d]
            w.writerow([d, name[d], len(v), sum(p == d for _, p in v), sum(p == "" for _, p in v), sum(p not in ("", d) for _, p in v),
                        sum(r["correct"] == "1" for r, _ in v), v[0][0]["aligned"], "|".join(sorted({r["truth"] for r, _ in v})),
                        "|".join(sorted({name.get(p, p)[:16] for _, p in v if p not in ("", d)}))])
    print(f"저장: {OUT} (summary.csv, pred_test.csv, per_drug.csv)")


if __name__ == "__main__":
    main()
