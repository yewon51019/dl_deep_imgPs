# =====================================================================
#  clean_table_cns.py  —  개발용 성능표: 전체 / 데이터 오류 제외 / + 구분 불가 약 제외
# =====================================================================
#
#  [빼는 사진]  (개발용 오답 원인 분류 3-7 에서 그림으로 확인)
#     데이터 오류 (48장): K-000147 사진·라벨 불일치 24 (사진 15 / CGP, 라벨 BT / KS) · K-035766 라벨 앞뒤 뒤바뀜 24 (앞면 사진에 CKD, 뒷면 사진에 EO05)
#     구분 불가 (12장): K-000040 — 라벨은 맞지만 YH + 하양 원형 정제가 DB 에 3종 (K-000397·K-000465) → 어떤 방식도 판정없음
#  [표]  방식마다 읽기 완전일치(DB 없음) · 약 정확도(DB 검색) · 3등 안 · 판정없음 · 긴 알약 / 그 외
#  [저장]  exp_cns\results\match_v3\clean_table.csv
#  [실행]  python exp_cns\clean_table_cns.py
# =====================================================================

import csv

import common_cns as cc

R = cc.RESULTS
OCR = R / "ocr_v1" / "dev" / "paddle_ft1_u2.0"
DATA_ERR = {"K-000147", "K-035766"}
AMBIG = {"K-000040"}
# 방식 이름 → (약 판별 pred 파일, 열 이름 앞부분, 읽기 pred 폴더)
METHODS = [
    ("v1 각인만", R / "match_v1" / "pred_dev.csv", "text", "orig_alignlong_line"),
    ("v1 각인 + CNN 동점 깨기", R / "match_v1" / "pred_dev.csv", "text_tie", "orig_alignlong_line"),
    ("v1 각인 + 0.02 겉모습", R / "match_v1" / "pred_dev.csv", "wA_0.02", "orig_alignlong_line"),
    ("v2 base (주 방식)", R / "match_v3" / "base" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line"),
    ("파인튜닝 PARSeq", R / "match_v3" / "pft1" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line_pft1"),
    ("판별기로 방향", R / "match_v3" / "porient" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line_porient"),
    ("파인튜닝 + 판별기", R / "match_v3" / "pft1_porient" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line_pft1_porient"),
    ("확실할 때만 바꾸기", R / "match_v3" / "pgate" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line_pgate"),
    ("파인튜닝 + 확실할 때만", R / "match_v3" / "pft1_pgate" / "pred_dev.csv", "best_wA_0.02", "orig_alignlong_line_pft1_pgate"),
]
SETS = [("전체", set()), ("데이터 오류 제외", DATA_ERR), ("+ 구분 불가 제외", DATA_ERR | AMBIG)]


def main():
    rows = []
    for name, f, key, od in METHODS:
        mp = {r["file_name"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
        ocr = {r["file_name"]: r for r in csv.DictReader(open(OCR / od / "pred.csv", encoding="utf-8-sig"))}
        for sname, drop in SETS:
            ks = [k for k, r in mp.items() if r["drug_N"] not in drop]
            pct = lambda xs: round(100 * sum(xs) / max(len(xs), 1), 1)
            ok = {k: mp[k][f"{key}_ok"] == "1" for k in ks}
            rows.append(dict(
                method=name, set=sname, n=len(ks),
                ocr_exact=pct([ocr[k]["correct"] == "1" for k in ks]),
                drug_acc=pct(list(ok.values())),
                top3=pct([mp[k]["drug_N"] in [x.rsplit(":", 1)[0] for x in mp[k][f"{key}_top3"].split()] for k in ks]),
                none=sum(mp[k][f"{key}_pred"] == "" for k in ks),
                long=pct([ok[k] for k in ks if ocr[k]["aligned"] == "1"]),
                other=pct([ok[k] for k in ks if ocr[k]["aligned"] != "1"]),
                n_long=sum(ocr[k]["aligned"] == "1" for k in ks)))
    out = R / "match_v3" / "clean_table.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['method']:22s} {r['set']:12s} n={r['n']:3d}  읽기 {r['ocr_exact']:5.1f}  약 {r['drug_acc']:5.1f}  3등안 {r['top3']:5.1f}  "
              f"판정없음 {r['none']:3d}  긴 {r['long']:5.1f} 그외 {r['other']:5.1f} (긴 {r['n_long']})")
    print("저장:", out)


if __name__ == "__main__":
    main()
