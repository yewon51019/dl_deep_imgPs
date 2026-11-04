# =====================================================================
#  pad_compare_cns.py  —  비율 유지 읽기(ocr_cns.py --pad) 채택 여부 판단 (저장된 결과만 읽음, 개발 468장 + 검증 828장)
# =====================================================================
#
#  [왜]  상자가 정사각형에 가까우면 PARSeq 입력(128×32)에 맞추면서 옆으로 크게 늘어난다 (예: F 215×190 상자는 약 3.5배).
#        가로/세로 < 2 인 조각은 늘리지 않고 양옆을 채워 읽으면 나아지는지 본다.
#  [비교]  기준 pft1 (파인튜닝 PARSeq, 지금 기본) vs pft1_pad (같은 모델·같은 상자 + --pad)
#  [채택 규칙, 결과를 보기 전에 정함]  다음을 모두 만족할 때만 채택
#     ① 개발·검증 모두 약 정확도가 떨어지지 않음  ② 읽기 맞은 장수가 개발 + 검증 합쳐 10장 이상 늘어남
#     ③ 읽기가 좋아진 사진 > 나빠진 사진 (개발 + 검증 합계)
#  [저장]  exp_cns\results\pad_v1\compare.csv, changed.csv  (이미 있으면 멈춤)
#  [실행]  python exp_cns\pad_compare_cns.py   (저장소 맨 위에서, ocr_cns.py --pad 와 match3_cns.py --tag pft1_pad 뒤에)
# =====================================================================

import collections
import csv
import os

import common_cns as cc

OCR = cc.RESULTS / "ocr_v1"
MATCH = cc.RESULTS / "match_v3"
OUT = cc.RESULTS / "pad_v1"
PARTS = {"dev": ("dev", "pred_dev.csv"), "val": ("trainval", "pred_val.csv")}   # 이름 → (ocr 폴더, 약 판별 결과 파일)


def rows(path):
    """csv → {파일명: 한 줄}."""
    return {r["file_name"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def drug_ok(r):
    """약 판별 맞음 여부. 열 이름이 λ·w 에 따라 달라서 '_ok' 로 끝나는 열을 찾음."""
    return r[next(k for k in r if k.endswith("_ok"))] == "1"


def main():
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    os.makedirs(OUT, exist_ok=True)
    table, changed = [], []
    gain = up = down = 0
    drug_fine = True
    for part, (sp, pf) in PARTS.items():
        a = rows(OCR / sp / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv")
        b = rows(OCR / sp / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1_pad" / "pred.csv")
        ma, mb = rows(MATCH / "pft1" / pf), rows(MATCH / "pft1_pad" / pf)
        assert set(a) == set(b) and set(ma) == set(mb), part
        r_a = sum(r["correct"] == "1" for r in a.values())
        r_b = sum(r["correct"] == "1" for r in b.values())
        d_a, d_b = sum(map(drug_ok, ma.values())), sum(map(drug_ok, mb.values()))
        better = [f for f in a if a[f]["correct"] != "1" and b[f]["correct"] == "1"]
        worse = [f for f in a if a[f]["correct"] == "1" and b[f]["correct"] != "1"]
        table.append({"part": part, "photos": len(a), "read_pft1": r_a, "read_pad": r_b, "read_better": len(better), "read_worse": len(worse),
                      "drug_pft1": d_a, "drug_pad": d_b,
                      "drug_better": sum(drug_ok(mb[f]) and not drug_ok(ma[f]) for f in ma),
                      "drug_worse": sum(drug_ok(ma[f]) and not drug_ok(mb[f]) for f in ma),
                      "F_read_pft1": sum(r["correct"] == "1" for r in a.values() if r["truth"] == "F"),
                      "F_read_pad": sum(r["correct"] == "1" for r in b.values() if r["truth"] == "F")})
        gain += r_b - r_a
        up += len(better)
        down += len(worse)
        drug_fine = drug_fine and d_b >= d_a
        for f in better + worse:
            changed.append([part, f, a[f]["drug_N"], a[f]["truth"], "좋아짐" if f in better else "나빠짐",
                            a[f]["read"], b[f]["read"], a[f]["angle"], b[f]["angle"]])
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["part", "file_name", "drug_N", "truth", "change", "read_pft1", "read_pad", "angle_pft1", "angle_pad"])
        w.writerows(changed)
    for t in table:
        print(t)
    print("각인별 변화:", collections.Counter((c[3], c[4]) for c in changed).most_common(16))
    ok = drug_fine and gain >= 10 and up > down
    print(f"읽기 합계 {gain:+d}장, 좋아짐 {up} / 나빠짐 {down}, 약 정확도 안 떨어짐 {drug_fine} → {'채택' if ok else '채택 안 함'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
