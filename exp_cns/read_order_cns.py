# =====================================================================
#  read_order_cns.py  —  순수 텍스트 정답률 + '상자 순서를 바꾸면 정답과 같아지는' 사진 세기 (저장된 pred.csv 만 읽음)
# =====================================================================
#
#  [왜]  글자는 다 맞게 읽었는데 상자를 잇는 순서만 뒤집혀 틀린 사진이 있다 (예: DASRC ↔ SRCDA, 180° 돌아간 분할 각인).
#  [지표]  exact      = 읽은 글자가 정답과 완전일치 (지금까지의 순수 텍스트 정답률)
#          order_free = exact 이거나, 고른 방향의 상자가 2~4개일 때 상자 글자의 어떤 순서가 정답과 완전일치
#                       (상자 수 제한은 약 판별 candidates 와 같음)
#  [주의]  정답을 보고 순서를 고르므로 '실제로 낼 수 있는 답'이 아니라 '글자 자체는 맞게 읽었는가'를 보는 보조 지표다.
#          약 판별(match2)의 후보에는 순서 바꾸기가 이미 들어 있어 약 정확도는 이 지표로 바뀌지 않는다.
#  [저장]  exp_cns\results\ocr_v1\dev\read_order_compare.csv        (조건별 표, 기준 대비 좋아짐/나빠짐)
#          exp_cns\results\ocr_v1\dev\read_order_only.csv           (순서만 틀린 사진 목록)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\read_order_cns.py   (저장소 맨 위에서)
# =====================================================================

import csv
import itertools
import json
import os

import common_cns as cc

OCR = cc.RESULTS / "ocr_v1"
DET = "paddle_ft1_u2.0"
K0 = "K0L17-c3-Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12_auto_link17_low85"
K6 = "K0L17R6-q2-Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12_auto_link17_low85_rim0.6"
# 조건 이름 → 결과 폴더 (첫 줄 = 기준 base)
RUNS = {"기준: 원본 박스 + 원본 읽기": "orig_alignlong_line",
        "K0L17 박스 + 원본 읽기": "orig_alignlong_boxK0L17-c_line",
        "K0L17R6 박스 + 원본 읽기": "orig_alignlong_boxK0L17R6_line",
        "K0L17 박스 + K0L17 읽기": f"{K0}_alignlong_line",
        "K0L17R6 박스 + K0L17R6 읽기": f"{K6}_alignlong_line"}


def order_free(r):
    """상자 글자의 어떤 순서가 정답과 같으면 True (상자 2~4개만, exact 포함)."""
    if r["correct"] == "1":
        return True
    pieces = [b[4] for b in json.loads(r["best_boxes"] or "[]") if b[4]]
    return 2 <= len(pieces) <= 4 and any("".join(p) == r["truth"] for p in itertools.permutations(pieces))


def main():
    out_cmp, out_only = OCR / "dev" / "read_order_compare.csv", OCR / "dev" / "read_order_only.csv"
    for p in (out_cmp, out_only):
        assert not os.path.exists(p), f"이미 있음, 덮어쓰지 않음: {p}"
    data = {k: list(csv.DictReader(open(OCR / "dev" / DET / d / "pred.csv", encoding="utf-8-sig"))) for k, d in RUNS.items()}
    base_name = next(iter(RUNS))
    base = {r["file_name"]: r for r in data[base_name]}
    rows, only = [], []
    for k, rs in data.items():
        assert len(rs) == 468, (k, len(rs))
        ex = sum(r["correct"] == "1" for r in rs)
        fr = sum(order_free(r) for r in rs)
        rows.append(dict(condition=k, n=len(rs), exact=ex, exact_pct=round(100 * ex / len(rs), 1), order_free=fr,
                         order_free_pct=round(100 * fr / len(rs), 1), order_only=fr - ex,
                         exact_better=sum(r["correct"] == "1" and base[r["file_name"]]["correct"] == "0" for r in rs),
                         exact_worse=sum(r["correct"] == "0" and base[r["file_name"]]["correct"] == "1" for r in rs),
                         free_better=sum(order_free(r) and not order_free(base[r["file_name"]]) for r in rs),
                         free_worse=sum(not order_free(r) and order_free(base[r["file_name"]]) for r in rs),
                         box0=sum(r["fallback"] == "1" for r in rs)))
        only += [dict(condition=k, file_name=r["file_name"], drug_N=r["drug_N"], truth=r["truth"], read=r["read"], boxes=r["best_boxes"],
                      angle=r["angle"], camera=r["camera_la"]) for r in rs if r["correct"] == "0" and order_free(r)]
    with open(out_cmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_only, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(only[0]))
        w.writeheader()
        w.writerows(only)
    for r in rows:
        print(f"  {r['condition']:30s} 완전일치 {r['exact_pct']:5.1f}% ({r['exact']})  순서 무시 {r['order_free_pct']:5.1f}% ({r['order_free']})  "
              f"순서만 틀림 {r['order_only']:3d}  기준 대비 완전 +{r['exact_better']}/−{r['exact_worse']}, 순서 무시 +{r['free_better']}/−{r['free_worse']}  박스0 {r['box0']}")
    print(f"저장: {out_cmp}\n      {out_only}")


if __name__ == "__main__":
    main()
