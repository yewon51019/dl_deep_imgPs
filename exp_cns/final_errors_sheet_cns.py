# =====================================================================
#  final_errors_sheet_cns.py  —  최종 평가 주 방식 오답 73장의 dt 시트 (사후 확인용, 방식 안 바꿈, OCR 안 돌림)
# =====================================================================
#
#  [대상]  test_final_pillset\pred_test.csv 에서 주 방식(main)이 맞히지 못한 사진 (틀림 + 판정 없음)
#          순서 = ARICEPT 먼저, 그다음 약별. 한 쪽에 5장, 세로로 쌓아 휴대폰에서 보기 쉽게
#  [그림]  ocr_cns 가 저장한 pft1 dt 그림 (원본 | 고른 방향 + 상자·글자 | 정답·결과) + 정답·읽기·답·방향별 읽기
#  [저장]  exp_cns\results\test_final_pillset\errors\  final_errors.pdf, sheet_NN.png  (이미 있으면 덮어쓰지 않고 멈춤, AI Hub 사진이 들어 있으므로 공개하지 않음)
#  [실행]  python exp_cns\final_errors_sheet_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common_cns as cc
import match_cns as m1
import ocr_common_cns as oc

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = cc.RESULTS / "test_final_pillset"
OUT = BASE / "errors"
READ = oc.OCR_OUT / "test" / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1"
PER = 5
FIRST = {"K-023336": 0, "K-023337": 1}


def main():
    if (OUT / "final_errors.pdf").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    byid = {d["drug_N"]: d for d in m1.load_db()}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(READ / "pred.csv", encoding="utf-8-sig"))}
    rows = [r for r in csv.DictReader(open(BASE / "pred_test.csv", encoding="utf-8-sig")) if r["main"] != r["drug_N"]]
    rows.sort(key=lambda r: (FIRST.get(r["drug_N"], 9), r["drug_N"], r["file_name"]))
    n = (len(rows) + PER - 1) // PER
    with PdfPages(OUT / "final_errors.pdf") as pdf:
        for s in range(n):
            cur = rows[s * PER:(s + 1) * PER]
            fig, ax = plt.subplots(PER, 1, figsize=(8.3, 2.95 * PER))
            for i, a in enumerate(ax):
                a.axis("off")
                if i >= len(cur):
                    continue
                r = cur[i]
                o = ocr[r["file_name"]]
                im = cc.imread(str(READ / "dt" / r["file_name"]))
                a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                answer = byid[r["main"]]["name"][:18] if r["main"] else "판정 없음"
                dirs = " ".join(f"{d.split(':')[0]}:{d.split(':')[1] or '-'}" for d in o["all_dirs"].split(" | "))
                faces = "/".join(byid[r["drug_N"]]["imprints"])
                title = (f"#{s * PER + i + 1} {byid[r['drug_N']]['name'][:22]}  정답 {r['truth']} (두 면 {faces})\n"
                         f"읽음 {r['read_pft1'] or '(없음)'}  → 답 {answer}{'  [TF3]' if r['tf3_used'] == '1' else ''}  카메라 {r['camera']}\n"
                         + "\n".join(textwrap.wrap(dirs, 82)))
                a.set_title(title, fontsize=8.5, loc="left")
            fig.suptitle(f"최종 평가 오답 (주 방식) {s + 1}/{n}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            fig.savefig(OUT / f"sheet_{s + 1:02d}.png", dpi=90)
            plt.close(fig)
    print(f"{len(rows)}장, {n}쪽, 저장: {OUT}")
    print("--- ARICEPT 읽기")
    for r in rows:
        if r["drug_N"] in FIRST:
            o = ocr[r["file_name"]]
            print(r["file_name"][:30], "| 면", r["truth"], "| 읽음", r["read_pft1"], "| 상자", o["n_boxes"], "| 방향", o["all_dirs"])


if __name__ == "__main__":
    main()
