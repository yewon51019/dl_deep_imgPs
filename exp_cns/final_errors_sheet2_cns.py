# =====================================================================
#  final_errors_sheet2_cns.py  —  최종 방식(pft1) 오답 사진만 모은 시트 PDF (틀림 + 판정 없음, 사후 확인용, OCR 안 돌림)
# =====================================================================
#
#  [평가·개발]  final_method_dt 그림 (원본 | 고른 방향 + 상자 | 읽기 판정 | 약 판별) + 약 이름·읽기·답·방향별 읽기, 한 쪽에 5장, 틀림 먼저
#  [검증]      저장된 dt 가 없어 match_v7\errors_pft1\val_sheet_NN.png 를 그대로 PDF 로 묶음 (180장)
#  [저장]      exp_cns\results\final_method_dt\errors\  errors_test.pdf, errors_dev.pdf, errors_val.pdf
#              이미 있으면 덮어쓰지 않고 멈춤. AI Hub 사진이 들어 있으므로 공개하지 않음
#  [실행]      python exp_cns\final_errors_sheet2_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv
import io
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import common_cns as cc
import compare_pft1_cns as cp
import match_cns as m1
import test_final2_cns as tf2

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RES = cc.RESULTS
OUT = RES / "final_method_dt" / "errors"
PER = 5


def load(part):
    """(사진 이름, 답) 목록과 읽기 줄."""
    if part == "test":
        ans = {r["file_name"]: r["pft1"] for r in csv.DictReader(open(RES / "test_final_pillset" / "pred_test.csv", encoding="utf-8-sig"))}
        path = tf2.READ["pft1"]
    else:
        path, saved, col = cp.CFG["pft1"][1]["dev"]
        ans = {r["file_name"]: r[col] for r in csv.DictReader(open(saved, encoding="utf-8-sig"))}
    return ans, {r["file_name"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def save_pdf(pages, path):
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=110, quality=82)
    print(f"저장 {path} {len(pages)}쪽 {path.stat().st_size // 1024} KB", flush=True)


def sheets(part, byid):
    ans, ocr = load(part)
    rows = [ocr[f] for f, a in ans.items() if a != ocr[f]["drug_N"]]
    rows.sort(key=lambda r: (ans[r["file_name"]] == "", r["drug_N"], r["file_name"]))
    n = (len(rows) + PER - 1) // PER
    pages = []
    name = {"test": "최종 평가", "dev": "개발"}[part]
    for s in range(n):
        cur = rows[s * PER:(s + 1) * PER]
        fig, ax = plt.subplots(PER, 1, figsize=(8.6, 2.75 * PER))
        for i, a in enumerate(ax):
            a.axis("off")
            if i >= len(cur):
                continue
            r = cur[i]
            im = cc.imread(str(RES / "final_method_dt" / part / r["file_name"]))
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            p = ans[r["file_name"]]
            dirs = " ".join(f"{d.split(':')[0]}:{d.split(':')[1] or '-'}" for d in r["all_dirs"].split(" | "))
            a.set_title(f"#{s * PER + i + 1} {byid[r['drug_N']]['name'][:22]}  정답 {r['truth']} (두 면 {'/'.join(byid[r['drug_N']]['imprints'])})\n"
                        f"읽음 {r['read'] or '(없음)'}  → {'틀림: ' + byid[p]['name'][:18] if p else '판정 없음'}  카메라 {r['camera_la']}\n"
                        + "\n".join(textwrap.wrap(dirs, 86)), fontsize=8.2, loc="left")
        fig.suptitle(f"최종 방식 오답 ({name}, {len(rows)}장) {s + 1}/{n}", fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, format="png")
        plt.close(fig)
        buf.seek(0)
        pages.append(Image.open(buf).convert("RGB"))
    print(part, "오답", len(rows), "틀림", sum(bool(ans[r["file_name"]]) for r in rows), flush=True)
    save_pdf(pages, OUT / f"errors_{part}.pdf")


def main():
    if (OUT / "errors_test.pdf").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    byid = {d["drug_N"]: d for d in m1.load_db()}
    for part in ("test", "dev"):
        sheets(part, byid)
    val = sorted((RES / "match_v7" / "errors_pft1").glob("val_sheet_*.png"))
    save_pdf([Image.open(p).convert("RGB") for p in val], OUT / "errors_val.pdf")


if __name__ == "__main__":
    main()
