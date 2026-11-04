# =====================================================================
#  curved_dirs_sheet_cns.py  —  휜 글자(ARICEPT 면) 사진마다 12방향 읽기와 극좌표 읽기를 한 쪽에 보기 (사후 확인용, 방식 안 바꿈)
# =====================================================================
#
#  [대상]  최종 평가 K-023336·K-023337 ARICEPT 면 24장
#  [한 쪽]  12방향 칸 = 돌린 사진 + 저장된 상자(ft1_u2.0) + 상자별 글자, 칸 제목 = 각도·읽은 글자·확신도 (빨강 = 실제로 고른 방향)
#          맨 아래 = 극좌표로 편 띠 (tps_cns.polar_candidates, 지금 파이프라인에는 없는 시험용) + 확신도 높은 후보 3개
#  [확인]  다시 읽은 방향별 글자가 저장된 all_dirs 와 같은지 사진마다 확인. 읽기는 pft1
#  [저장]  exp_cns\results\test_final_pillset\errors\curved_dirs.pdf  (이미 있으면 덮어쓰지 않고 멈춤, AI Hub 사진이 들어 있으므로 공개하지 않음)
#  [실행]  python exp_cns\curved_dirs_sheet_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import align_long_cns as al
import common_cns as cc
import match_cns as m1
import ocr_cns as ocr
import ocr_common_cns as oc
import rerank_tf_cns as rt
import tps_cns as tp

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = cc.RESULTS / "test_final_pillset" / "errors"
READ = oc.OCR_OUT / "test" / rt.DET / "orig_alignlong_line_pft1"
DRUGS = ("K-023336", "K-023337")
SIZE = 300


def tile(rot, d):
    """돌린 사진에 상자·상자별 글자를 그려 SIZE 정사각형으로."""
    im = rot.copy()
    for (x0, y0, x1, y1), t in zip(d["boxes"], d["texts"]):
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255) if d["fallback"] else (0, 200, 0), 2)
        cv2.putText(im, t or "-", (x0, max(16, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    s = SIZE / max(im.shape[:2])
    return cv2.cvtColor(cc.fit_side(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), SIZE), cv2.COLOR_BGR2RGB)


def main():
    path = OUT / "curved_dirs.pdf"
    if path.exists():
        raise SystemExit(f"이미 있음: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(READ / "pred.csv", encoding="utf-8-sig")) if r["drug_N"] in DRUGS and r["truth"] == "ARICEPT"]
    assert len(rows) == 24, len(rows)
    names = {d["drug_N"]: d["name"] for d in m1.load_db()}
    items = {it["file_name"]: it for it in oc.load_split("test", 0)}
    prep = oc.Prep("orig", "test")
    saved = ocr.SavedBoxes(oc.out_dir("test", rt.DET, "orig", 0, 12, "long") / "boxes.json")
    reader = ocr.Reader(str(rt.WEIGHTS))
    with PdfPages(path) as pdf:
        for k, r in enumerate(rows, 1):
            fn = r["file_name"]
            bgr = cc.imread(items[fn]["image_path"])
            img, _ = prep(items[fn], bgr)
            angs, _ = al.align_angles(bgr)
            dirs = ocr.read_all(img, fn, angs, rt.DET, reader, saved, "tb")
            assert " | ".join(f"{d['ang']}:{d['text']}:{d['conf']:.1f}" for d in dirs) == r["all_dirs"], fn
            cands, strip = tp.polar_candidates(bgr, reader.ps)
            top = sorted(cands, key=lambda c: -c[1])[:3]
            n_row = len(dirs) // 3
            fig = plt.figure(figsize=(8.3, 2.75 * n_row + 1.5))
            gs = fig.add_gridspec(n_row + 1, 3, height_ratios=[2.75] * n_row + [1.2])
            for i, d in enumerate(dirs):
                a = fig.add_subplot(gs[i // 3, i % 3])
                a.imshow(tile(oc.rotate_color(img, d["ang"]), d))
                a.axis("off")
                chosen = d["text"] == r["read"] and abs(float(d["ang"]) - float(r["angle"])) < 1e-6
                a.set_title(f"{d['ang']}°  {d['text'] or '(없음)'}  ({d['conf']:.1f})", fontsize=9.5, color="red" if chosen else "black",
                            fontweight="bold" if chosen else "normal")
            a = fig.add_subplot(gs[n_row, :])
            a.imshow(strip, cmap="gray")
            a.axis("off")
            a.set_title("극좌표로 편 띠 (시험용, 파이프라인에 없음)  후보 위 3개: " + " / ".join(f"{c[0] or '-'} ({c[1]:.2f})" for c in top), fontsize=9)
            fig.suptitle(f"{k}/24  {names[r['drug_N']][:18]}  정답 ARICEPT  카메라 {r['camera_la']}  {fn[:28]}", fontsize=10.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {k}/24 {fn[:28]} 극좌표: {[c[0] for c in top]}", flush=True)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
