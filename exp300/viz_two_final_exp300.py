# =====================================================================
#  viz_two_final_exp300.py  —  최종 두 방식(보정 V3 + 색 / C4)만 모은 결과 그림 (OCR 다시 안 돌림)
# =====================================================================
#   final_exp300.py 가 만든 사진별 비교 그림(photos_dev / photos_test)에서
#   [잘라낸 알약 | 보정 V3 + 색 | C4] 칸만 잘라 붙이고, 위에 정답·두 방식의 판정을 다시 적어요.
#   평가용은 이미 한 번 돌린 결과(final_test.csv)와 그림만 쓰므로 다시 실행하지 않아요.
#  [저장]  --out 폴더 (원본 사진 포함 → GitHub 금지, viz_* 라서 .gitignore 에 걸림)
#    dev\, test\          : 사진별 PNG (파일 이름: 번호_약코드_rot_cam_V3판정_C4판정.png)
#    test_differ\         : 두 방식의 답이 다른 사진만 — 개발용 2장(dev_...) + 평가용 13장(test_...) (V3+색 '판정 없음' → C4 가 dbscore 로 고른 것)
#    sheet_dev_N.png, sheet_test_N.png, sheet_differ_N.png : 6장씩 모은 그림 (differ = 개발용+평가용 15장)
#  [실행 예]  python exp300\viz_two_final_exp300.py --final exp300\results\test_final ^
#             --photos exp300\results\viz_final_report --out exp300\results\viz_final_two
# =====================================================================

import argparse
import csv
import glob
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

import run_exp300 as base
from viz_boxes_exp300 import ascii_only

BAR, CELL_W, CELL_H = 34, 376, 442       # final_exp300.py 그림의 머리줄 높이, 칸 너비·높이
COLS = {"crop": 0, "V3+색": 3, "C4": 4}   # 칸 순서: 잘라낸 알약 | 다수결 | 길 A | V3+색 | C4


def compose(im, k, split, r3, r4):
    parts = [im[BAR:BAR + CELL_H, COLS[c] * CELL_W:(COLS[c] + 1) * CELL_W] for c in ("crop", "V3+색", "C4")]
    grid = np.hstack(parts)
    bar = np.full((52, grid.shape[1], 3), 245, np.uint8)
    t = f"{k:03d} {split}  {r3['truth_drug']}  {'round' if r3['shape'] == '둥근' else 'long'}  rot={r3['rot_deg']}  cam={r3['camera_la']}  truth={ascii_only(r3['truth_imprint'])}"
    cv2.putText(bar, t, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    by = {"글자": "text", "판정없음": "none", "dbscore": "dbscore", "크기": "size"}
    for x, name, r in ((CELL_W, "V3+color (MAIN)", r3), (2 * CELL_W, "C4", r4)):
        why = by.get(r["decided_by"], "color" if r["decided_by"].startswith("색") else r["decided_by"])
        col = (0, 150, 0) if r["drug_ok"] == "1" else (0, 0, 200)
        cv2.putText(bar, f"{name}: {'OK' if r['drug_ok'] == '1' else 'NG'} by {why}", (x + 6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def sheets(files, out, prefix, per=6, cols=2, width=1100):
    for s in range(0, len(files), per):
        ims = [base.imread_any(f) for f in files[s:s + per]]
        ims = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1])), interpolation=cv2.INTER_AREA) for i in ims]
        while len(ims) % cols:
            ims.append(np.full_like(ims[0], 255))
        rows = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
        cv2.imencode(".png", np.vstack(rows))[1].tofile(os.path.join(out, f"sheet_{prefix}_{s // per + 1}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="test_final 폴더 (final_dev.csv, final_test.csv)")
    ap.add_argument("--photos", required=True, help="viz_final_report 폴더 (photos_dev, photos_test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allp = {}
    for split in ("dev", "test"):
        rows = list(csv.DictReader(open(os.path.join(a.final, f"final_{split}.csv"), encoding="utf-8-sig")))
        by = defaultdict(dict)
        order = []
        for r in rows:
            by[r["file_name"]][r["method"]] = r
            if r["file_name"] not in order:
                order.append(r["file_name"])
        photos = {int(os.path.basename(p)[:3]): p for p in glob.glob(os.path.join(a.photos, f"photos_{split}", "*.png"))}
        od = os.path.join(a.out, split)
        os.makedirs(od, exist_ok=True)
        files, differ = [], []
        for k, fn in enumerate(order, 1):
            r3, r4 = by[fn]["V3+색"], by[fn]["C4"]
            img = compose(base.imread_any(photos[k]), k, split, r3, r4)
            ok = lambda r: "OK" if r["drug_ok"] == "1" else "NG"
            path = os.path.join(od, f"{k:03d}_{r3['truth_drug']}_rot{r3['rot_deg']}_cam{r3['camera_la']}_V3{ok(r3)}_C4{ok(r4)}.png")
            cv2.imencode(".png", img)[1].tofile(path)
            files.append(path)
            if r3["pred_drug"] != r4["pred_drug"]:
                differ.append(path)
        sheets(files, a.out, split)
        dd = os.path.join(a.out, "test_differ")          # 개발용(dev_)·평가용(test_) 모두, 두 방식의 답이 다른 사진만
        os.makedirs(dd, exist_ok=True)
        for p in differ:
            shutil.copy(p, os.path.join(dd, f"{split}_{os.path.basename(p)}"))
        allp[split] = [os.path.join(dd, f"{split}_{os.path.basename(p)}") for p in differ]
        print(f"{split}: {len(files)}장 (두 방식 답이 다른 사진 {len(differ)}장)")
    sheets(allp["dev"] + allp["test"], a.out, "differ")
    print("저장 완료:", a.out)


if __name__ == "__main__":
    main()
