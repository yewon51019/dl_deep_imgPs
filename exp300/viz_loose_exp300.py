# =====================================================================
#  viz_loose_exp300.py  —  넉넉하게 자르기(L) 결과 그림, 개발용 51장 (OCR 다시 안 돌림)
# =====================================================================
#   사진마다 [지금 자르기 (crop_pill) | 넉넉하게 자르기 (L)] 를 나란히 놓고, 위에 정답과
#   U1 (P0+P4+S1) / UL1 (P0+P4+L+S1) 의 읽은 글자·각인 판정(O/~/X)·약 판정을 적어요 (loose_cv410/results_dev.csv).
#  [저장]  --out 폴더 (원본 사진 포함 → GitHub 금지, viz_* 라서 .gitignore 에 걸림)
#    changed\  : 자르기 넓이가 1.1배 넘게 바뀐 사진 (낮은 기준으로 덩어리를 붙인 '예외 처리' 사진)
#    all\      : 개발용 51장 전부
#    sheet_changed_N.png, sheet_all_N.png : 6장씩 모은 그림
#  [실행 예]  python exp300\viz_loose_exp300.py --manifest exp300\results\manifest.csv ^
#             --loose exp300\results\loose_cv410 --out exp300\results\viz_loose_dt
# =====================================================================

import argparse
import csv
import os

import cv2
import numpy as np

import run_exp300 as base
from loose_exp300 import loose_crop
from viz_boxes_exp300 import ascii_only
from viz_two_final_exp300 import sheets

CELL_H, CELL_W = 300, 560
CHANGED = 1.1


def fit(g):
    """흑백 자른 사진 → 칸 크기에 맞춰 줄이고 가운데 놓기 (BGR)"""
    s = min(CELL_W / g.shape[1], CELL_H / g.shape[0])
    g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cell = np.full((CELL_H, CELL_W), 255, np.uint8)
    y, x = (CELL_H - g.shape[0]) // 2, (CELL_W - g.shape[1]) // 2
    cell[y:y + g.shape[0], x:x + g.shape[1]] = g
    return cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)


def compose(k, it, g0, gl, ratio, ru, rl):
    grid = np.hstack([fit(g0), np.full((CELL_H, 8, 3), 255, np.uint8), fit(gl)])
    bar = np.full((84, grid.shape[1], 3), 245, np.uint8)
    t = f"{k:02d} dev  {it['drug_N']}  rot={it['rot_deg']}  cam={it['camera_la']}  truth={ascii_only(base.norm(it['truth']))}  area x{ratio}"
    cv2.putText(bar, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    for x, name, r in ((0, "now crop | U1 (P0+P4+S1)", ru), (CELL_W + 8, "loose crop L | UL1 (+L)", rl)):
        cv2.putText(bar, name, (x + 6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 90, 90), 1, cv2.LINE_AA)
        col = (0, 150, 0) if r["judge"] == "O" else (0, 140, 230) if r["judge"] == "~" else (0, 0, 200)
        drug = "drug OK" if r["drug_ok"] == "1" else "drug NG"
        cv2.putText(bar, f"[{r['judge']}] read: {ascii_only(r['read']) or '(none)'}   {drug}", (x + 6, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--loose", required=True, help="loose_cv410 폴더 (results_dev.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for sub in ("changed", "all"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    res = {(r["variant"], r["file_name"]): r for r in csv.DictReader(open(os.path.join(a.loose, "results_dev.csv"), encoding="utf-8-sig"))}
    files = {"changed": [], "all": []}
    for k, it in enumerate(base.load_manifest(a.manifest, "dev"), 1):                 # 개발용만
        fn = it["file_name"]
        bgr = base.imread_any(it["image_path"])
        g0 = base.crop_pill(bgr)[0]
        gl, _, ratio = loose_crop(bgr)
        img = compose(k, it, g0, gl, ratio, res[("U1", fn)], res[("UL1", fn)])
        name = f"{k:02d}_{fn[:-4]}_U1{res[('U1', fn)]['judge']}_UL1{res[('UL1', fn)]['judge']}.png"
        for sub in ("all", "changed") if ratio > CHANGED else ("all",):
            path = os.path.join(a.out, sub, name)
            cv2.imencode(".png", img)[1].tofile(path)
            files[sub].append(path)
    for sub in ("changed", "all"):
        sheets(files[sub], a.out, sub)
        print(f"{sub}: {len(files[sub])}장")
    print(f"저장 완료: {a.out}")


if __name__ == "__main__":
    main()
