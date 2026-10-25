# =====================================================================
#  align_all_exp300.py  —  정렬 방법 ①③④⑤⑥ 로 사진 300장 전부를 정렬해서 방법별 폴더에 저장 (정렬만, 점수 안 냄)
# =====================================================================
#
#  [하는 일]  manifest.csv 의 300장 전부(개발용·평가용·채점 제외 12장 포함)를 잘라서(P0 흑백) 방법별로 정렬한 이미지를 저장해요.
#             정답 각인과 비교하거나 정확도를 계산하지 않아요 (평가용 점수는 이미 한 번 돌렸음). 파일 이름에도 판정을 안 넣어요.
#  [방법]    run_exp300.py / align_exp300.py 의 같은 코드 (비교 때와 같음)
#    1_box_craft : ① CRAFT 글자 상자의 기울기로 글줄 방향을 구해 돌림 (상자를 못 찾으면 ③ 획 방향으로 대신)
#    2_box_paddle: ① PaddleOCR 검출 상자(알약 넓이 절반이 넘는 상자는 제외)의 기울기로 돌림, 상자가 없으면 ③ 으로 대신,
#                  위아래(0°/180°)는 PaddleOCR 글줄 방향 분류기로 고름 (CPU 라 느림)
#    3_stroke    : ③ 블랙햇+탑햇으로 획만 뽑아 PCA 로 글줄 방향을 구해 돌림
#    4_polar     : ④ 알약 중심 기준으로 가장자리 띠를 펼친 직사각형 (cv2.warpPolar)
#    5_cnn       : ⑤ 각도 분류기(합성 사진으로 학습)가 예측한 각도 상위 2개를 읽어 확신도가 높은 쪽으로 돌림
#    6_tps       : ⑥ 12방향 중 확신도가 가장 높은 방향의 알약을 TPS 로 펴서 꺼낸 글자 띠 (32×128 을 4배로 키움)
#  [위아래(0°/180°)]  ①③ 은 θ 와 θ+180 중 읽기 모델(PARSeq) 확신도가 높은 쪽, ④ 는 띠와 180° 뒤집은 띠 중 확신도가 높은 쪽
#  [길쭉한 알약]      crop_pill 이 이미 긴 쪽을 가로로 폈으므로 모든 방법이 0°/180° 만 골라요 (④⑥ 도 펼치지 않고 돌린 알약을 저장)
#  [파일 이름]  번호_약코드_rot회전각_cam카메라각도_{dev|test}_ang고른각도_출처.png    (출처: box/stroke/none = 각도를 얻은 곳)
#               ※ rot(파일 이름의 회전각)은 이름에만 적고 정렬에는 쓰지 않아요.
#  [저장]  --out 폴더 (원본 사진이 들어 있어 GitHub 금지, viz_* 폴더라 .gitignore 에 걸림)
#          aligned_300.csv : 사진별·방법별 고른 각도, 출처 (정답·판정 없음)
#  [실행 예]  python exp300\align_all_exp300.py --manifest exp300\results\manifest.csv --out exp300\results\viz_aligned_300
# =====================================================================

import argparse
import csv
import os
import time

import torch                                         # Paddle 보다 먼저 (align_exp300 참고)

import cv2
import numpy as np

import align_exp300 as al
import run_exp300 as base

FOLDERS = {"box": "1_box_craft", "pbox_cls": "2_box_paddle", "stroke": "3_stroke", "polar": "4_polar", "cnn": "5_cnn",
           "tps": "6_tps"}


def run(method, g0, aspect, ocr, step):
    """→ (저장할 이미지, 고른 각도 문자열, 각도를 얻은 곳)"""
    if method == "box":
        t, c, cands, ang, info = base.read_aligned(g0, aspect, ocr)
        return base.rotate(g0, ang), f"{ang:g}", info["src"]
    if method == "pbox_cls":
        t, c, cands, ang, info = al.read_pbox(g0, aspect, ocr, use_cls=True)
        return base.rotate(g0, ang), f"{ang:g}", info["src"]
    if method == "stroke":
        t, c, cands, ang, info = al.read_stroke(g0, aspect, ocr)
        return base.rotate(g0, ang), f"{ang:g}", info["src"]
    if method == "cnn":
        t, c, cands, ang, info = al.read_cnn(g0, aspect, ocr)
        return base.rotate(g0, ang), f"{ang:g}", info["src"]
    if method == "polar":
        t, c, cands, ang, info = al.read_polar(g0, aspect, ocr)
        if "strip" not in info:                              # 길쭉: 0/180 으로 돌린 알약
            return base.rotate(g0, ang), f"{ang:g}", info["src"]
        s = info["strip"]
        s = s[::-1, ::-1].copy() if str(ang).endswith("r") else s
        return s, str(ang), "polar"
    if method == "tps":
        t, c, cands, ang, info = al.read_tps(g0, aspect, ocr, step)
        if "rect" not in info:                               # 길쭉: 0/180 으로 돌린 알약
            return base.rotate(g0, ang), f"{ang:g}", info["src"]
        return cv2.resize(info["rect"], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), f"{ang:g}", "tps"
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞에서 N장만")
    ap.add_argument("--only", default="", help="일부 방법만 (쉼표, 예: pbox_cls). 이미 만든 폴더·CSV 는 건드리지 않고 aligned_300_<이름>.csv 로 저장")
    a = ap.parse_args()
    todo = {m: d for m, d in FOLDERS.items() if not a.only or m in a.only.split(",")}
    with open(a.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[: a.limit]
    for d in todo.values():
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    ocr = base.PARSeq()
    prep = base.PREPS["P0"][1]
    out_rows, t0 = [], time.time()
    for k, r in enumerate(rows, 1):
        img = base.imread_any(r["image_path"])
        gray, aspect, color, ok = base.crop_pill(img)
        g0 = prep(gray)
        shape = base.shape_group(aspect)
        for m, d in todo.items():
            im, ang, src = run(m, g0, aspect, ocr, a.step)
            name = f"{k:03d}_{r['drug_N']}_rot{r.get('rot_deg', '')}_cam{r.get('camera_la', '')}_{r['split']}_ang{ang.replace('.', 'p')}_{src}.png"
            cv2.imencode(".png", im)[1].tofile(os.path.join(a.out, d, name))
            out_rows.append({"no": k, "file_name": r["file_name"], "split": r["split"], "shape": shape, "method": d,
                             "angle": ang, "source": src, "saved_as": name})
        if k % 25 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)}  ({time.time() - t0:.0f}s)")
    csv_name = "aligned_300.csv" if not a.only else f"aligned_300_{a.only.replace(',', '_')}.csv"
    with open(os.path.join(a.out, csv_name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장 완료: {a.out}  (사진 {len(rows)}장 × 방법 {len(todo)}개, {csv_name})")


if __name__ == "__main__":
    main()
