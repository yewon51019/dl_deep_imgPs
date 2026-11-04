# =====================================================================
#  orient_check_cns.py  —  파일 이름의 회전각이 '각인이 똑바로 선 방향'과 맞는지 사람이 확인하는 그림 (학습 계획 3번 준비)
# =====================================================================
#
#  [왜]  0°/180° 판별 CNN 의 정답 라벨을 파일 이름의 회전각(예: K-000069_0_0_1_2_75_000_200 의 '000')으로 만들 수 있으면
#        학습용 전체를 사람 손 없이 라벨로 쓸 수 있음. 그러려면 회전각 000 사진에서 각인이 늘 똑바로 서 있어야 함.
#  [하는 일]  학습용(train, 개발용 36종·Λ 각인 약 5종 제외)에서 회전각 000 사진을 약마다 1장, 시드 42 로 n 장 골라
#             한 그림에 모음 (사진 위 글씨: 번호, 약 코드, 정답 각인, 모양). 원본 사진 그대로 (돌리거나 자르지 않음)
#  [저장]  exp_cns\results\orient_check\rot000_<n>.png  (원본 사진이 들어가므로 .gitignore 로 안 올림)
#          exp_cns\results\orient_check\rot000_<n>.csv   사람이 채울 칸 upright: 1 똑바로 / 0 아님(몇 도인지 note) / ? 판단 불가
#  [실행]  python exp_cns\orient_check_cns.py
# =====================================================================

import argparse
import csv
import os
import random

import cv2
import numpy as np

import common_cns as cc

LAMBDA_DRUGS = {"K-000027", "K-001061", "K-005676", "K-025922", "K-030177"}   # make_det_labels_cns 와 같음


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="고를 약 수 (약마다 1장)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=230)
    a = ap.parse_args()
    allt = cc.load_items(cc.TRAIN_ROOT)
    sp = cc.split_train(allt)
    cands = [it for it in allt if sp[it["file_name"]] == "train" and it["drug_N"] not in LAMBDA_DRUGS
             and it["rot"] == "000" and it["truth"]]                            # 정답 각인이 있는 사진만
    by_drug = {}
    for it in sorted(cands, key=lambda it: it["file_name"]):
        by_drug.setdefault(it["drug_N"], []).append(it)
    rng = random.Random(42)
    drugs = rng.sample(sorted(by_drug), min(a.n, len(by_drug)))
    pick = [rng.choice(by_drug[d]) for d in drugs]
    out = cc.RESULTS / "orient_check"
    os.makedirs(out, exist_ok=True)
    cells = []
    for i, it in enumerate(pick, 1):
        img = cc.imread(it["image_path"])
        s = a.tile / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cell = np.full((a.tile, a.tile, 3), 255, np.uint8)
        cell[:img.shape[0], :img.shape[1]] = img
        bar = np.full((44, a.tile, 3), 255, np.uint8)
        cv2.putText(bar, f"#{i} {it['drug_N']} cam{it['camera_la']}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(bar, f"T: {it['truth']}", (5, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 0), 1)
        cells.append(np.pad(np.vstack([bar, cell]), ((3, 3), (3, 3), (0, 0)), constant_values=128))
    while len(cells) % a.cols:
        cells.append(np.full_like(cells[0], 255))
    grid = np.vstack([np.hstack(cells[i:i + a.cols]) for i in range(0, len(cells), a.cols)])
    title = np.full((36, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"rot 000 photos (train, 1 per drug, seed 42) - is the imprint upright?", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    png = out / f"rot000_{len(pick)}.png"
    cc.imwrite(str(png), np.vstack([title, grid]))
    with open(out / f"rot000_{len(pick)}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "file_name", "drug_N", "truth", "shape", "form", "camera_la", "upright", "note"])
        for i, it in enumerate(pick, 1):
            w.writerow([i, it["file_name"], it["drug_N"], it["truth"], it["shape"], it["form"], it["camera_la"], "", ""])
    print(f"저장: {png}  ({len(pick)}장, 후보 약 {len(by_drug)}종)")


if __name__ == "__main__":
    main()
