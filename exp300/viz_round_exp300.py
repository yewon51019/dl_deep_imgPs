# =====================================================================
#  viz_round_exp300.py  —  '둥근 알약' 전부의 글자 검출(DT) 박스 그림 (읽기는 하지 않음)
# =====================================================================
#
#  [하는 일]
#   manifest.csv 의 사진을 전부(300장) 알약 자르기까지만 해서 '둥근 알약'(긴변/짧은변 ≤ 1.3, 실험의 '판정 모양=둥근' 과 같은 기준)만 남기고,
#   그 알약마다 세 가지 검출 박스를 그려요. (글자 읽기·점수 계산은 없음 → 빠름)
#     · CRAFT          : PARSeq / TrOCR 이 쓰는 글자 위치 찾기 (전처리 P0)
#     · Paddle 기본    : PaddleOCR 기본 검출 기준 (전처리 P1)
#     · Paddle 0.0     : PaddleOCR 검출 기준 0.0   (전처리 P1)
#
#  [저장]  --out 폴더
#   round_NNN_rot..._cam....png : 사진별 2×2 비교 (잘라낸 알약 / CRAFT / Paddle 기본 / Paddle 0.0)
#   sheet_1.png ...             : 6장씩 모은 그림
#   round_boxes.csv             : 사진별·방식별 박스 개수와 좌표
#
#  [주의]  알약을 돌리지 않은 0° 상태에서 한 번만 찾은 박스예요. (본 실험은 12방향으로 돌려 읽음)
#
#  [실행 예]
#     python viz_round_exp300.py --manifest manifest.csv --out viz_round
#     python viz_round_exp300.py --manifest manifest.csv --out viz_round_test --limit 10    # 시험 운전
# =====================================================================

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

import run_exp300 as base
from run_exp300 import imread_any, crop_pill, shape_group
from run_exp300_paddle import PaddleOCRModel, PaddleOCRModelT0
from viz_boxes_exp300 import (METHODS, TITLES, paddle_detect, draw_panel, compose, make_sheets, ascii_only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="viz_round")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 둥근 알약 앞에서 N장만 (0=전부)")
    ap.add_argument("--per-sheet", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"사진 {len(rows)}장에서 둥근 알약 찾는 중... (알약 자르기만 해서 빨라요)")
    p0, p1 = base.PREPS["P0"][1], base.PREPS["P1"][1]
    rounds, n_long = [], 0
    for r in rows:
        img = imread_any(r["image_path"])
        if img is None:
            print(f"  [경고] 이미지를 못 읽음: {r['image_path']}")
            continue
        crop, aspect, *_ = crop_pill(img)
        if shape_group(aspect) == "둥근":
            rounds.append((r, crop))
        else:
            n_long += 1
    by_split, by_cam, by_rot = defaultdict(int), defaultdict(int), defaultdict(int)
    for r, _ in rounds:
        by_split[r.get("split", "")] += 1
        by_cam[r.get("camera_la", "")] += 1
        by_rot[r.get("rot_deg", "")] += 1
    print(f"둥근 알약 {len(rounds)}장 (길쭉 {n_long}장 제외) | 세트별 {dict(by_split)} | 카메라각도별 {dict(sorted(by_cam.items()))} "
          f"| 회전각별 {dict(sorted(by_rot.items()))}")
    if a.limit:
        rounds = rounds[: a.limit]
        print(f"※ 시험 운전: 앞 {len(rounds)}장만")

    print("모델 불러오는 중... (EasyOCR CRAFT, PaddleOCR 두 설정)")
    base.easy_reader()
    models = {"paddle": PaddleOCRModel(), "paddle_t0": PaddleOCRModelT0()}

    csv_rows, files = [], []
    for k, (r, crop) in enumerate(rounds, 1):
        g0, g1 = p0(crop), p1(crop)
        results = {}
        for m in METHODS:
            t0, err, polys = time.time(), None, None
            try:
                if m == "craft":
                    boxes, g = [list(b) for b in base.craft_boxes(g0)], g0
                else:
                    boxes, _texts, polys = paddle_detect(models[m], g1)
                    g = g1
            except Exception as e:                       # 한 방식이 실패해도 나머지는 계속
                boxes, g, err = [], g1, f"{type(e).__name__}: {e}"
                print(f"  [경고] {r['file_name'][:20]} {m}: {err}")
            im, head = draw_panel(g, m, boxes, [], polys, err)       # 글자는 적지 않음 (검출만)
            results[m] = (im, head, boxes, "", "-")
            csv_rows.append({"file_name": r["file_name"], "split": r.get("split", ""), "rot_deg": r.get("rot_deg", ""),
                             "camera_la": r.get("camera_la", ""), "form": r.get("form", ""), "method": m,
                             "prep": "P0" if m == "craft" else "P1", "n_boxes": len(boxes),
                             "ms": int((time.time() - t0) * 1000), "error": err or "",
                             "boxes": json.dumps([[round(v) for v in b] for b in boxes])})
        cap = f"{k:03d}  round  rot={r.get('rot_deg', '')}  camera={r.get('camera_la', '')}  {r.get('split', '')}"
        path = os.path.join(a.out, f"round_{k:03d}_rot{r.get('rot_deg', '')}_cam{r.get('camera_la', '')}.png")
        cv2.imwrite(path, compose(g0, results, cap, "", show_read=False))
        files.append(path)
        if k % 10 == 0 or k == len(rounds):
            print(f"  {k}/{len(rounds)}  (최근: " + "  ".join(f"{TITLES[m].split('+')[0]}={len(results[m][2])}" for m in METHODS) + ")")

    make_sheets(files, a.out, per_sheet=a.per_sheet)
    with open(os.path.join(a.out, "round_boxes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    print("\n[둥근 알약 검출 요약]")
    print(f"{'방식':<16} {'사진':>4} {'평균 박스':>9} {'박스 0개':>8} {'0개 비율':>8} {'오류':>4}")
    for m in METHODS:
        rs = [x for x in csv_rows if x["method"] == m]
        if rs:
            z = sum(x["n_boxes"] == 0 for x in rs)
            print(f"{TITLES[m].split('+')[0]:<16} {len(rs):4d} {np.mean([x['n_boxes'] for x in rs]):9.1f} {z:8d} "
                  f"{z / len(rs) * 100:7.1f}% {sum(bool(x['error']) for x in rs):4d}")
    print("\n[카메라 각도별 박스 0개 비율]")
    for cam in sorted({x["camera_la"] for x in csv_rows}):
        line = f"  카메라 {cam}°: "
        for m in METHODS:
            rs = [x for x in csv_rows if x["method"] == m and x["camera_la"] == cam]
            if rs:
                line += f"{TITLES[m].split('+')[0]} {sum(x['n_boxes'] == 0 for x in rs) / len(rs) * 100:.0f}% ({len(rs)}장)   "
        print(line)
    print(f"\n저장 완료: {a.out}/  (사진별 PNG {len(files)}장, sheet_*.png, round_boxes.csv)")


if __name__ == "__main__":
    main()
