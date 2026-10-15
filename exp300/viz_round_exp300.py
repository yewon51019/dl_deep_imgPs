# =====================================================================
#  viz_round_exp300.py  —  '둥근 알약' 전부의 글자 검출(DT) 박스 + 읽은 글자 그림
# =====================================================================
#
#  [하는 일]
#   manifest.csv 의 사진을 전부(300장) 알약 자르기까지만 해서 '둥근 알약'(긴변/짧은변 ≤ 1.3, 실험의 '판정 모양=둥근' 과 같은 기준)만 남기고,
#   그 알약마다 세 가지 방식의 박스와 읽은 글자를 그려요.
#     · CRAFT+PARSeq   : CRAFT 로 박스를 찾고 박스마다 PARSeq 로 읽음 (전처리 P0. 박스가 없으면 알약 전체를 읽음)
#     · Paddle 기본    : PaddleOCR 기본 검출 기준 (전처리 P1)
#     · Paddle 0.0     : PaddleOCR 검출 기준 0.0   (전처리 P1)
#
#  [저장]  --out 폴더
#   round_NNN_rot..._cam....png : 사진별 2×2 비교 (잘라낸 알약 / CRAFT / Paddle 기본 / Paddle 0.0)
#   sheet_1.png ...             : 6장씩 모은 그림
#   round_boxes.csv             : 사진별·방식별 박스 개수·좌표, 읽은 글자, 정답 비교(O 정확 / ~ 포함 / X 불일치)
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
from viz_boxes_exp300 import (METHODS, TITLES, paddle_detect, craft_detect, joined_read, judge, draw_panel, compose,
                               make_sheets, ascii_only, print_versions)


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

    print_versions()
    print("모델 불러오는 중... (EasyOCR CRAFT, PARSeq, PaddleOCR 두 설정)")
    base.easy_reader()
    parseq = base.PARSeq()
    models = {"paddle": PaddleOCRModel(), "paddle_t0": PaddleOCRModelT0()}

    csv_rows, files = [], []
    for k, (r, crop) in enumerate(rounds, 1):
        g0, g1 = p0(crop), p1(crop)
        truth_norm = base.norm(r.get("truth", ""))
        results = {}
        for m in METHODS:
            t0, err, polys, pinfo, fb = time.time(), None, None, {}, False
            try:
                if m == "craft":
                    boxes, texts, fb = craft_detect(g0, parseq)          # CRAFT 박스 + PARSeq 읽기
                    g = g0
                else:
                    boxes, texts, polys, pinfo = paddle_detect(models[m], g1)
                    g = g1
            except Exception as e:                       # 한 방식이 실패해도 나머지는 계속
                boxes, texts, g, err = [], [], g1, f"{type(e).__name__}: {e}"
                print(f"  [경고] {r['file_name'][:20]} {m}: {err}")
            read = joined_read(boxes, texts)
            j = judge(read, truth_norm)
            im, head = draw_panel(g, m, boxes, texts, polys, err)
            if fb:
                head += " (fallback)"
            results[m] = (im, head, boxes, read, j)
            csv_rows.append({"file_name": r["file_name"], "split": r.get("split", ""), "rot_deg": r.get("rot_deg", ""),
                             "camera_la": r.get("camera_la", ""), "form": r.get("form", ""), "method": m,
                             "prep": "P0" if m == "craft" else "P1",
                             "n_boxes": 0 if fb else len(boxes),          # CRAFT 가 못 찾아 알약 전체를 읽은 경우는 0개
                             "fallback": int(fb), "n_dt": pinfo.get("n_dt", ""), "n_text": pinfo.get("n_text", ""),
                             "read": read, "truth_norm": truth_norm, "judge": j, "texts": "|".join(texts),
                             "ms": int((time.time() - t0) * 1000), "error": err or "",
                             "boxes": json.dumps([[round(v) for v in b] for b in boxes])})
        cap = f"{k:03d}  round  rot={r.get('rot_deg', '')}  camera={r.get('camera_la', '')}  {r.get('split', '')}"
        path = os.path.join(a.out, f"round_{k:03d}_rot{r.get('rot_deg', '')}_cam{r.get('camera_la', '')}.png")
        cv2.imwrite(path, compose(g0, results, cap, truth_norm))
        files.append(path)
        if k % 10 == 0 or k == len(rounds):
            print(f"  {k}/{len(rounds)}  (최근: " + "  ".join(f"{TITLES[m]}={results[m][4]}" for m in METHODS) + ")")

    make_sheets(files, a.out, per_sheet=a.per_sheet)
    with open(os.path.join(a.out, "round_boxes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    def rate(rs, cond):
        return f"{sum(cond(x) for x in rs) / len(rs) * 100:5.1f}%" if rs else "   -  "

    print("\n[1. 둥근 알약 검출 요약]  (CRAFT 박스 0개 = 알약 전체를 한 줄로 읽은 경우)")
    print(f"{'방식':<16} {'사진':>4} {'평균 박스':>9} {'박스 0개':>8} {'0개 비율':>8} {'오류':>4}")
    for m in METHODS:
        rs = [x for x in csv_rows if x["method"] == m]
        if rs:
            z = sum(x["n_boxes"] == 0 for x in rs)
            print(f"{TITLES[m]:<16} {len(rs):4d} {np.mean([x['n_boxes'] for x in rs]):9.1f} {z:8d} "
                  f"{z / len(rs) * 100:7.1f}% {sum(bool(x['error']) for x in rs):4d}")

    graded = [x for x in csv_rows if x["judge"] != "-"]                 # 정답이 '마크'뿐인 사진은 제외
    n_img = len({x["file_name"] for x in graded})
    print(f"\n[2. 읽기 결과]  정답 각인이 있는 사진 {n_img}장, 0° 한 번 읽기 기준")
    print(f"{'방식':<16} {'정확(O)':>8} {'포함(~)':>8} {'불일치(X)':>9} {'빈 결과':>8}")
    for m in METHODS:
        rs = [x for x in graded if x["method"] == m]
        print(f"{TITLES[m]:<16} {rate(rs, lambda x: x['judge'] == 'O'):>8} {rate(rs, lambda x: x['judge'] == '~'):>8} "
              f"{rate(rs, lambda x: x['judge'] == 'X'):>9} {rate(rs, lambda x: not x['read']):>8}")

    print("\n[3. 박스를 찾았을 때 vs 못 찾았을 때 정확(O) 비율]")
    for m in METHODS:
        rs = [x for x in graded if x["method"] == m]
        has = [x for x in rs if x["n_boxes"] > 0]
        no = [x for x in rs if x["n_boxes"] == 0]
        extra = " (못 찾으면 PARSeq 가 알약 전체를 읽음)" if m == "craft" else ""
        print(f"  {TITLES[m]:<16} 박스 있음 {rate(has, lambda x: x['judge'] == 'O')} ({len(has)}장)   "
              f"박스 없음 {rate(no, lambda x: x['judge'] == 'O')} ({len(no)}장){extra}")

    print("\n[4. PaddleOCR: 검출 박스 vs 글자가 실제로 나온 박스]  (검출은 됐는데 글자가 비면 '읽기 단계'에서 사라진 것)")
    for m in ("paddle", "paddle_t0"):
        rs = [x for x in csv_rows if x["method"] == m and x["n_dt"] != ""]
        if rs:
            print(f"  {TITLES[m]:<16} 검출 박스 평균 {np.mean([x['n_dt'] for x in rs]):.1f}, 글자 나온 박스 평균 "
                  f"{np.mean([x['n_text'] for x in rs]):.1f}, 글자 0개인 사진 {sum(x['n_text'] == 0 for x in rs)}장 / {len(rs)}장")

    print("\n[5. 카메라 각도별: 박스 0개 비율 / 정확(O) 비율]")
    for cam in sorted({x["camera_la"] for x in csv_rows}):
        line = f"  카메라 {cam}°: "
        for m in METHODS:
            rs = [x for x in csv_rows if x["method"] == m and x["camera_la"] == cam]
            gs = [x for x in rs if x["judge"] != "-"]
            if rs:
                line += (f"{TITLES[m]} 0개 {sum(x['n_boxes'] == 0 for x in rs) / len(rs) * 100:.0f}% / "
                         f"O {rate(gs, lambda x: x['judge'] == 'O').strip()}   ")
        print(line)
    print("\n  ※ 0° 에서 한 번만 읽은 값이라 본 실험(12방향 다수결)의 점수와는 달라요. O/~/X 는 '읽은 글자 vs 정답 각인'(약 이름 맞힘 아님).")
    print(f"\n저장 완료: {a.out}/  (사진별 PNG {len(files)}장, sheet_*.png, round_boxes.csv)")

if __name__ == "__main__":
    main()
