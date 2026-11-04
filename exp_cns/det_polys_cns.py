# =====================================================================
#  det_polys_cns.py  —  검출기 라벨 v2: 저장된 반듯한 상자 → Paddle 원래 네 꼭짓점 (Paddle 전용 GPU 환경에서 실행)
# =====================================================================
#
#  [왜]  det_label_v1 의 상자는 ocr_common_cns.polys_to_rects 가 Paddle 다각형을 가로·세로 반듯한 사각형으로 바꾼 것.
#        30° 단위로 돌린 사진에서 기울어진 글자를 반듯한 사각형으로 감싸면 모서리에 빈 곳이 생겨 넉넉해지고,
#        되돌린 상자의 기울기도 각인이 아니라 '돌린 각도'를 따름. 사람 확인 20장(spot20) 모두 넉넉 / 겹침, 8장은 기울기 다름.
#        → 같은 사진·같은 방향·같은 전처리로 Paddle 을 다시 돌려, 반듯하게 바꾸기 전의 네 꼭짓점을 저장.
#  [하는 일]  det_label_v1\auto.csv 500장 (상자 0개 사진은 건너뜀) 마다
#        1) 상자를 낸 전처리: retry=1 이면 K0L17R6, 아니면 원본 (ocr_cns --retry-box-prep 과 같음)
#        2) 고른 방향(angle, 축 정렬이면 소수 각도)으로 돌린 사진에 paddle_t0 검출 → (사각형, 네 꼭짓점)
#        3) pred.csv 의 best_boxes 사각형과 같은 사각형을 찾아 짝 (같은 사진·같은 설정이라 똑같이 나와야 함.
#           똑같은 게 없으면 겹침(IoU) 0.8 이상 중 최고로, 그것도 없으면 '짝 없음'으로 기록하고 v1 사각형 그대로)
#  [저장]  exp_cns\results\det_label_v2\polys.json
#          { 파일명: {"angle", "prep", "boxes": [{"rect", "poly", "text", "match"("same"/"iou"/"none")}]} }
#          poly 는 '돌린 사진' 좌표 — 돌리기 전 좌표로 되돌리기는 make_det_labels_cns.py --step labels2 에서
#  [실행]  .\.venv_paddle_gpu\Scripts\python.exe exp_cns\det_polys_cns.py
# =====================================================================

import csv
import json
import os
from collections import Counter

import common_cns as cc
import ocr_common_cns as oc
from make_det_labels_cns import K0L17R6, OUT as OUT_V1, PRED_DIR

OUT = cc.RESULTS / "det_label_v2"


def iou(a, b):
    """두 사각형 (x0, y0, x1, y1) 의 겹친 넓이 / 합친 넓이."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    rows = list(csv.DictReader(open(OUT_V1 / "auto.csv", encoding="utf-8-sig")))
    best = {r["file_name"]: json.loads(r["best_boxes"])
            for r in csv.DictReader(open(PRED_DIR / "pred.csv", encoding="utf-8-sig"))}
    items = {it["file_name"]: it for it in oc.load_split("train500", 0)}
    preps = {"orig": oc.Prep("orig", "train500"), K0L17R6: oc.Prep(K0L17R6, "train500")}
    os.makedirs(OUT, exist_ok=True)
    res, cnt = {}, Counter()
    for k, r in enumerate(rows, 1):
        fn = r["file_name"]
        if not best[fn]:                                                # 상자 0개 사진: 짝 맞출 상자 없음
            continue
        pname = K0L17R6 if r["retry"] == "1" else "orig"
        img, _ = preps[pname](items[fn], cc.imread(items[fn]["image_path"]))
        ang = float(r["angle"])
        dets = oc.paddle_t0_polys(oc.rotate_color(img, ang))
        boxes = []
        for b in best[fn]:
            rect, text = tuple(b[:4]), b[4]
            same = [p for rc, p in dets if tuple(rc) == rect]
            if same:
                boxes.append(dict(rect=list(rect), poly=same[0], text=text, match="same"))
            else:
                sc = [(iou(rc, rect), p) for rc, p in dets]
                s, p = max(sc, key=lambda t: t[0]) if sc else (0.0, None)
                if s >= 0.8:
                    boxes.append(dict(rect=list(rect), poly=p, text=text, match="iou"))
                else:                                                   # 짝 없음: v1 반듯한 사각형의 네 꼭짓점 그대로
                    x0, y0, x1, y1 = rect
                    boxes.append(dict(rect=list(rect), poly=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                      text=text, match="none"))
            cnt[boxes[-1]["match"]] += 1
        res[fn] = dict(angle=ang, prep=pname, boxes=boxes)
        if k % 50 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)}  짝: {dict(cnt)}")
    json.dump(res, open(OUT / "polys.json", "w", encoding="utf-8"))
    print(f"저장: {OUT / 'polys.json'} ({len(res)}장, 상자 {sum(cnt.values())}개, 짝 {dict(cnt)})")


if __name__ == "__main__":
    main()
