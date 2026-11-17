"""0단계: 알약 찾기·자르기 — 배경째 찍은 사진 1장에서 알약을 모두 찾아 하나씩 컬러로 자른다.

방법 (exp300 의 알약 영역 찾기를 '덩어리 여러 개'로 넓힌 것):
  1. 긴 변 1000px 로 줄인 사진에서, 테두리(4%) 색의 중앙값 = 배경색. Lab 색 차이 > Otsu 기준(최소 20) = 알약 후보
  2. 열기 5×5 · 닫기 15×15 로 잡음 정리 → 덩어리마다 볼록 껍질 (글자 홈 포함)
  3. 버림: 넓이 < 0.2% (먼지, 기록도 안 함) · > 30% · 가장자리에 닿음 · 긴변/짧은변 > 4 · 채움 < 0.8
  4. 원본 해상도에서 기울지 않은 네모 + 사방 여유 15% 로 자름 (배경은 그대로)
한계: 테두리가 배경이라는 가정. 서로 닿은 알약은 한 덩어리로 나온다 (나누는 기능 없음).
"""
import cv2
import numpy as np

from . import config as C


def find_pills(bgr):
    """→ (알약 [(원본 좌표 네모 (x0, y0, x1, y1), 볼록 껍질)], 버린 덩어리 [(껍질, 이유)]). 알약은 왼쪽부터 번호 순."""
    H, W = bgr.shape[:2]
    s = C.WORK_SIDE / max(H, W)                       # 작업 좌표 / 원본 좌표
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(5, int(min(h, w) * 0.04))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    # 색 거리 × 2 를 0~255 로 자른 값에서 Otsu (최소 기준 20 도 이 값 기준)
    dist8 = np.clip(np.linalg.norm(lab - np.median(border, axis=0), axis=2) * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist8 > max(thr, 20)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep, drop = [], []
    e = C.EDGE * max(h, w)
    for c in cnts:
        hull = cv2.convexHull(c)
        area, harea = cv2.contourArea(c), cv2.contourArea(hull)
        x, y, bw, bh = cv2.boundingRect(hull)
        rw, rh = cv2.minAreaRect(hull)[1]
        if area < C.MIN_AREA * h * w:
            continue                                  # 먼지
        reason = ""
        if area > C.MAX_AREA * h * w:
            reason = "too big"
        elif x <= e or y <= e or x + bw >= w - e or y + bh >= h - e:
            reason = "touches edge"
        elif max(rw, rh) / max(1e-6, min(rw, rh)) > C.MAX_ELONG:
            reason = "thin strip"
        elif area / max(harea, 1e-6) < C.MIN_SOLID:
            reason = "ragged"
        hull_full = (hull.astype(np.float32) / s).astype(np.int32)   # 원본 좌표로
        if reason:
            drop.append((hull_full, reason))
            continue
        X, Y, BW, BH = cv2.boundingRect(hull_full)
        m = int(round(max(BW, BH) * C.MARGIN))
        keep.append(((max(0, X - m), max(0, Y - m), min(W, X + BW + m), min(H, Y + BH + m)), hull_full))
    keep.sort(key=lambda k: (k[0][0], k[0][1]))
    return keep, drop


def crop_pills(bgr):
    """→ 알약마다 dict(pill 번호, box, hull, img 자른 컬러 사진) 목록과 버린 덩어리 목록."""
    pills, drop = find_pills(bgr)
    out = [dict(pill=i, box=box, hull=hull, img=bgr[box[1]:box[3], box[0]:box[2]])
           for i, (box, hull) in enumerate(pills, 1)]
    return out, drop


def draw_check(bgr, pills, drop):
    """확인 그림: 초록 = 알약 윤곽, 파랑 네모 = 자른 영역, 주황 = 버린 덩어리와 이유 (원본 복사본에 그림)."""
    vis = bgr.copy()
    t = max(3, max(bgr.shape[:2]) // 400)
    for hull, reason in drop:
        cv2.drawContours(vis, [hull], -1, (0, 140, 255), t)
        x, y = hull[:, 0].min(0)
        cv2.putText(vis, f"drop: {reason}", (int(x), max(30, int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, (0, 140, 255), t)
    for p in pills:
        x0, y0, x1, y1 = p["box"]
        cv2.drawContours(vis, [p["hull"]], -1, (0, 220, 0), t)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), t)
        cv2.putText(vis, f"#{p['pill']}", (x0, max(40, y0 - 15)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (255, 200, 0), t)
    return vis
