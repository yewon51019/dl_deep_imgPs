"""1단계: 방향 정하기 — 몇 개의 각도로 돌려 읽을지 정한다.

  긴 알약 (긴변/짧은변 > 1.3, 감싸는 상자가 사진을 0.8 이상 덮음) → [축 각도, 축 각도 + 180] 2방향
  그 밖 (원형 등)                                                 → 30° 간격 12방향
축은 '자른 원본'을 긴 변 400 정사각형으로 맞춘 사진의 알약 영역(배경과의 색 차이)에서 최소 외접 사각형으로 구한다.
(exp_cns/align_long_cns.py v3 와 같은 계산)
"""
import cv2
import numpy as np

from . import config as C
from .imageio import fit_side


def content_rect(src_hw, side=C.SIDE):
    """fit_side 결과에서 실제 사진이 놓인 사각형 (x, y, w, h). 나머지는 정사각형을 맞추려 채운 여백."""
    h, w = src_hw
    s = side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return (side - nw) // 2, (side - nh) // 2, nw, nh


def pill_mask(img):
    """400 정사각형 컬러 → 알약 마스크(0/255). 배경색 = 네 모서리(6%) 중앙값. 못 찾으면 None."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    c = max(6, int(min(h, w) * 0.06))
    corners = np.concatenate([lab[:c, :c].reshape(-1, 3), lab[:c, -c:].reshape(-1, 3),
                              lab[-c:, :c].reshape(-1, 3), lab[-c:, -c:].reshape(-1, 3)])
    dist8 = np.clip(np.linalg.norm(lab - np.median(corners, axis=0), axis=2) * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (dist8 > max(thr, 20)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))     # 작은 점 지우기
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))    # 각인 홈 구멍 메우기
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))                       # 가장 큰 덩어리
    if st[k, cv2.CC_STAT_AREA] < 0.05 * h * w:
        return None
    return (lab_cc == k).astype(np.uint8) * 255


def long_axis(img, crect):
    """→ dict(ok, ratio 긴변/짧은변, angle 돌리면 긴 축이 가로가 되는 각도 0~180, cover 덮은 비율)."""
    m = pill_mask(img)
    if m is None:
        return dict(ok=False, ratio=1.0, angle=0.0, cover=0.0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (_, _), (rw, rh), ang = rect
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    _, _, bw, bh = cv2.boundingRect(cv2.boxPoints(rect).astype(np.float32))
    cover = min(bw / crect[2], bh / crect[3])
    return dict(ok=True, ratio=rw / max(rh, 1.0), angle=float(ang % 180), cover=float(min(cover, 1.0)))


def is_long(info):
    return info["ok"] and info["ratio"] > C.LONG_RATIO and info["cover"] >= C.MIN_COVER


def angles(src):
    """자른 원본 컬러 → (읽을 각도 목록, 축 정보). 긴 알약 각도는 소수 첫째 자리 (boxes.json 의 열쇠가 이 글자)."""
    info = long_axis(fit_side(src, C.SIDE), content_rect(src.shape[:2]))
    if is_long(info):
        a = round(info["angle"], 1)
        return [a, round((a + 180) % 360, 1)], info
    return list(range(0, 360, 360 // C.N_ANGLES)), info
