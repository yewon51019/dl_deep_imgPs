"""사진 읽기·저장·크기 맞추기·돌리기 (cv2, numpy 만 사용 — 두 가상환경이 같이 씀)."""
import os

import cv2
import numpy as np


def imread(path):
    """한글 경로도 읽는 BGR 컬러 읽기 (cv2.imread 는 한글 경로에서 실패)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    """한글 경로 저장: 메모리에서 압축한 뒤 파일로 씀."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".png", img)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    buf.tofile(str(path))


def fit_side(img, side, fill=None):
    """비율을 유지해 긴 변을 side 로 맞추고 남는 곳을 채워 side×side 정사각형으로.
    줄일 때 INTER_AREA, 키울 때 INTER_CUBIC. fill 이 없으면 네 모서리 픽셀의 중앙값(배경과 비슷한 색)."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if fill is None:
        fill = np.median(np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]), axis=0)
    out = np.empty((side, side) + img.shape[2:], img.dtype)
    out[...] = np.asarray(fill, dtype=img.dtype)
    y, x = (side - nh) // 2, (side - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def rotate_color(img, ang):
    """컬러 사진을 ang 도 돌림. 모서리가 잘리지 않게 캔버스를 키우고 빈 곳은 가장자리 색으로 채움.
    검출(2단계)과 읽기(3단계)가 같은 함수를 써야 저장된 상자 좌표가 사진과 맞는다."""
    if ang == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
