# =====================================================================
#  extra_preps.py  —  추가 전처리 6종 (P5 ~ P10)
# =====================================================================
#
#  [팀원 B 실험의 전처리 8종과 이름 맞추기]
#     grayscale_output          = P0 (이미 있음: 흑백 + 크기 맞추기)
#     clahe_output              = P3 (이미 있음: CLAHE)
#     adaptive_threshold_output = P5  ← 이 파일
#     blackhat_output           = P6  ← 이 파일
#     closing_output            = P7  ← 이 파일
#     global_otsu_output        = P8  ← 이 파일
#     localized_otsu_output     = P9  ← 이 파일
#     tophat_output             = P10 ← 이 파일
#
#  모든 함수는 run_exp300.py 의 PREPS 와 같은 모양이에요: 흑백 이미지(uint8) -> 흑백 이미지(uint8)
#  · 긴 변을 400px 로 맞춘 뒤 처리합니다 (P0~P3 과 같은 크기).
#  · 결과는 '흰 바탕에 어두운 글자'로 맞춰서 OCR 이 읽기 쉽게 했어요.
#  · 숫자(커널 크기, 블록 크기 등)는 400px 기준의 '기본값'이에요. 바꾸고 싶으면 아래 상수만 고치세요.
# =====================================================================

import cv2
import numpy as np

from run_exp300 import BASE, resize_long

K_MORPH = 15        # 탑햇·블랙햇 커널 크기(px): 글자 획보다 약간 크게
K_CLOSE = 3         # 클로징 커널 크기(px): 아주 가는 틈만 메우도록 작게
ADAPT_BLOCK = 0.15  # 적응형 이진화: 이웃 영역 크기 = 짧은 변의 15%
ADAPT_C = 7         # 적응형 이진화: 평균에서 빼는 값(클수록 글자로 보는 기준이 엄격)
TILES = 3           # 지역 Otsu: 사진을 TILES×TILES 칸으로 나눔
FLAT_STD = 8        # 지역 Otsu: 밝기 변화가 이보다 작은 칸은 '글자 없음'(흰색)


def _norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)


def _blur(g):
    return cv2.GaussianBlur(g, (3, 3), 0)


def adaptive_threshold(g):
    """적응형 이진화: 주변 밝기와 비교해서 칸마다 기준을 달리함 (조명이 고르지 않을 때 유리)"""
    g = _blur(resize_long(g, BASE))
    block = max(11, int(min(g.shape) * ADAPT_BLOCK) // 2 * 2 + 1)         # 홀수여야 함
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, ADAPT_C)


def global_otsu(g):
    """전역 Otsu 이진화: 사진 전체에서 하나의 기준으로 흑/백 나눔"""
    g = _blur(resize_long(g, BASE))
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def localized_otsu(g):
    """지역 Otsu 이진화: 사진을 3×3 칸으로 나눠 칸마다 따로 Otsu (GO-PILL 의 '영역별 Otsu' 아이디어).
       글자 영역을 따로 찾지 않아서 칸 단위로 근사했어요. 밝기 변화가 거의 없는 칸은 흰색으로."""
    g = _blur(resize_long(g, BASE))
    out = np.full_like(g, 255)
    ys = np.linspace(0, g.shape[0], TILES + 1, dtype=int)
    xs = np.linspace(0, g.shape[1], TILES + 1, dtype=int)
    for i in range(TILES):
        for j in range(TILES):
            tile = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if tile.size and tile.std() >= FLAT_STD:
                out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = cv2.threshold(
                    tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return out


def closing(g):
    """클로징(팽창 후 침식): 어두운 아주 가는 틈·잡티를 메움 (작은 커널)"""
    g = resize_long(g, BASE)
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE, K_CLOSE)))


def tophat(g):
    """탑햇(원본 − 열기): 주변보다 '밝은' 작은 무늬만 뽑음 (어두운 알약의 밝은 글자). 뽑은 걸 뒤집어 흰 바탕에 어두운 글자로"""
    g = resize_long(g, BASE)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_MORPH, K_MORPH)))
    return 255 - _norm(th)


def blackhat(g):
    """블랙햇(닫기 − 원본): 주변보다 '어두운' 작은 무늬만 뽑음 (음각 그림자·검은 잉크). 뒤집어 흰 바탕에 어두운 글자로"""
    g = resize_long(g, BASE)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_MORPH, K_MORPH)))
    return 255 - _norm(bh)


# run_exp300.PREPS 에 추가할 항목 (이름, (설명, 함수))
PREPS_EXTRA = {
    "P5":  ("적응형 이진화(adaptive_threshold)", adaptive_threshold),
    "P6":  ("블랙햇(blackhat)",                   blackhat),
    "P7":  ("클로징(closing)",                    closing),
    "P8":  ("전역 Otsu(global_otsu)",             global_otsu),
    "P9":  ("지역 Otsu(localized_otsu)",          localized_otsu),
    "P10": ("탑햇(tophat)",                       tophat),
}
