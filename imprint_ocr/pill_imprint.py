# -*- coding: utf-8 -*-
"""
pill_imprint.py — 알약 각인 인식 실험 통합 코드
===============================================
캡스톤디자인 "정신건강의학과 약 분류 AI" 팀 (김태린, 남예원, 장혜정, 조하은)

이 파일 하나로 아래 4가지를 할 수 있어요.

  1) fuse  : 같은 알약을 조명 방향만 바꿔 여러 장 찍은 사진을 1장으로 합성
             (비스듬한 조명 = 사광 → 파인 각인의 그림자를 모아서 글자를 진하게 만듦)
  2) crop  : 여러 알약이 찍힌 사진에서 알약을 1개씩 잘라냄
  3) run   : 잘라낸 알약들의 각인을 OCR로 읽고, 약을 판별하고, 점수를 매김
  4) check : 설치가 잘 됐는지 확인

자세한 사용법은 README.md 를 보세요. 가장 기본 실행은:

    python pill_imprint.py check
    python pill_imprint.py crop --input images/multi_scene.jpg --out crops
    python pill_imprint.py run  --labels labels.csv --db pill_db.csv

전체 처리 흐름 (run)
--------------------
  사진 → [배경 지우기] → [모양·색 뽑기] → [인쇄/음각 자동 구분] → [전처리 A~D]
       → [돌려가며 OCR] → [다수결로 1개 선택] → [약 판별(매칭 3종)] → [채점·저장]

참고 논문
---------
  - 크롭 후 OCR       : Sai Sachin & Karthikeyan (2026), Frontiers in AI — EasyOCR이 Tesseract보다 알약 각인에 적합
  - 편집거리 매칭     : Heo et al. (2023), JMIR — 각인 유사도 = (길이-편집거리)/(길이+편집거리)
  - 다방향 조명 합성  : Xiang et al. (2018), EURASIP JIVP — 금속 각인 문자, 단일 조명 93~98% → 합성 99.6%
"""

import argparse
import csv
import glob
import os
import time
from collections import Counter

import cv2
import numpy as np


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 0. 공통 도구                                                        ║
# ╚════════════════════════════════════════════════════════════════════╝

def imread_any(path):
    """한글 경로도 읽을 수 있는 이미지 읽기.
    (윈도우에서 cv2.imread 는 경로에 한글이 있으면 None 을 돌려주는 문제가 있음)"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_any(path, img):
    """한글 경로에도 저장할 수 있는 이미지 저장"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def norm(s):
    """각인 글자 정리: 대문자로 바꾸고 공백·기호를 지움.  'alza 18' -> 'ALZA18'"""
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def load_csv(path):
    """CSV 읽기. '#'으로 시작하는 줄은 메모로 보고 건너뜀"""
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 1. fuse : 다방향 조명(사광) 사진 합성                                ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# 왜 필요한가?
#   파인(음각) 각인은 잉크가 없고 '홈'만 있어서, 정면에서 조명을 비추면 그림자가 안 생겨
#   글자가 거의 안 보임. 조명을 옆에서 비스듬히(사광) 비추면 홈의 한쪽 벽에 그림자가 생김.
#   단, 한 방향 조명은 글자의 '한쪽 테두리'만 진해짐 → 방향을 바꿔 여러 장 찍고 합치면
#   글자 테두리 전체가 진해짐. (Xiang et al., 2018 의 금속 각인 인식 방식을 단순화)
#
# 촬영 방법 (자세한 건 README):
#   - 폰을 거치대에 고정 (사진끼리 위치가 같아야 함)
#   - 방을 어둡게 하고, 손전등을 알약 높이에서 비스듬히(약 20~30도) 비춤
#   - 위 → 오른쪽 → 아래 → 왼쪽 순서로 조명 위치만 바꿔 4장 촬영
#
# 만들어지는 결과 3장:
#   fused_min.png  : 픽셀마다 '가장 어두운 값'만 모은 사진 → 모든 방향의 그림자가 합쳐짐 (OCR 추천)
#   fused_diff.png : 반대 방향끼리 밝기 차이(위-아래, 왼-오른) → 각인 테두리만 하얗게 강조
#   fused_mean.png : 단순 평균 (비교용)

def _align_to(ref_gray, img_gray, img_bgr):
    """폰이 살짝 흔들렸을 때 기준 사진에 맞춰 위치를 보정 (ECC 정합).
    실패하면 보정 없이 그대로 돌려줌."""
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        # 조명 방향이 달라 밝기가 다르므로, 경계선(기울기) 이미지끼리 맞춤
        def edges(g):
            g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
            return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        cv2.findTransformECC(edges(ref_gray), edges(img_gray), warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        h, w = ref_gray.shape
        return cv2.warpAffine(img_bgr, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE), True
    except cv2.error:
        return img_bgr, False


def fuse_lights(input_paths, out_dir, align=True):
    """같은 장면을 조명 방향만 바꿔 찍은 사진들을 합성.
    input_paths 순서가 [위, 오른쪽, 아래, 왼쪽] 이면 반대 방향 차이(fused_diff)를 정확히 계산."""
    os.makedirs(out_dir, exist_ok=True)
    imgs = [imread_any(p) for p in input_paths]
    if any(im is None for im in imgs) or len(imgs) < 2:
        raise ValueError("사진을 2장 이상, 올바른 경로로 넣어주세요.")

    # 크기를 첫 장에 맞춤
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im for im in imgs]

    # (선택) 흔들림 보정
    ref = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    if align:
        for i in range(1, len(imgs)):
            imgs[i], ok = _align_to(ref, cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), imgs[i])
            print(f"  [{i + 1}/{len(imgs)}] 위치 보정 {'성공' if ok else '실패(원본 사용)'}")

    # 사진마다 전체 밝기가 달라서, 각 사진을 '조명 지도'로 나눠 밝기를 맞춤
    grays = []
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(g, (0, 0), max(5.0, min(g.shape) * 0.05))
        grays.append(g / (bg + 1.0))
    stack = np.stack(grays)

    # 밝기를 맞추면 어두운 배경도 밝게 변하므로, 원본 평균 사진에서 알약 위치를 찾아
    # 알약 바깥은 검정으로 칠함 → 다음 단계 crop 이 '어두운 배경 위 밝은 알약'으로 인식할 수 있게
    raw_mean = np.mean([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs], axis=0).astype(np.uint8)
    _, pill = cv2.threshold(cv2.GaussianBlur(raw_mean, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pill = cv2.morphologyEx(pill, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0

    def to8(x):
        x = x.copy()
        lo, hi = np.percentile(x[pill], [1, 99]) if pill.any() else (x.min(), x.max())
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255   # 알약 안쪽 밝기 기준으로 0~255
        x[~pill] = 0                                            # 바깥은 검정
        return x.astype(np.uint8)

    fused_min = to8(stack.min(axis=0))     # 모든 방향 그림자 합치기
    fused_mean = to8(stack.mean(axis=0))   # 평균
    if len(grays) == 4:                    # 위-아래, 오른-왼 반대 방향 차이
        diff = np.abs(grays[0] - grays[2]) + np.abs(grays[1] - grays[3])
    else:                                  # 4장이 아니면 최대-최소 차이로 대신함
        diff = stack.max(axis=0) - stack.min(axis=0)
    fused_diff = to8(diff)

    # 다음 단계(crop/run)는 컬러 사진을 받으므로, 3채널로 저장
    for name, img in [("fused_min.png", fused_min), ("fused_diff.png", fused_diff), ("fused_mean.png", fused_mean)]:
        imwrite_any(os.path.join(out_dir, name), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    print(f"[fuse] {len(imgs)}장 합성 완료 -> {out_dir}/fused_min.png (OCR 추천), fused_diff.png, fused_mean.png")


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 2. crop : 여러 알약 사진 → 알약 1개씩 잘라내기                        ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# 원리: 어두운 배경 위의 밝은 알약을 Otsu(자동 기준값)로 흰색/검정 두 색으로 나누고,
#       흰 덩어리마다 기울어진 상자를 씌워 똑바로 펴서 잘라냄.
# 주의: 배경이 알약보다 어두워야 함 (검은 천·어두운 책상 추천). 흰 알약을 흰 종이 위에
#       놓으면 구분이 안 됨.

def crop_pills(input_path, out_dir, min_area_ratio=0.0005):
    os.makedirs(out_dir, exist_ok=True)
    img = imread_any(input_path)
    if img is None:
        raise FileNotFoundError(input_path)

    # 빠르게 찾기 위해 긴 변 1000px로 줄인 사진에서 위치만 찾음
    scale = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Otsu: 밝은 알약 / 어두운 배경을 나누는 기준값을 자동으로 골라 흑백으로 나눔
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # 작은 먼지 제거
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 알약 속 구멍 메우기

    # 흰 덩어리(=알약)마다 윤곽선을 찾고, 너무 작은 건(먼지) 버림
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    rects = [cv2.minAreaRect(c) for c in cnts if cv2.contourArea(c) >= min_area]
    rects.sort(key=lambda r: (round(r[0][1] / 50), r[0][0]))   # 위→아래, 왼→오른 순서로 번호

    overview = small.copy()
    saved = []
    for i, ((cx, cy), (w, h), ang) in enumerate(rects):
        # 축소본 좌표 → 원본 좌표, 상자는 15% 넉넉하게
        cx, cy, w, h = cx / scale, cy / scale, w / scale * 1.15, h / scale * 1.15
        if w < h:                       # 긴 쪽이 가로가 되도록 90도 돌림
            w, h, ang = h, w, ang + 90
        # 사진 전체를 상자 각도만큼 돌린 뒤, 상자 부분만 잘라냄 = 기울어진 알약을 똑바로 펴기
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (cx, cy))
        path = os.path.join(out_dir, f"crop_{i:02d}.png")
        imwrite_any(path, crop)
        saved.append(path)

        # 확인용 그림에 빨간 상자 + 번호 표시
        box = np.int32(cv2.boxPoints(((cx * scale, cy * scale), (w * scale, h * scale), ang)))
        cv2.drawContours(overview, [box], 0, (0, 0, 255), 2)
        cv2.putText(overview, str(i), (int(cx * scale) - 10, int(cy * scale) - int(h * scale / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    imwrite_any(os.path.join(out_dir, "crop_overview.png"), overview)
    print(f"[crop] 알약 {len(saved)}개 검출 -> {out_dir}/  (번호 확인: crop_overview.png)")
    return saved


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 3. 배경 지우기 + 모양·색 + 인쇄/음각 구분                             ║
# ╚════════════════════════════════════════════════════════════════════╝

def pill_mask(bgr):
    """알약 영역 마스크(흰색=알약)와 '긴 변/짧은 변' 비율을 돌려줌"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 0.3 * g.size:
        # 알약이 화면을 꽉 채운 사진(배경이 거의 없음): 사진 전체를 알약으로 봄
        return np.full_like(g, 255), max(g.shape) / min(g.shape)
    c = max(cnts, key=cv2.contourArea)
    (_, (w, h), _) = cv2.minAreaRect(c)
    m = np.zeros_like(g)
    cv2.drawContours(m, [c], -1, 255, -1)
    k = max(3, int(min(g.shape) * 0.04))
    m = cv2.erode(m, np.ones((k, k), np.uint8))   # 테두리 그림자는 빼고 살짝 안쪽만 사용
    return m, max(w, h) / max(1.0, min(w, h))


def isolate_pill(bgr):
    """배경을 알약의 중간 밝기로 칠해서 지운 흑백 이미지 + 모양 비율"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m, aspect = pill_mask(bgr)
    x, y, w, h = cv2.boundingRect(m)
    out = np.full_like(g, int(np.median(g[m > 0])))
    out[m > 0] = g[m > 0]
    return out[y:y + h, x:x + w], aspect


def pill_features(bgr):
    """후보를 좁힐 때 쓰는 모양·색 (일부러 거칠게 나눔)
    - 모양: 긴 변/짧은 변 > 1.3 이면 '장방형', 아니면 '원형'
    - 색  : 채도(색의 진하기)가 낮으면 '무채색' (하양·회색은 조명에 따라 뒤바뀌어서 하나로 묶음)"""
    m, aspect = pill_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S = int(np.median(hsv[..., 0][m > 0])), int(np.median(hsv[..., 1][m > 0]))
    shape = "장방형" if aspect > 1.3 else "원형"
    if S < 50:
        color = "무채색"
    elif H < 8 or H >= 160:
        color = "빨강"
    elif H < 15:
        color = "주황"
    elif H < 35:
        color = "노랑"
    elif H < 85:
        color = "초록"
    elif H < 130:
        color = "파랑"
    else:
        color = "보라"
    return shape, color


# pill_db.csv 의 색 이름(식약처 표기)을 위 색 묶음으로 바꾸는 표
COLOR_GROUP = {"하양": "무채색", "흰색": "무채색", "회색": "무채색", "검정": "무채색",
               "노랑": "노랑", "주황": "주황", "빨강": "빨강", "분홍": "빨강",
               "초록": "초록", "연두": "초록", "파랑": "파랑", "남색": "파랑", "보라": "보라"}


def detect_imprint_type(gray):
    """인쇄/음각 자동 구분.
    조명을 평평하게 맞춘 뒤 '주변 밝기의 65%보다 어두운 픽셀(=진한 잉크)'이 0.5% 이상이면 인쇄.
    1~3차 실험 12장에서 12/12 정확 (인쇄 0.9~3.2%, 음각 0~0.02%로 차이가 큼)."""
    bg = cv2.GaussianBlur(gray, (0, 0), max(3.0, min(gray.shape) * 0.08)).astype(np.float32)
    dark = float((gray.astype(np.float32) / (bg + 1) < 0.65).mean())
    return ("인쇄" if dark >= 0.005 else "음각"), dark


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 4. 전처리 A~D                                                       ║
# ╚════════════════════════════════════════════════════════════════════╝
# 1~3차 결과로 이진화(흑백 나누기)·Top-hat 은 손해로 확인돼 뺐음.

def resize_long(g, long_side=400):
    """긴 변을 long_side 픽셀로 맞춤 (글자 크기를 일정하게 해야 OCR이 안정적)"""
    s = long_side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def flatten_light(g):
    """조명 평탄화: 크게 흐린 사진(=조명 지도)으로 나눠서 한쪽이 어둡게 찍힌 그늘을 없앰.
    3차 실험: 인쇄 각인에선 잉크도 연해져 손해였음 → 음각 전용 후보."""
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, min(g.shape) * 0.08)).astype(np.float32)
    out = g.astype(np.float32) / (bg + 1) * float(np.median(g))
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_glare(g):
    """반사광 제거: 거의 하얗게 날아간(240 이상) 작은 부분만 주변 값으로 메움.
    코팅이 반짝이는 알약에서 글자를 가리는 하이라이트를 없애기 위함."""
    med = float(np.median(g))
    m = ((g >= 240) & (g > med + 25)).astype(np.uint8) * 255
    if m.sum() == 0:
        return g
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.inpaint(g, m, 5, cv2.INPAINT_TELEA)


def pA(g):  # A: 배경 제거만 (1~3차 최고 성능)
    return g


def pB(g):  # B: + 조명 평탄화
    return flatten_light(g)


def pC(g):  # C: + 반사광 제거
    return remove_glare(g)


def pD(g):  # D: + 반사광 제거 + 조명 평탄화
    return flatten_light(remove_glare(g))


PREPROCESS = {"A": ("A_배경제거만", pA), "B": ("B_+조명평탄화", pB),
              "C": ("C_+반사광제거", pC), "D": ("D_+반사광+평탄화", pD),
              "AUTO": ("AUTO_유형별", None)}
AUTO_RULE = {"인쇄": pA, "음각": pD}   # AUTO: 인쇄는 A, 음각은 D (EasyOCR 결과 보고 바꿔도 됨)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 5. OCR 엔진                                                         ║
# ╚════════════════════════════════════════════════════════════════════╝
# 두 엔진 모두 "이미지 → (읽은 글자, 신뢰도 0~100)" 형태로 맞춰 둠.

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"   # 각인에 나올 글자만 허용


class EasyOCREngine:
    """EasyOCR: CRAFT(글자 위치 찾기) + CRNN(글자 읽기). 일상 사진 글자로 학습돼 알약에 더 적합.
    처음 실행 시 모델 파일(약 100MB)을 인터넷에서 자동으로 내려받음."""
    name = "easyocr"

    def __init__(self, gpu=False):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def __call__(self, img):
        # detail=1 → [(상자좌표, 글자, 신뢰도0~1), ...]
        res = self.reader.readtext(img, allowlist=ALLOW, detail=1)
        if not res:
            return "", 0.0
        res.sort(key=lambda r: min(p[0] for p in r[0]))   # 왼쪽 글자부터 이어 붙임
        return "".join(t for _, t, _ in res), float(np.mean([c for _, _, c in res]) * 100)


class TesseractEngine:
    """Tesseract: 종이 문서용 OCR. 비교 기준(baseline)으로만 사용.
    윈도우는 Tesseract 프로그램을 따로 설치하고, 아래 경로를 맞춰야 할 수 있음."""
    name = "tesseract"

    def __init__(self, cmd=None):
        import pytesseract
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd   # 예: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.t = pytesseract
        self.cfg = f"--psm 7 -c tessedit_char_whitelist={ALLOW}"   # psm 7 = 한 줄짜리 글자

    def __call__(self, img):
        img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_REPLICATE)
        d = self.t.image_to_data(img, config=self.cfg, output_type=self.t.Output.DICT)
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return "".join(w for w, _ in words), float(np.mean([c for _, c in words]))


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 6. 돌려가며 읽고 다수결로 고르기                                     ║
# ╚════════════════════════════════════════════════════════════════════╝

def rotate(g, ang):
    """사진을 ang도 돌림 (잘리지 않게 캔버스를 키움)"""
    if ang == 0:
        return g
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def read_imprint(gray, aspect, prep_fn, ocr, step=30):
    """원형 알약: 각인 방향을 모르니 step도 간격으로 360도를 돌며 읽음 (step=30 → 12번)
    장방형 알약: crop에서 이미 가로로 폈으니 0도/180도 2번만 읽음
    → 여러 결과 중 '가장 많이 나온 글자'(다수결)를 고르고, 동점이면 신뢰도가 높은 것.
    정답을 보고 고르지 않으므로 공정한 비교."""
    g = prep_fn(resize_long(gray))
    angles = [0, 180] if aspect > 1.3 else list(range(0, 360, step))
    cands = []
    for a in angles:
        text, conf = ocr(rotate(g, a))
        cands.append((norm(text), conf))
    votes = Counter(t for t, _ in cands if 1 <= len(t) <= 10)
    pick = max(cands, key=lambda c: (votes.get(c[0], 0) if c[0] else -1, c[1]))
    return pick[0], [t for t, _ in cands]


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 7. 약 판별 (매칭 3종) + 채점                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

def levenshtein(a, b):
    """편집거리: a를 b로 바꾸려면 글자를 최소 몇 번 고쳐야 하나 (삽입·삭제·교체 각 1번)"""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def similarity(a, b):
    """각인 유사도 (Heo et al., 2023): (길이 - 편집거리) / (길이 + 편집거리). 같으면 1, 다를수록 0"""
    if not a or not b:
        return 0.0
    L, ed = max(len(a), len(b)), levenshtein(a, b)
    return (L - ed) / (L + ed)


MIN_SIM = 0.3   # 이보다 덜 비슷하면 '판별 불가' (6글자 각인 기준 3글자 틀린 것까지 인정)
MATCH_MODES = {"none": "①매칭없음", "edit": "②편집거리", "edit+attr": "③편집거리+색모양"}


def identify(pred, shape, color, db, mode):
    """읽은 각인(pred)으로 약 이름을 판별. 판별 못하면 '' 반환.
      none      : 읽은 글자가 DB 각인과 정확히 같을 때만 인정
      edit      : DB에서 가장 비슷한 각인을 고름
      edit+attr : 모양·색이 같은 후보로 먼저 좁히고 가장 비슷한 각인을 고름
                  (후보가 1개만 남으면 글자를 못 읽어도 그 약으로 판별)"""
    cands = db
    if mode == "edit+attr":
        cands = [d for d in db if d["shape"] == shape and COLOR_GROUP.get(d["color"], d["color"]) == color] or db
        if len(cands) == 1:
            return cands[0]["name"]
    if mode == "none":
        hit = [d for d in cands if d["imprint"] == pred]
        return hit[0]["name"] if len(hit) == 1 else ""
    if not pred:
        return ""
    scores = sorted(((similarity(pred, d["imprint"]), d["name"]) for d in cands), reverse=True)
    if scores[0][0] < MIN_SIM or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        return ""   # 충분히 비슷한 게 없거나 1등이 동점 → 판별 불가
    return scores[0][1]


def char_acc(pred, gt):
    """문자 정확도 = 1 - (틀린 글자 수 / 정답 글자 수).  예) 정답 ALZA27, 인식 ALZA27P → 83%"""
    return max(0.0, 1 - levenshtein(pred, gt) / max(len(gt), 1))


def error_type(pred, gt, all_texts):
    """왜 틀렸는지 분류 → 어디를 고쳐야 할지 알려줌"""
    if pred == gt:
        return "정답"
    if not pred:
        return "빈 결과(아무것도 못 읽음)"
    if gt in all_texts:
        return "방향/선택 실패(후보엔 정답 있었음)"
    if char_acc(pred, gt) >= 0.5:
        return "부분 인식(헷갈리는 글자)"
    return "오인식"


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 8. run : 실험 실행                                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def run(labels_path, db_path, engine_name, out_dir, methods, gpu=False, step=30, tess_cmd=None):
    out_dir = out_dir or f"results_{engine_name}"
    os.makedirs(out_dir, exist_ok=True)

    # labels.csv : 이미지경로,정답각인,각인유형(인쇄/음각, 선택)
    items = [(r[0].strip(), r[1].strip(), r[2].strip() if len(r) > 2 else "") for r in load_csv(labels_path)]
    # pill_db.csv : 약이름,각인,모양,색  (첫 줄은 제목)
    db = [{"name": r[0].strip(), "imprint": norm(r[1]), "shape": r[2].strip(), "color": r[3].strip()}
          for r in load_csv(db_path)[1:]]
    name_of = {d["imprint"]: d["name"] for d in db}

    print(f"[run] 엔진={engine_name}, 전처리={methods}, 이미지 {len(items)}장, 후보 약 {len(db)}종")
    ocr = EasyOCREngine(gpu=gpu) if engine_name == "easyocr" else TesseractEngine(tess_cmd)

    rows = []
    for idx, (path, gt_raw, ptype) in enumerate(items, 1):
        bgr = imread_any(path)
        if bgr is None:
            print("  이미지 없음:", path)
            continue
        gt = norm(gt_raw)
        gray, aspect = isolate_pill(bgr)            # 배경 지우기
        shape, color = pill_features(bgr)           # 모양·색
        auto_type, dark = detect_imprint_type(gray) # 인쇄/음각 자동 판정

        for key in methods:
            label, fn = PREPROCESS[key]
            if fn is None:                          # AUTO: 판정 결과에 맞는 전처리 선택
                fn = AUTO_RULE[auto_type]
            t0 = time.perf_counter()
            pred, all_texts = read_imprint(gray, aspect, fn, ocr, step)
            ms = (time.perf_counter() - t0) * 1000

            row = {"image": path, "type": ptype, "auto_type": auto_type, "gt": gt,
                   "method": label, "pred": pred,
                   "exact": int(pred == gt),                       # 글자 완전 일치
                   "char_acc": round(char_acc(pred, gt), 3),       # 문자 정확도
                   "oracle": int(gt in all_texts),                 # 후보 중 정답 있었나 (방향만 맞으면 됐나)
                   "shape": shape, "color": color, "ms": round(ms, 1),
                   "error": error_type(pred, gt, all_texts)}
            for mode in MATCH_MODES:                               # 약 판별 3종
                got = identify(pred, shape, color, db, mode)
                row[f"id_{mode}"] = got
                row[f"ok_{mode}"] = int(got != "" and got == name_of.get(gt))
            rows.append(row)
            print(f"  ({idx}/{len(items)}) {os.path.basename(path):20s} {label:16s} 정답={gt:8s} "
                  f"인식={pred or '-':10s} 판별(①②③)={row['ok_none']}{row['ok_edit']}{row['ok_edit+attr']}  {ms:5.0f}ms")

    if not rows:
        print("결과가 없습니다. labels.csv 경로를 확인하세요.")
        return
    save_results(rows, methods, out_dir, engine_name, len(items))


def _write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글 안 깨짐
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)


def save_results(rows, methods, out_dir, engine_name, n_img):
    labels = [PREPROCESS[k][0] for k in methods]
    pct = lambda xs: round(100 * float(np.mean(xs)), 1) if xs else 0.0

    _write_csv(os.path.join(out_dir, "detail.csv"), rows)   # 이미지 × 전처리별 전체 기록

    # 전처리 × 매칭 조합별 평균
    summary = []
    for lab in labels:
        r = [x for x in rows if x["method"] == lab]
        for mode, mlab in MATCH_MODES.items():
            summary.append({"전처리": lab, "매칭": mlab, "n": len(r),
                            "글자완전일치%": pct([x["exact"] for x in r]),
                            "문자정확도%": pct([x["char_acc"] for x in r]),
                            "방향정답가정%": pct([x["oracle"] for x in r]),
                            "약판별Top1%": pct([x[f"ok_{mode}"] for x in r]),
                            "평균ms": round(float(np.mean([x["ms"] for x in r])), 1)})
    _write_csv(os.path.join(out_dir, "summary.csv"), summary)

    # 인쇄/음각 유형별 평균 (labels.csv 3번째 칸 기준, 없으면 자동 판정 기준)
    by_type = []
    for t in sorted({x["type"] or x["auto_type"] for x in rows}):
        for lab in labels:
            r = [x for x in rows if (x["type"] or x["auto_type"]) == t and x["method"] == lab]
            by_type.append({"유형": t, "전처리": lab, "n": len(r),
                            "문자정확도%": pct([x["char_acc"] for x in r]),
                            **{f"Top1_{m}": pct([x[f'ok_{k}'] for x in r]) for k, m in MATCH_MODES.items()}})
    _write_csv(os.path.join(out_dir, "by_type.csv"), by_type)

    # 오류 원인별 개수
    errs = [{"전처리": lab, "오류원인": k, "개수": v}
            for lab in labels for k, v in Counter(x["error"] for x in rows if x["method"] == lab).most_common()]
    _write_csv(os.path.join(out_dir, "errors.csv"), errs)

    # 인쇄/음각 자동 판정 정확도 (labels.csv에 유형을 적은 경우만)
    first = [x for x in rows if x["method"] == labels[0] and x["type"]]
    if first:
        ok = sum(x["type"] == x["auto_type"] for x in first)
        print(f"\n[인쇄/음각 자동 구분] {ok}/{len(first)} 정확")

    print(f"\n=== 요약 (엔진: {engine_name}, 알약 {n_img}개) ===")
    for s in summary:
        print(f"{s['전처리']:16s} {s['매칭']:12s} 글자완전일치 {s['글자완전일치%']:5.1f}%  "
              f"문자정확도 {s['문자정확도%']:5.1f}%  약판별 Top-1 {s['약판별Top1%']:5.1f}%  {s['평균ms']:.0f}ms")
    plot_summary(summary, labels, os.path.join(out_dir, "chart_summary.png"), engine_name, n_img)
    print(f"\n결과 저장: {out_dir}/ (summary.csv, by_type.csv, errors.csv, detail.csv, chart_summary.png)")


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 9. 그래프                                                           ║
# ╚════════════════════════════════════════════════════════════════════╝

def _setup_font():
    """한글 폰트 자동 설정 (윈도우: 맑은 고딕, 맥: AppleGothic)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    # 코랩(리눅스)처럼 새로 설치한 폰트가 목록에 아직 없을 수 있어 파일 경로로 직접 등록
    for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_summary(summary, labels, path, engine_name, n):
    plt = _setup_font()
    modes = list(MATCH_MODES.values())
    colors = ["#c9d8ee", "#7aa6e0", "#2a78d6"]   # 같은 파랑 계열 진하기로 매칭 단계 표현
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.6, 1]})

    ax = axes[0]   # 왼쪽: 전처리별 × 매칭별 약 판별 Top-1
    x = np.arange(len(labels))
    bw = 0.27
    for i, mode in enumerate(modes):
        vals = [next(s["약판별Top1%"] for s in summary if s["전처리"] == m and s["매칭"] == mode) for m in labels]
        bars = ax.bar(x + (i - 1) * bw, vals, bw - 0.02, color=colors[i], label=mode)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n", 1) for m in labels], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("약 판별 Top-1 정확도 (%)", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[1]   # 오른쪽: 알약 1개당 처리 시간
    ms = [next(s["평균ms"] for s in summary if s["전처리"] == m) for m in labels]
    names = [m.split("_")[0] for m in labels]
    ax.barh(names[::-1], ms[::-1], height=0.45, color="#2a78d6")
    for i, v in enumerate(ms[::-1]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(ms) * 1.3)
    ax.set_title("알약 1개당 처리시간 (ms)", fontsize=11)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"각인 인식 실험 결과 (OCR 엔진: {engine_name}, 알약 {n}개)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="white")
    plt.close()


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 10. check : 설치 확인                                               ║
# ╚════════════════════════════════════════════════════════════════════╝

def check():
    print(f"OpenCV  : {cv2.__version__}")
    print(f"NumPy   : {np.__version__}")
    try:
        import matplotlib
        print(f"Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib: 없음 → pip install matplotlib")
    try:
        import easyocr, torch
        print(f"EasyOCR : {easyocr.__version__}  (PyTorch {torch.__version__}, GPU 사용 가능: {torch.cuda.is_available()})")
    except ImportError:
        print("EasyOCR : 없음 → pip install easyocr")
    try:
        import pytesseract
        print(f"Tesseract: {pytesseract.get_tesseract_version()}")
    except Exception:
        print("Tesseract: 없음 (비교용이라 없어도 됨)")


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 명령어 정리                                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

def main():
    ap = argparse.ArgumentParser(description="알약 각인 인식 실험")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="설치 확인")

    f = sub.add_parser("fuse", help="조명 방향만 바꿔 찍은 사진들을 1장으로 합성")
    f.add_argument("--inputs", nargs="+", required=True,
                   help="사진 경로들 (순서: 위 오른쪽 아래 왼쪽). 예: shots/up.jpg shots/right.jpg ... 또는 'shots/*.jpg'")
    f.add_argument("--out", default="fused")
    f.add_argument("--no-align", action="store_true", help="흔들림 보정 끄기")

    c = sub.add_parser("crop", help="여러 알약 사진에서 알약을 1개씩 잘라내기")
    c.add_argument("--input", required=True)
    c.add_argument("--out", default="crops")

    r = sub.add_parser("run", help="각인 인식 실험 실행")
    r.add_argument("--labels", default="labels.csv", help="이미지경로,정답각인,유형")
    r.add_argument("--db", default="pill_db.csv", help="약이름,각인,모양,색")
    r.add_argument("--engine", choices=["easyocr", "tesseract"], default="easyocr")
    r.add_argument("--methods", nargs="+", default=["A", "D", "AUTO"], choices=list(PREPROCESS),
                   help="비교할 전처리 (기본: A D AUTO). 전부: A B C D AUTO")
    r.add_argument("--step", type=int, default=30, help="원형 알약 회전 간격(도). 크게 하면 빨라짐. 기본 30")
    r.add_argument("--out", default=None, help="결과 폴더 (기본: results_엔진이름)")
    r.add_argument("--gpu", action="store_true", help="EasyOCR에서 GPU 사용")
    r.add_argument("--tesseract-cmd", default=None, help="윈도우에서 tesseract.exe 경로")

    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "fuse":
        paths = [p for pat in a.inputs for p in (sorted(glob.glob(pat)) or [pat])]
        fuse_lights(paths, a.out, align=not a.no_align)
    elif a.cmd == "crop":
        crop_pills(a.input, a.out)
    else:
        run(a.labels, a.db, a.engine, a.out, a.methods, a.gpu, a.step, a.tesseract_cmd)


if __name__ == "__main__":
    main()
