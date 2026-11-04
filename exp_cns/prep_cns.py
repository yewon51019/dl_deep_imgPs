# =====================================================================
#  prep_cns.py  —  1단계: 논문(Al-Hussaeni 외 2023) 전처리 재현 + 점 제거(획 보호), 평가용 432장
# =====================================================================
#
#  [논문 전처리 흐름 (그림 그대로)]
#     흑백 변환 → ┬ 색   : 가우시안(흐림·잡음 제거) → 평균 필터(부드럽게) → 히스토그램(대비 정규화)
#                 ├ 형태 : 소벨(경계) → 영역 분할 → 히스토그램(대비 정규화)
#                 └ 각인 : 캐니(경계) → 팽창 → SIFT (+ 본문의 MLBP)
#     논문에 파라미터 값이 거의 없어서, 흔히 쓰는 값 주변 후보를 모두 시험한다.
#  [추가 갈래 11 점 제거]  사용자 제안 방식: ① 획(각인)을 먼저 찾아 보호 → ② 보호 구역 밖의 '어둡고 색이 튀는 작은 점'만 inpaint.
#     원래 제안의 '갈색' 조건은 색 차이의 크기 √(Δa²+Δb²) 로 일반화 (파란 반점·검은 점도 잡히게).
#     크기 기준(획 길이 18px, 점 넓이 60px)은 이미지 크기에 따라 의미가 바뀌므로 긴 변 400px 로 맞춘 뒤 적용.
#  [한 번에 하나만 바꾸기]  모든 조합을 곱하면 수만 가지 → 한 단계의 값만 바꾸고 나머지는 DEFAULT 로 고정 (경우 수 = 합)
#     각 폴더 = '그 단계까지'의 결과 (예: 02_mean_k5 = 흑백 → 가우시안(기본) → 평균 k5)
#  [크기]  흑백 변환 직후 비율 유지로 227×227 (논문 CNN 입력). 11번만 400 에서 처리한 뒤 227 로 줄임.
#  [알약 마스크 (통계·그림 확인용, 4중 안전장치)]
#     ① 네 모서리 칸으로 배경색 추정 (크롭이 알약에 딱 맞아 테두리 띠는 알약이 섞임)
#     ② 넉넉 자르기 방식(낮은 기준으로 붙어 있는 덩어리 추가) + 볼록 껍질
#     ③ (배경 칠하기는 CNN 단계에서: 마스크 밖 + 배경색과 비슷한 픽셀만)  ④ 넓이 60% 미만이면 실패로 표시
#  [저장]  --out (기본 exp_cns\results\prep, --sample 이면 prep_sample)
#     NN_이름_파라미터\  : 원본과 같은 파일명 PNG (글자 안 씀 — 2단계 OCR 입력이 되므로) + params.json + stats.csv
#     masks\             : 알약 마스크 테두리 그림 + mask_stats.csv
#     sheets\            : 단계별 비교 시트(샘플 6장 × 그 단계 경우들), 전체 한눈에 보기, 마스크 의심 사진, sample_rows.csv
#     prep_index.csv     : 경우마다 파라미터·한 장당 시간(평균·중앙값·95%·최대)·통계 평균·용량
#  [시간 기록]  step_ms = 그 단계 연산만 / chain_ms = 흑백 변환부터 그 결과까지 전체 / load_ms·save_ms 따로.
#     첫 장은 준비 시간(워밍업)이 섞여 평균에서 빼고 warmup_ms 로 따로 적음.
#  [샘플]  72장 = 약 36종 × 카메라 75°·90°, 각각 앞면 우선 → 회전각 가장 작은 사진 (규칙이라 항상 같은 사진)
#  [실행 예]  python exp_cns\prep_cns.py --sample          ← 72장으로 빠르게 시험 (prep_sample 폴더)
#             python exp_cns\prep_cns.py                   ← 평가용 432장 전부
#             python exp_cns\prep_cns.py --only 03,11      ← 일부 단계만 (이미 있는 폴더는 건너뜀, 다시 만들려면 --force)
# =====================================================================

import argparse
import csv
import json
import os
import platform
import time
from collections import defaultdict

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

import common_cns as cc

# ---------------------------------------------------------------------
# 기본값: 다른 단계를 바꿔 볼 때 고정해 두는 값 (PDF 4.1 의 출발값 + 점 제거 제안값)
# ---------------------------------------------------------------------
DEFAULT = dict(
    gauss_k=5, gauss_s=1.0,          # 가우시안: 5×5 창, σ 1.0
    mean_k=3,                        # 평균 필터: 3×3
    hist="clahe", clip=2.0, tile=8,  # 히스토그램: CLAHE 대비 상한 2.0, 8×8 칸
    sobel_k=3,                       # 소벨 커널 3
    close_k=5,                       # 분할 후 닫기(구멍 메우기) 커널 5
    canny="50_150",                  # 캐니 아래/위 기준
    dil_k=3, dil_it=1,               # 팽창 3×3, 1회
    sift_ct=0.04,                    # SIFT contrastThreshold (OpenCV 기본)
    lbp_r=1,                         # LBP 반지름
    spot_h=12, spot_bh=13, spot_len=18, spot_dark=6, spot_color=6, spot_area=60, spot_final=6,
)
SIDE = 227          # 논문 CNN 입력 크기
SPOT_SIDE = 400     # 점 제거는 이 크기에서 (크기 기준을 일정하게)
MASK_FAIL = 0.60    # 크롭이 알약에 딱 맞으므로 알약 넓이가 이보다 작으면 마스크 실패로 봄 (원형 내접원 ≈ 0.785)


def make_variants():
    """(폴더 이름, 그룹, 목표 단계, 기본값에서 바꾼 값, 샘플만 저장?) 목록. 한 번에 하나만 바꾸기."""
    v = [("00_gray", "00", "gray", {}, False)]
    for k in (3, 5, 7):                                            # 01 가우시안 9가지
        for s in (0.5, 1.0, 2.0):
            v.append((f"01_gauss_k{k}_s{s}", "01", "gauss", {"gauss_k": k, "gauss_s": s}, False))
    for k in (3, 5):                                               # 02 평균 2가지
        v.append((f"02_mean_k{k}", "02", "mean", {"mean_k": k}, False))
    hists = [("he", {"hist": "he"})] + [(f"clahe_c{c:g}_t{t}", {"hist": "clahe", "clip": c, "tile": t})
                                         for c in (1.0, 2.0, 4.0) for t in (4, 8)]
    for name, o in hists:                                          # 03 색 갈래 히스토그램 7가지
        v.append((f"03_hist_{name}", "03", "hist", o, False))
    for k in (3, 5):                                               # 04 소벨 2가지
        v.append((f"04_sobel_k{k}", "04", "sobel", {"sobel_k": k}, False))
    for k in (0, 5, 9):                                            # 05 분할 3가지 (0 = 닫기 안 함)
        v.append((f"05_seg_otsu{'' if k == 0 else f'_close{k}'}", "05", "seg", {"close_k": k}, False))
    for name, o in hists:                                          # 06 형태 갈래 히스토그램 7가지
        v.append((f"06_shapehist_{name}", "06", "shapehist", o, False))
    for c in ("30_90", "50_150", "100_200", "auto"):               # 07 캐니 4가지
        v.append((f"07_canny_{c}", "07", "canny", {"canny": c}, False))
    for k in (3, 5):                                               # 08 팽창 4가지
        for it in (1, 2):
            v.append((f"08_dilate_k{k}_i{it}", "08", "dilate", {"dil_k": k, "dil_it": it}, False))
    for ct in (0.04, 0.02):                                        # 09 SIFT 2가지 (특징점 그림이라 샘플만)
        v.append((f"09_sift_ct{ct}", "09", "sift", {"sift_ct": ct}, True))
    for r in (1, 2, 3):                                            # 10 MLBP 3가지
        v.append((f"10_mlbp_r{r}", "10", "mlbp", {"lbp_r": r}, False))
    spots = [("default", {}), ("h8", {"spot_h": 8}), ("bh9", {"spot_bh": 9}), ("bh17", {"spot_bh": 17}),
             ("len12", {"spot_len": 12}), ("len24", {"spot_len": 24}), ("dark4", {"spot_dark": 4}),
             ("dark8", {"spot_dark": 8}), ("color4", {"spot_color": 4}), ("area40", {"spot_area": 40}),
             ("area100", {"spot_area": 100}), ("final0", {"spot_final": 0}), ("final4", {"spot_final": 4})]
    for name, o in spots:                                          # 11 점 제거 13가지
        v.append((f"11_spot_{name}", "11", "spot", o, False))
    return v


# ---------------------------------------------------------------------
# 알약 마스크 (4중 안전장치 중 ①②④. ③ 배경 칠하기는 CNN 단계에서 사용)
# ---------------------------------------------------------------------
def corner_bg(lab):
    """① 네 모서리 칸(한 변의 8%)의 Lab 중앙값으로 배경색 추정.
       한 모서리에 알약이 걸쳐도 버티도록, 서로 색 거리 15 안쪽인 '비슷한 모서리'가 가장 많은 묶음만 평균.
       → (배경 Lab, 사용한 모서리 수)"""
    h, w = lab.shape[:2]
    c = max(3, int(round(min(h, w) * 0.08)))
    patches = [lab[:c, :c], lab[:c, -c:], lab[-c:, :c], lab[-c:, -c:]]
    meds = np.array([np.median(p.reshape(-1, 3), axis=0) for p in patches])
    dist = np.linalg.norm(meds[:, None] - meds[None], axis=2)      # 모서리끼리 색 거리 (4×4)
    support = (dist < 15).sum(axis=1)                              # 나와 비슷한 모서리 수 (자기 자신 포함)
    keep = support == support.max()
    return meds[keep].mean(axis=0), int(keep.sum())


def pill_mask(bgr):
    """② 배경색과의 색 차이로 알약을 찾고, 낮은 기준으로 붙어 있는 옅은 부분(진한 캡슐 절반 등)을 더한 뒤 볼록 껍질로 채움.
       ④ 넓이가 크롭의 60% 미만이면 실패(fail=1).  → (마스크 uint8 0/255, 넓이 비율, 실패, 배경 Lab, 사용 모서리 수)"""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bg, n_corner = corner_bg(lab)
    d = np.clip(np.linalg.norm(lab - bg, axis=2) * 2, 0, 255).astype(np.uint8)   # 배경색과의 거리 (×2 로 대비 키움)
    thr, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)       # Otsu: 히스토그램을 둘로 가르는 기준 자동
    t_main = max(thr, 20)                                                        # 너무 작은 차이는 무시
    k5 = np.ones((5, 5), np.uint8)
    main = cv2.morphologyEx((d > t_main).astype(np.uint8), cv2.MORPH_OPEN, k5)  # 열기: 작은 점 잡티 제거
    c = max(3, int(round(min(h, w) * 0.08)))
    corner_d = np.concatenate([d[:c, :c].ravel(), d[:c, -c:].ravel(), d[-c:, :c].ravel(), d[-c:, -c:].ravel()])
    t_low = min(t_main, max(np.percentile(corner_d, 99) + 10, 20))              # 배경 흔들림(99%)보다 조금 큰 낮은 기준
    low = cv2.morphologyEx((d > t_low).astype(np.uint8), cv2.MORPH_OPEN, k5)
    _, lbl = cv2.connectedComponents(low)
    ids = np.setdiff1d(np.unique(lbl[(main > 0) & (low > 0)]), [0])            # 지금 알약 덩어리에 닿은 옅은 덩어리만
    grown = ((main > 0) | np.isin(lbl, ids)).astype(np.uint8) * 255
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # 닫기: 각인 홈 같은 틈 메우기
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    if cnts:
        cv2.drawContours(mask, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)  # 볼록 껍질(고무줄)로 채움
    ratio = float((mask > 0).mean())
    return mask, ratio, int(ratio < MASK_FAIL), bg, n_corner


# ---------------------------------------------------------------------
# 논문 전처리 함수들 (모두 uint8 흑백 입력 → uint8 출력)
# ---------------------------------------------------------------------
def gauss(g, k, s):
    """색 갈래 1 가우시안 블러: 흐림·잡음 제거. k = 창 크기(홀수, 클수록 넓게 흐림), s = σ(흐림 정도)."""
    return cv2.GaussianBlur(g, (k, k), s)


def mean_f(g, k):
    """색 갈래 2 평균 필터: k×k 창의 평균으로 한 번 더 부드럽게."""
    return cv2.blur(g, (k, k))


def _he_masked(g, m):
    """마스크 안쪽 픽셀의 히스토그램만으로 평활화 표(LUT)를 만들어 적용 (배경이 계산에 끼지 않게)."""
    vals = g[m]
    if vals.size == 0:
        return cv2.equalizeHist(g)
    cdf = np.bincount(vals, minlength=256).cumsum()                # 누적 분포
    cmin = cdf[cdf > 0][0]
    lut = np.clip(np.round((cdf - cmin) / max(cdf[-1] - cmin, 1) * 255), 0, 255).astype(np.uint8)
    return lut[g]


def hist(g, mode, clip, tile, mask=None):
    """히스토그램 대비 정규화.
       mode='he'    : 전역 평활화 (사진 전체에 같은 규칙)
       mode='clahe' : 국소 평활화. 사진을 tile×tile 칸으로 나눠 칸마다 평활화, clip = 대비 상한(클수록 세지만 잡티도 커짐)
       mask 가 있으면 알약 안쪽만 계산하고 바깥은 0 (형태 갈래용)."""
    if mode == "he":
        out = _he_masked(g, mask) if mask is not None else cv2.equalizeHist(g)
    else:
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0
    return out


def sobel(g, k):
    """형태 갈래 1 소벨: x·y 방향 밝기 기울기의 크기(= 경계 세기). CV_32F 로 계산해야 음수 기울기가 잘리지 않음.
       사진마다 0~255 로 늘려서 저장."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=k)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def segment(sob, close_k):
    """형태 갈래 2 영역 분할: 소벨 경계를 Otsu 로 이진화 → 닫기(close_k, 0 이면 안 함)로 끊긴 테두리 잇기 →
       가장 큰 윤곽선 안쪽을 채워 알약 모양 마스크(0/255)."""
    _, b = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if close_k:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(b)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def canny(g, spec):
    """각인 갈래 1 캐니: 기울기 hi 이상은 확실한 경계, lo~hi 는 확실한 경계와 이어질 때만 경계.
       spec='auto' 면 밝기 중앙값 m 의 0.67배·1.33배 (흔히 쓰는 자동 기준)."""
    if spec == "auto":
        m = float(np.median(g))
        lo, hi = int(max(0, 0.67 * m)), int(min(255, 1.33 * m))
    else:
        lo, hi = map(int, spec.split("_"))
    return cv2.Canny(g, lo, hi)


def dilate(e, k, it):
    """각인 갈래 2 팽창(그림의 'Dilution' = dilation 오타로 보임): k×k 로 it 번 굵게 해서 끊긴 획을 이음."""
    return cv2.dilate(e, np.ones((k, k), np.uint8), iterations=it)


def sift_draw(g, edge, ct):
    """각인 갈래 3 SIFT: 팽창한 경계 근처에서만 특징점을 찾아 원(크기·방향 포함)으로 그림. → (컬러 그림, 특징점 수).
       ct = contrastThreshold (작을수록 약한 특징점도 찾음)."""
    kps = cv2.SIFT_create(contrastThreshold=ct).detect(g, mask=edge)
    vis = cv2.drawKeypoints(g, kps, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis, len(kps)


def mlbp(g, r):
    """각인 갈래 4 LBP: 픽셀마다 반지름 r 원 위의 8점과 밝기를 비교한 무늬 번호(uniform, 0~9, 회전 불변).
       반지름 1·2·3 을 합치면 논문의 MLBP(다중 스케일). 보기 좋게 0~255 로 늘려 저장."""
    lbp = local_binary_pattern(g, P=8, R=r, method="uniform")
    return (lbp * (255.0 / 9.0)).astype(np.uint8)


# ---------------------------------------------------------------------
# 11 점 제거 (획 보호 + 점 inpaint)
# ---------------------------------------------------------------------
def _keep_components(binary, ok):
    """연결된 덩어리마다 ok(w, h, area) 가 참인 것만 남김."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = ok(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA])
    return keep[lbl]


def _spots(L, a, b, mask, dark, color, max_area):
    """점 후보: 주변(15×15 중앙값)보다 dark 이상 어둡고 '그리고' 색이 color 이상 튀는 작은 덩어리.
       (색 조건만 쓰면 각인 테두리 색번짐이, 밝기 조건만 쓰면 각인 자체가 걸림)"""
    dark_dev = cv2.medianBlur(L, 15).astype(np.int16) - L.astype(np.int16)
    color_dev = np.hypot(a.astype(np.float32) - cv2.medianBlur(a, 15), b.astype(np.float32) - cv2.medianBlur(b, 15))
    cand = (dark_dev > dark) & (color_dev > color) & mask
    return _keep_components(cand, lambda w, h, area: area <= max_area and max(w, h) <= 10)


def remove_spots(bgr400, mask400, h, bh, min_len, dark, color, max_area, final_h):
    """사용자 제안 점 제거. 입력은 긴 변 400 으로 맞춘 컬러·마스크. → (결과 L, 획 보호 마스크, 점 마스크, 점 넓이 비율)
       ① 획 보호: 강한 NLmeans(h)로 점을 뭉갠 임시 이미지 → black-hat(bh, 주변보다 어두운 선 = 파인 각인) → Otsu →
          길이 min_len 이상 덩어리만 획 → 5 만큼 넓혀 획 테두리까지 보호
       ② 점: _spots, 3 만큼 넓혀 옅은 테두리까지 포함, 획 보호 구역은 제외
       ③ inpaint(주변 픽셀로 구멍 메우기) 2회 (2회차는 기준을 조금 낮춤) → 약한 NLmeans(final_h, 0 이면 안 함)"""
    L, a, b = cv2.split(cv2.cvtColor(bgr400, cv2.COLOR_BGR2LAB))   # L = 밝기, a = 초록↔빨강, b = 파랑↔노랑
    m = mask400 > 0
    tmp = cv2.fastNlMeansDenoising(L, None, h, 7, 21)              # NLmeans: 비슷한 조각끼리 평균 → 흩어진 점은 묻히고 획은 남음
    blackhat = cv2.morphologyEx(tmp, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh, bh)))
    vals = blackhat[m].reshape(-1, 1)
    thr = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if vals.size else 255   # 알약 안쪽만으로 Otsu
    lines = (blackhat > thr) & m
    stroke = _keep_components(lines, lambda w, hh, area: max(w, hh) >= min_len)
    stroke = cv2.dilate(stroke.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    spot = _spots(L, a, b, m, dark, color, max_area)
    spot = cv2.dilate(spot.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    spot &= ~stroke                                                # ★ 획 보호 구역은 절대 안 건드림
    out = cv2.inpaint(L, spot.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    spot2 = _spots(out, a, b, m, dark - 1, color / 2.0, max_area) & ~stroke   # 첫 번에 못 지운 옅은 점 (색은 원본 a·b 기준)
    if spot2.any():
        out = cv2.inpaint(out, spot2.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    if final_h > 0:
        out = cv2.fastNlMeansDenoising(out, None, final_h, 7, 21)   # 남은 자잘한 얼룩 정리 (h 가 크면 획도 뭉개짐)
    ratio = float((spot | spot2).sum() / max(m.sum(), 1))
    return out, stroke, spot | spot2, ratio


# ---------------------------------------------------------------------
# 한 경우(variant) 실행: 기본값 + 바꾼 값으로 목표 단계까지 계산, 단계별 시간 측정
# ---------------------------------------------------------------------
def run_variant(ctx, cfg, target):
    """ctx: 사진 한 장의 준비물(gray227, t_gray, bgr, mask400 등). → (결과 그림, step_ms, chain_ms, 추가 정보)
       chain_ms 는 흑백 변환(t_gray)부터 목표 단계까지 앞 단계를 포함한 전체 시간 (실제 파이프라인 비용)."""
    times = [ctx["t_gray"]]                                        # 각 단계 시간(ms) 누적
    extra = {}

    def timed(fn, *args):                                          # 함수 하나를 재서 times 에 추가
        t0 = time.perf_counter()
        out = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
        return out

    g = ctx["gray"]
    if target == "gray":
        out = g
    elif target in ("gauss", "mean", "hist"):                      # 색 갈래
        out = timed(gauss, g, cfg["gauss_k"], cfg["gauss_s"])
        if target in ("mean", "hist"):
            out = timed(mean_f, out, cfg["mean_k"])
        if target == "hist":
            out = timed(hist, out, cfg["hist"], cfg["clip"], cfg["tile"])
    elif target in ("sobel", "seg", "shapehist"):                  # 형태 갈래
        out = timed(sobel, g, cfg["sobel_k"])
        if target in ("seg", "shapehist"):
            out = timed(segment, out, cfg["close_k"])
        if target == "shapehist":
            out = timed(hist, g, cfg["hist"], cfg["clip"], cfg["tile"], out > 0)
    elif target in ("canny", "dilate", "sift"):                    # 각인 갈래
        out = timed(canny, g, cfg["canny"])
        if target in ("dilate", "sift"):
            out = timed(dilate, out, cfg["dil_k"], cfg["dil_it"])
        if target == "sift":
            out, extra["n_keypoints"] = timed(sift_draw, g, out, cfg["sift_ct"])
    elif target == "mlbp":
        out = timed(mlbp, g, cfg["lbp_r"])
    elif target == "spot":                                         # 점 제거: 400 에서 처리 후 227 로
        times = [ctx["t_400"]]                                     # 이 갈래의 출발점은 컬러 400 맞춤
        L, _, spot, ratio = timed(remove_spots, ctx["bgr400"], ctx["mask400"], cfg["spot_h"], cfg["spot_bh"],
                                  cfg["spot_len"], cfg["spot_dark"], cfg["spot_color"], cfg["spot_area"], cfg["spot_final"])
        step = times[-1]
        out = timed(cc.fit_side, L, SIDE)
        extra["spot_ratio"] = round(ratio, 4)
        return out, step, sum(times), extra
    return out, times[-1], sum(times), extra


def stats(img, mask):
    """알약 안쪽 통계 (그림 확인용 숫자, 점수 아님): 선명도(라플라시안 분산), 대비(표준편차), 경계 밀도(캐니 50/150 비율)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if not mask.any():
        mask = np.ones_like(mask)
    lap = cv2.Laplacian(g, cv2.CV_64F)                              # CV_64F: 음수도 담는 실수형 (8비트면 음수가 잘려 값이 왜곡됨)
    return {"lap_var": round(float(lap[mask].var()), 1), "std": round(float(g[mask].std()), 2),
            "edge_density": round(float((cv2.Canny(g, 50, 150) > 0)[mask].mean()), 4)}


# ---------------------------------------------------------------------
# 샘플 고르기 (규칙 기반, 무작위 아님)
# ---------------------------------------------------------------------
def pick_sample72(items):
    """약 × 카메라마다 1장: 앞면 우선 → 회전각이 가장 작은 사진 → 파일명 순."""
    best = {}
    for it in items:
        key = (it["drug_N"], it["camera_la"])
        rank = (it["face"] != "앞면", it["rot"], it["file_name"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, it)
    return sorted((v[1] for v in best.values()), key=lambda x: x["file_name"])


def pick_rows6(sample, spot_ratio, lap_var):
    """비교 시트의 6행: 조건이 다른 6장. 라벨 조건(1~4)은 아직 안 쓴 배경색을 우선, 측정 조건(5·6)은 값 순위.
       → [(사진, 이유)]"""
    rows, used, bgs = [], set(), set()
    conds = [("흰 원형 정제", lambda x: x["color1"] == "하양" and x["shape"] == "원형" and "캡슐" not in x["form"]),
             ("색 있는 정제", lambda x: x["color1"] != "하양" and "캡슐" not in x["form"]),
             ("두 색 경질캡슐", lambda x: "경질캡슐" in x["form"] and bool(x["color2"])),
             ("연질캡슐", lambda x: "연질캡슐" in x["form"])]
    for why, ok in conds:
        cand = [x for x in sample if ok(x) and x["file_name"] not in used]
        if not cand:
            continue
        pick = next((x for x in cand if x["back_color"] not in bgs), cand[0])   # 배경이 섞이도록
        rows.append((pick, why))
        used.add(pick["file_name"])
        bgs.add(pick["back_color"])
    for why, score, rev in (("점이 가장 많음", spot_ratio, True), ("각인이 가장 흐림(선명도 최하)", lap_var, False)):
        cand = sorted((x for x in sample if x["file_name"] not in used and x["file_name"] in score),
                      key=lambda x: score[x["file_name"]], reverse=rev)
        if cand:
            rows.append((cand[0], f"{why} ({score[cand[0]['file_name']]})"))
            used.add(cand[0]["file_name"])
    return rows


# ---------------------------------------------------------------------
# 비교 시트 (제목은 시트에만 씀 — OpenCV 글씨는 영문만 가능)
# ---------------------------------------------------------------------
CELL = 150


def _cell(img, title):
    """그림 한 칸: 150×150 으로 줄이고 위에 제목 줄."""
    im = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    im = cv2.resize(im, (CELL, CELL), interpolation=cv2.INTER_AREA)
    bar = np.full((22, CELL, 3), 245, np.uint8)
    cv2.putText(bar, title[:24], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def _label_cell(lines):
    """행 맨 앞의 설명 칸 (파일 정보)."""
    c = np.full((CELL + 22, 170, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(c, t[:26], (4, 30 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return c


def _short(folder):
    """폴더 이름 → 시트 칸 제목 (예: 03_hist_clahe_c2_t8 → '03 hist clahe c2 t8')."""
    return folder.replace("_", " ")


def make_sheets(out_root, variants, rows6, sample72, mask_rows):
    """단계별 시트(6행 × 그 단계 경우들, 맨 앞에 00 흑백), 전체 한눈에 보기(1행 사진 × 모든 경우), 마스크 의심 사진."""
    sd = os.path.join(out_root, "sheets")
    os.makedirs(sd, exist_ok=True)
    have = [v for v in variants if os.path.isdir(os.path.join(out_root, v[0]))]

    def load(folder, fn):
        p = os.path.join(out_root, folder, fn)
        return cc.imread(p) if os.path.exists(p) else None

    groups = sorted({v[1] for v in have if v[1] != "00"})
    for gname in groups:                                           # 단계별 시트
        cols = [v[0] for v in have if v[1] == "00"] + [v[0] for v in have if v[1] == gname]
        lines = []
        for it, why in rows6:
            cells = [_label_cell([it["drug_N"], f"rot{it['rot']} cam{it['camera_la']}",
                                  f"truth {it['truth'][:14]}", why.encode("ascii", "ignore").decode()[:24]])]
            for f in cols:
                im = load(f, it["file_name"])
                cells.append(_cell(im if im is not None else np.zeros((10, 10), np.uint8), _short(f)))
            lines.append(np.hstack(cells))
        if lines:
            w = max(l.shape[1] for l in lines)
            lines = [np.hstack([l, np.full((l.shape[0], w - l.shape[1], 3), 255, np.uint8)]) for l in lines]
            name = next(v[0] for v in have if v[1] == gname).split("_")[1]
            cc.imwrite(os.path.join(sd, f"sheet_{gname}_{name}.png"), np.vstack(lines))
    if rows6:                                                      # 전체 한눈에 보기: 첫 행 사진의 모든 경우, 10열
        it = rows6[0][0]
        cells = [_cell(load(v[0], it["file_name"]), _short(v[0])) for v in have if load(v[0], it["file_name"]) is not None]
        per = 10
        grid = []
        for i in range(0, len(cells), per):
            row = cells[i:i + per]
            row += [np.full_like(cells[0], 255)] * (per - len(row))
            grid.append(np.hstack(row))
        cc.imwrite(os.path.join(sd, "sheet_overview_1.png"), np.vstack(grid))
    sus = sorted(mask_rows, key=lambda r: r["area_ratio"])[:24]     # 마스크 넓이 비율이 가장 작은 24장 = 의심 사진
    cells = []
    for r in sus:
        p = os.path.join(out_root, "masks", r["file_name"])
        if os.path.exists(p):
            cells.append(_cell(cc.imread(p), f"{r['drug_N']} {r['area_ratio']:.2f}{' FAIL' if r['fail'] else ''}"))
    if cells:
        per = 8
        grid = [np.hstack(cells[i:i + per] + [np.full_like(cells[0], 255)] * (per - len(cells[i:i + per])))
                for i in range(0, len(cells), per)]
        cc.imwrite(os.path.join(sd, "sheet_masks_suspect.png"), np.vstack(grid))
    with open(os.path.join(sd, "sample_rows.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "file_name", "drug_N", "dl_name", "back_color", "reason"])
        for i, (it, why) in enumerate(rows6, 1):
            w.writerow([i, it["file_name"], it["drug_N"], it["dl_name"], it["back_color"], why])
    with open(os.path.join(sd, "sample72.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "drug_N", "face", "camera_la", "rot"])
        for it in sample72:
            w.writerow([it["file_name"], it["drug_N"], it["face"], it["camera_la"], it["rot"]])


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
def summarize(values):
    """한 장당 시간 요약 (첫 장 = 워밍업은 빼고). → 평균, 중앙값, 95%, 최대, 워밍업"""
    v = np.array(values[1:] if len(values) > 1 else values, float)
    return (round(float(v.mean()), 3), round(float(np.median(v)), 3), round(float(np.percentile(v, 95)), 3),
            round(float(v.max()), 3), round(float(values[0]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="샘플 72장만 (exp_cns\\results\\prep_sample)")
    ap.add_argument("--only", default="", help="일부 그룹만, 예: 03,11")
    ap.add_argument("--force", action="store_true", help="이미 있는 폴더도 다시 만듦")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out_root = a.out or str(cc.RESULTS / ("prep_sample" if a.sample else "prep"))
    items = cc.load_items(cc.EVAL_ROOT)                              # 평가용 432장 (1단계는 점수를 매기지 않음)
    cc.write_items(items, str(cc.RESULTS / "items_eval.csv"))
    sample72 = pick_sample72(items)
    sample_names = {it["file_name"] for it in sample72}
    run_items = sample72 if a.sample else items
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    variants = [v for v in make_variants() if not only or v[1] in only or v[1] == "00"]
    todo = []
    for v in variants:                                             # 덮어쓰기 방지: 이미 만든 폴더(params.json 있음)는 건너뜀
        if os.path.exists(os.path.join(out_root, v[0], "params.json")) and not a.force:
            print(f"[건너뜀] {v[0]} (이미 있음, 다시 만들려면 --force)")
        else:
            todo.append(v)
    print(f"[1단계] 사진 {len(run_items)}장 × 경우 {len(todo)}가지 → {out_root}")

    env = {"cpu": platform.processor(), "opencv": cv2.__version__, "numpy": np.__version__,
           "cv2_threads": cv2.getNumThreads(), "date": time.strftime("%Y-%m-%d %H:%M")}
    rows = defaultdict(list)                                       # 경우마다 사진별 기록
    mask_rows, lap_gray, spot_ratio = [], {}, {}
    t_start = time.time()
    for n, it in enumerate(run_items):
        t0 = time.perf_counter()
        bgr = cc.imread(it["image_path"])
        load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()                                   # 흑백 변환 + 227 맞춤 = 모든 갈래의 출발점
        gray = cc.fit_side(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), SIDE)
        t_gray = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()                                   # 마스크 (통계·그림 확인용, 시간 따로 기록)
        mask, ratio, fail, bg, n_corner = pill_mask(bgr)
        mask_ms = (time.perf_counter() - t0) * 1000
        mask227 = cc.fit_side(mask, SIDE, fill=0, interp_mask=True) > 0
        t0 = time.perf_counter()                                   # 점 제거 갈래의 출발점: 컬러·마스크 400 맞춤
        bgr400 = cc.fit_side(bgr, SPOT_SIDE)
        t_400 = (time.perf_counter() - t0) * 1000
        mask400 = cc.fit_side(mask, SPOT_SIDE, fill=0, interp_mask=True)
        ctx = {"gray": gray, "t_gray": t_gray, "bgr400": bgr400, "mask400": mask400, "t_400": t_400}

        over = bgr.copy()                                          # 마스크 테두리 그림 (초록 = 정상, 빨강 = 실패)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (0, 0, 255) if fail else (0, 255, 0), 2)
        cv2.putText(over, f"ratio {ratio:.2f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cc.imwrite(os.path.join(out_root, "masks", it["file_name"]), over)
        mask_rows.append({"file_name": it["file_name"], "drug_N": it["drug_N"], "area_ratio": round(ratio, 4),
                          "fail": fail, "corners_used": n_corner, "bg_L": round(float(bg[0]), 1),
                          "bg_a": round(float(bg[1]), 1), "bg_b": round(float(bg[2]), 1), "mask_ms": round(mask_ms, 3)})
        lap_gray[it["file_name"]] = stats(gray, mask227)["lap_var"]

        for folder, grp, target, over_cfg, sample_only in todo:
            if sample_only and it["file_name"] not in sample_names:
                continue
            cfg = {**DEFAULT, **over_cfg}
            img, step_ms, chain_ms, extra = run_variant(ctx, cfg, target)
            t0 = time.perf_counter()
            cc.imwrite(os.path.join(out_root, folder, it["file_name"]), img)
            save_ms = (time.perf_counter() - t0) * 1000
            if folder == "11_spot_default":
                spot_ratio[it["file_name"]] = extra.get("spot_ratio", 0)
                if it["file_name"] in sample_names:                # 획 보호·점 마스크 확인 그림: [L | 획 보호 | 점 | 결과]
                    L = cv2.split(cv2.cvtColor(bgr400, cv2.COLOR_BGR2LAB))[0]
                    _, stroke, spot, _ = remove_spots(bgr400, mask400, cfg["spot_h"], cfg["spot_bh"], cfg["spot_len"],
                                                      cfg["spot_dark"], cfg["spot_color"], cfg["spot_area"], cfg["spot_final"])
                    panels = [L, stroke.astype(np.uint8) * 255, spot.astype(np.uint8) * 255, cc.fit_side(img, SPOT_SIDE)]
                    cc.imwrite(os.path.join(out_root, folder, "_masks", it["file_name"]),
                               np.hstack([cv2.resize(p, (SIDE, SIDE), interpolation=cv2.INTER_AREA) for p in panels]))
            rows[folder].append({"file_name": it["file_name"], "step_ms": round(step_ms, 3), "chain_ms": round(chain_ms, 3),
                                 "load_ms": round(load_ms, 3), "save_ms": round(save_ms, 3), **stats(img, mask227), **extra})
        if (n + 1) % 50 == 0 or n + 1 == len(run_items):
            print(f"  {n + 1}/{len(run_items)}장  ({time.time() - t_start:.0f}초)")

    # 저장: 경우마다 stats.csv + params.json, 전체 prep_index.csv, 마스크 표
    index = []
    for folder, grp, target, over_cfg, sample_only in todo:
        r = rows[folder]
        if not r:
            continue
        d = os.path.join(out_root, folder)
        with open(os.path.join(d, "stats.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(r[0]))
            w.writeheader()
            w.writerows(r)
        json.dump({"folder": folder, "group": grp, "target": target, "changed": over_cfg, "config": {**DEFAULT, **over_cfg},
                   "n_images": len(r), "sample_only": sample_only, "env": env},
                  open(os.path.join(d, "params.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        st, ch = summarize([x["step_ms"] for x in r]), summarize([x["chain_ms"] for x in r])
        mb = sum(os.path.getsize(os.path.join(d, x["file_name"])) for x in r) / 1024 / 1024
        index.append({"folder": folder, "group": grp, "changed": json.dumps(over_cfg), "n": len(r),
                      "step_ms_mean": st[0], "step_ms_median": st[1], "step_ms_p95": st[2], "step_ms_max": st[3], "step_ms_warmup": st[4],
                      "chain_ms_mean": ch[0], "chain_ms_median": ch[1], "chain_ms_p95": ch[2], "chain_ms_max": ch[3],
                      "load_ms_mean": round(float(np.mean([x["load_ms"] for x in r])), 3),
                      "save_ms_mean": round(float(np.mean([x["save_ms"] for x in r])), 3),
                      "lap_var_mean": round(float(np.mean([x["lap_var"] for x in r])), 1),
                      "std_mean": round(float(np.mean([x["std"] for x in r])), 2),
                      "edge_density_mean": round(float(np.mean([x["edge_density"] for x in r])), 4),
                      "spot_ratio_mean": round(float(np.mean([x.get("spot_ratio", 0) for x in r])), 4), "MB": round(mb, 1)})
    if index:
        with open(os.path.join(out_root, "prep_index.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
    with open(os.path.join(out_root, "masks", "mask_stats.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0]))
        w.writeheader()
        w.writerows(mask_rows)

    if not spot_ratio:                                             # --only 로 11 을 안 돌렸으면 기존 stats 에서 읽음
        p = os.path.join(out_root, "11_spot_default", "stats.csv")
        if os.path.exists(p):
            spot_ratio = {r["file_name"]: float(r["spot_ratio"]) for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
    rows6 = pick_rows6(sample72, spot_ratio, lap_gray)
    make_sheets(out_root, make_variants(), rows6, sample72, mask_rows)

    print(f"\n{'경우':<26} {'장수':>4} {'단계 ms(평균/중앙/95%)':>24} {'전체 ms(평균)':>12} {'선명도':>8} {'MB':>6}")
    for x in index:
        print(f"{x['folder']:<26} {x['n']:>4} {x['step_ms_mean']:>8.2f}/{x['step_ms_median']:>6.2f}/{x['step_ms_p95']:>7.2f} "
              f"{x['chain_ms_mean']:>12.2f} {x['lap_var_mean']:>8.1f} {x['MB']:>6.1f}")
    fails = sum(r["fail"] for r in mask_rows)
    print(f"\n마스크: 실패(넓이 {MASK_FAIL:.0%} 미만) {fails}/{len(mask_rows)}장, 넓이 비율 최소 {min(r['area_ratio'] for r in mask_rows):.2f}"
          f" / 중앙값 {np.median([r['area_ratio'] for r in mask_rows]):.2f}")
    print("시트 6행:", [f"{it['drug_N']}({why})" for it, why in rows6])
    print(f"총 {time.time() - t_start:.0f}초, 저장 위치: {out_root}")


if __name__ == "__main__":
    main()
