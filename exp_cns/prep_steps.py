# =====================================================================
#  prep_steps.py  —  전처리 단계별 처리 함수 (시트 만들기 · 432장 적용 · 나중의 뷰어가 모두 같은 함수를 씀)
# =====================================================================
#
#  [처리 크기]  모든 단계는 '긴 변 400px' 으로 맞춘 흑백 이미지에서 한다 (SIDE).
#     커널 크기 같은 파라미터가 픽셀 단위라서, 크기를 먼저 고정해야 '5×5' 의 의미가 사진마다 같아진다.
#     227(논문 CNN 입력)로 줄이면 희미한 각인이 더 흐려지므로 OCR 용 전처리는 400 에서 한다.
#  [마스크]  쓰지 않는다 (알약을 절반만 잡는 경우가 많아서). 점 제거도 사진 전체에서 점을 찾는다.
#  [단계]  (사용자와 정한 순서) 1 가우시안 → 2 평균 → 3 히스토그램 → 4 샤프닝 → 5 점 제거
#          단계마다 사람이 시트를 보고 3개를 채택해 다음 단계로 넘긴다 (sweep_cns.py).
#          단계 함수는 모두 'uint8 흑백 → uint8 흑백' 이다.
# =====================================================================

import cv2
import numpy as np

import common_cns as cc

SIDE = 400          # 처리 크기: 긴 변 픽셀 수


def to_gray400(bgr):
    """시작점: 컬러 크롭 → 흑백(gray = 0.299R + 0.587G + 0.114B) → 비율 유지로 긴 변 400, 남는 곳은 모서리 색으로 채운 정사각형."""
    return cc.fit_side(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), SIDE)


# ---------------------------------------------------------------------
# 0단계 후보: 흐림 개선 (크기 맞추기 단계에서) — 원본 초점 흐림 + 작은 사진을 늘린 흐림 대응
# ---------------------------------------------------------------------
_SR = {}


def sr_fit400(bgr):
    """초해상도로 키운 뒤 긴 변 400 정사각형으로 맞춤.
       긴 변이 400 보다 작으면 ESPCN(가벼운 딥러닝 초해상도, OpenCV dnn_superres) ×2·×3·×4 중
       400 이상이 되는 가장 작은 배율로 키우고, 그다음 INTER_AREA 로 400 에 맞춤. 400 이상이면 지금 방식(cc.fit_side)과 같음.
       모델 파일: exp_cns\\models\\ESPCN_x{2,3,4}.pb (공개 모델, GitHub fannymonori/TF-ESPCN)."""
    h, w = bgr.shape[:2]
    if max(h, w) >= SIDE:
        return cc.fit_side(bgr, SIDE)
    scale = min(4, max(2, int(np.ceil(SIDE / max(h, w)))))
    if scale not in _SR:
        m = cv2.dnn_superres.DnnSuperResImpl_create()
        m.readModel(str(cc.ROOT / "exp_cns" / "models" / f"ESPCN_x{scale}.pb"))
        m.setModel("espcn", scale)
        _SR[scale] = m
    return cc.fit_side(_SR[scale].upsample(bgr), SIDE)


def deconv_fit400(bgr, sigma, iters):
    """디컨볼루션(흐림 되돌리기): 지금 방식으로 400 에 맞춘 뒤, 밝기(L)에 Richardson-Lucy 를 적용하고 색은 그대로 합침.
       흐린 사진 = 선명한 사진 * 번짐(PSF) 이라고 가정하고 그 번짐을 거꾸로 되돌림. 번짐은 초점 흐림을 흉내 낸 가우시안(sigma, 400 기준 px).
       sigma : 가정한 번짐 크기. 실제보다 크면 테두리 물결(링잉)이 생김
       iters : 반복 횟수. 키우면 더 선명해지지만 잡음이 커짐"""
    from skimage.restoration import richardson_lucy
    base = cc.fit_side(bgr, SIDE)
    L, a, b = cv2.split(cv2.cvtColor(base, cv2.COLOR_BGR2LAB))
    r = int(np.ceil(3 * sigma))
    k = cv2.getGaussianKernel(2 * r + 1, sigma)
    psf = (k @ k.T).astype(np.float64)                              # 2차원 가우시안 번짐 (합 = 1)
    out = richardson_lucy(L.astype(np.float64) / 255.0, psf, num_iter=iters, clip=True)
    L2 = np.clip(out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------
# 색 채널 고르기 (가우시안 앞): 컬러 3장(B·G·R)을 흑백 1장으로 만들 때 어떤 정보를 쓸지
# ---------------------------------------------------------------------
CHANNELS = ["gray", "L", "R", "G", "B", "min", "auto"]
CHANNEL_DESC = {
    "gray": "사람 눈 밝기 0.299R+0.587G+0.114B (지금까지 방식)",
    "L": "Lab 의 L = 색과 분리한 순수 밝기 (파인 각인의 그늘)",
    "R": "빨강 채널만", "G": "초록 채널만", "B": "파랑 채널만",
    "min": "세 채널 중 가장 어두운 값 (흰 바탕 위 색 글자가 진하게)",
    "auto": "사진마다 위 6가지 중 가운데 영역의 대비(표준편차)가 가장 큰 채널",
}


def channel(bgr, name):
    """컬러 크롭 → 흑백 한 장 (원래 크기 그대로). name 은 CHANNELS 중 하나.
       'auto' 는 (이미지, 고른 채널 이름) 을 돌려주는 channel_auto 를 거쳐 이미지만 반환."""
    if name == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if name == "L":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]          # OpenCV 는 L 을 0~255 로 늘려 줌
    if name in ("R", "G", "B"):
        return np.ascontiguousarray(bgr[..., {"B": 0, "G": 1, "R": 2}[name]])   # OpenCV 컬러 순서는 B, G, R
    if name == "min":
        return bgr.min(axis=2)
    if name == "auto":
        return channel_auto(bgr)[0]
    raise ValueError(f"모르는 채널: {name}")


# 컬러를 유지하는 방식 (흑백 채널이 아니라 컬러 결과를 냄)
COLOR_MODES = ["colorL"]
CHANNEL_DESC["colorL"] = "컬러 유지: Lab 의 L(밝기)에만 처리하고 a·b(색)는 그대로 둔 채 다시 합침"


def on_L(bgr, fn):
    """컬러 유지 처리: BGR → Lab 으로 바꿔 L(밝기)에만 fn 을 적용하고, a·b(색)는 그대로 둔 채 다시 BGR 로 합침.
       색감은 그대로 두고 밝기 대비만 바꾸므로 색이 틀어지지 않음 (R·G·B 를 따로 처리하면 채널마다 늘어나는 정도가 달라 색이 틀어짐).
       L 을 크게 바꾸면 일부 색이 표현 범위를 넘어 살짝 바랠 수 있음 (OpenCV 가 0~255 로 잘라 줌)."""
    L, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([fn(L), a, b]), cv2.COLOR_LAB2BGR)


def recolor(bgr, bright):
    """색 입히기: bright(흑백 한 장, bgr 과 같은 크기)를 밝기(Lab 의 L)로 쓰고, 색(a·b)은 원본 bgr 의 것을 그대로 붙여 컬러로 만듦.
       → 채널마다 '글자 진하기(밝기)'는 그 채널의 것, '색감'은 원본 그대로인 컬러 사진.
       예: R 채널을 밝기로 쓰면 파란 부분은 R 값이 작아 어둡게, 빨간·노란 부분은 밝게 보인다."""
    _, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([bright, a, b]), cv2.COLOR_LAB2BGR)


def channel_auto(bgr):
    """6가지 채널 중 '가운데 절반 영역'의 표준편차(대비)가 가장 큰 채널을 고름 → (이미지, 채널 이름).
       마스크를 쓰지 않으므로 배경이 많은 가장자리를 빼고 각인이 주로 있는 가운데만 봄."""
    h, w = bgr.shape[:2]
    best = None
    for n in CHANNELS[:-1]:
        img = channel(bgr, n)
        sd = float(img[h // 4: h * 3 // 4, w // 4: w * 3 // 4].std())
        if best is None or sd > best[0]:
            best = (sd, img, n)
    return best[1], best[2]


# ---------------------------------------------------------------------
# 6단계 색 특징 더하기 (채널 단계에 붙음): 고른 채널(auto)에 '글자 크기의 색 무늬'를 획으로 더함
#   [왜] auto 는 사진 전체 대비로 채널 하나를 골라서, 두 색 캡슐의 색 인쇄(TRD 캡슐의 빨간 YH, 빨간 반쪽 위 OSTEMIN)가
#        그 채널에서는 바탕과 밝기가 같아 사라짐. 색(Lab a·b)에는 그 글씨가 남아 있으므로 그 무늬만 꺼내 더한다.
# ---------------------------------------------------------------------
def color_detail(bgr, thr, ab_sigma=1.5):
    """색 무늬 지도 D (원래 크기, float): Lab 의 a(빨강↔초록)·b(노랑↔파랑)에서 '글자 획 크기'의 무늬만 꺼냄.
       ab_sigma : a·b 를 먼저 이만큼 흐림 → 점 하나 크기의 색 잡음(무색 알약 표면 알갱이)을 미리 지움 (0 = 안 함)
       띠 대역  : |흐림σ1 − 흐림σ6| → 픽셀 잡음(σ1 보다 작음)과 반쪽 색 차이 같은 큰 변화(σ6 보다 큼)는 빠지고 획 굵기만 남음
       thr (t)  : 이보다 약한 무늬는 0 으로 (무색 알약의 약한 색 흔들림 무시). 키우면 잡음이 줄지만 옅은 색 글씨도 사라짐
       a·b 두 결과를 더해 어떤 색 글씨든 잡음."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = 0
    for i in (1, 2):
        c = lab[..., i]
        if ab_sigma:
            c = cv2.GaussianBlur(c, (0, 0), ab_sigma)
        hp = np.abs(cv2.GaussianBlur(c, (0, 0), 1) - cv2.GaussianBlur(c, (0, 0), 6))
        d = d + np.maximum(hp - thr, 0)
    return d


def add_color(bgr, base, alpha, thr, both=True, ab_sigma=1.5, lo=40, hi=60):
    """채널 사진 base(흑백, bgr 과 같은 크기)에 색 무늬 D 를 세기 alpha 로 더함 → 흑백 uint8.
       both=False (한 방향) : base − α·D  → 색 글씨를 언제나 '어두운 획'으로. 이미 검은 곳(빨간 반쪽의 G 채널 ≈ 0)에서는 효과 없음
       both=True  (양방향)  : base − s·α·D, s = 동네 밝기(흐림σ15)로 정한 부호
                              동네 밝기 ≤ lo → s = −1 (밝게 더함: 어두운 바탕 위 글씨를 밝은 획으로)
                              동네 밝기 ≥ hi → s = +1 (어둡게 더함), 그 사이는 부드럽게 바뀜
                              lo·hi 를 넓게(40~120) 잡았더니 회색 반쪽(80~100)이 s≈0 이 돼 YH 가 사라져서 40~60 으로 좁힘"""
    f = base.astype(np.float32)
    d = color_detail(bgr, thr, ab_sigma if both else 0)
    if both:
        s = np.clip((cv2.GaussianBlur(f, (0, 0), 15) - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1)
        f = f - s * alpha * d
    else:
        f = f - alpha * d
    return np.clip(f, 0, 255).astype(np.uint8)


def _line_kernel(n, ang):
    """길이 n, 굵기 1픽셀의 막대 모양 필터 (각도 ang 도). n×n 칸 가운데를 지나는 직선 위만 1."""
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx, dy = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    for t in np.linspace(-c, c, 4 * n):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return k


def link_strokes(bgr, n=17, low=85, high=97, min_area=20, strength=40, rim_ext=None):
    """획 잇기 (마지막 단계, 컬러 결과에 적용): 끊긴 음각 획을 이어 진하게 하고 바탕 얼룩은 건드리지 않음.
       [왜] 흐리고 작은 음각 사진(E10 등)은 대비를 늘린 뒤 획이 조각조각 끊기고, 표면 얼룩과 진하기가 비슷해짐.
       1) D   = 밝기 L 의 '어두운 가는 구조' max(0, 흐림σ6 − 흐림σ1)  (음각 그늘·얼룩 모두 포함)
       2) R   = 8방향(22.5° 간격) 막대(길이 n) 평균 중 최댓값
                → 길쭉한 획은 한 방향에서 크게 반응해 끊긴 틈이 메워지고, 둥근 얼룩은 어느 방향에서도 반응이 작음
       3) 히스테리시스(8방향 연결): R 이 상위 (100−high)% 인 '확실한 획'에서 출발해, R 이 low 백분위보다 큰 이웃을 따라 이어 붙임.
          확실한 획과 이어지지 않은 덩어리(얼룩)와 min_area 픽셀 미만 조각은 버림
       4) 남은 획 지도 M 을 상위 1% 값이 strength 가 되게 맞춰 L 에서 뺌 (획을 진하게), 색 a·b 는 그대로
       n   : 키우면 더 먼 틈까지 잇지만 획이 굵어지고 이웃 획끼리 붙을 수 있음 (사용자 선택 17, E10 시트 기준)
       low : 키우면 얼룩이 더 잘 빠지지만 약한 획 조각도 사라짐 (85)
       한계: 알약 테두리도 진한 '획'으로 잡혀 같이 진해짐 → rim_ext 로 뺄 수 있음.
       rim_ext : (None = 안 함) 연결 조각의 외접 사각형 긴 변이 '사진 긴 변 × rim_ext' 이상이면 테두리로 보고 진하게 하지 않음.
                 18장 측정: 테두리 조각 0.52~1.00, 글자 조각 0.13~0.43 → 0.6 (0.5 는 YH P-D 540 처럼 테두리에 붙은 글자까지 빠짐).
                 빠진 조각은 지워지는 게 아니라 '더 진하게 하지 않을' 뿐 (원래 밝기 그대로).
                 테두리·캡슐 이음매에 붙은 글자(TRESTAN, 9542)는 0.6 에서도 같이 빠짐."""
    L, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    Lf = L.astype(np.float32)
    D = np.maximum(cv2.GaussianBlur(Lf, (0, 0), 6) - cv2.GaussianBlur(Lf, (0, 0), 1), 0)
    R = np.max([cv2.filter2D(D, -1, k / k.sum()) for k in (_line_kernel(n, i * 22.5) for i in range(8))], axis=0)
    hi, lo = np.percentile(R, high), np.percentile(R, low)
    _, lab, st, _ = cv2.connectedComponentsWithStats((R > lo).astype(np.uint8), connectivity=8)
    strong = np.unique(lab[R > hi])
    keep = np.isin(lab, strong[strong > 0]) & (st[lab, cv2.CC_STAT_AREA] >= min_area)
    scale = strength / max(float(np.percentile(R * keep, 99)), 1e-3)      # 세기 기준은 테두리 빼기 전 지도로 (빼도 획 진하기는 같게)
    if rim_ext:
        ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) / max(L.shape)
        keep = keep & (ext[lab] < rim_ext)
    L2 = np.clip(Lf - R * keep * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


def channel_p(bgr, p):
    """설정 p 로 채널 사진 만들기: p['ch'] 채널 + (p 에 add_alpha > 0 이 있으면) 색 특징 더하기.
       예전 설정(add_alpha 없음)은 channel(bgr, p['ch']) 와 똑같음."""
    g = channel(bgr, p["ch"])
    if p.get("add_alpha", 0) > 0:
        g = add_color(bgr, g, p["add_alpha"], p["add_thr"], p.get("add_both", True))
    return g


# ---------------------------------------------------------------------
# 1단계 가우시안 블러
# ---------------------------------------------------------------------
def gauss(g, k, s):
    """가우시안 블러: 픽셀마다 주변 k×k 창의 값을 '가까울수록 크게' 섞어 평균 (흐림·잡음 제거).
       k (홀수) : 창 크기. 키우면 더 넓은 범위가 섞여 잡음이 줄지만 가는 획도 흐려짐.
       s (σ)    : 흐림 세기(멀리 있는 픽셀을 얼마나 섞을지). 키우면 더 흐림.
       둘은 함께 움직인다 — σ 가 커도 k 가 작으면 창이 작아 흐림에 한계가 있음 (대략 k ≈ 6σ 일 때 σ 가 다 반영됨).
       k = 0 이면 '안 함' (원본 그대로)."""
    if k == 0:
        return g
    return cv2.GaussianBlur(g, (k, k), s)


# ---------------------------------------------------------------------
# 2단계 평균 필터
# ---------------------------------------------------------------------
def mean_f(g, k):
    """평균 필터: 픽셀마다 주변 k×k 칸을 '똑같은 비중으로' 평균 (논문 색 갈래 2: 한 번 더 부드럽게).
       k : 창 크기. 키우면 더 많이 뭉개짐 — 같은 크기 가우시안보다 경계가 더 뭉개짐 (가까운 픽셀을 더 믿지 않으므로).
       k = 1 이면 '안 함' (자기 자신만 평균 = 원본). 이미 가우시안으로 흐린 뒤라 '안 함'이 나을 수도 있어 꼭 비교."""
    if k <= 1:
        return g
    return cv2.blur(g, (k, k))


# ---------------------------------------------------------------------
# 3단계 히스토그램 (대비 키우기)
# ---------------------------------------------------------------------
def hist(g, mode, clip=2.0, tile=8):
    """대비 정규화. OpenCV CLAHE 가 받는 파라미터는 clip·tile 두 개뿐.
       mode='none'  : 안 함
       mode='he'    : 전역 평활화 — 사진 전체의 밝기 분포를 고르게 펼침 (배경 잡티도 크게 커질 수 있음)
       mode='clahe' : 국소 평활화 — 사진을 tile×tile 칸으로 나눠 칸마다 평활화하고, 칸 사이는 부드럽게 이음
         clip : 칸마다 대비를 최대 몇 배까지 키울지 상한. 키우면 글자가 진해지지만 잡티도 커짐 (1 근처는 아주 약함)
         tile : 칸 수. 처리 크기 400 에서 tile 8 = 한 칸 50px. 키우면 더 국소적(그늘 보정↑, 칸 얼룩·잡음↑),
                줄이면 전역 HE 에 가까워짐. 한 칸이 글자보다 작아지면 글자 주변이 얼룩질 수 있음."""
    if mode == "none":
        return g
    if mode == "he":
        return cv2.equalizeHist(g)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def sharpen(g, amount, sigma=2.0, thr=0):
    """4단계 샤프닝 (언샤프 마스킹): 결과 = 원본 + amount × (원본 − 가우시안(원본, sigma)).
       '원본 − 흐린 원본' = 경계 성분. 이것을 더해 획 테두리를 진하게 함.
       amount : 세기. 키우면 획이 진해지지만 잡티도 커지고 경계 양옆에 밝고 어두운 띠(오버슈트)가 생김. 0 = 안 함
       sigma  : 반경 = 어떤 굵기의 경계를 강조할지. 키우면 굵은 획·큰 윤곽, 줄이면 가는 디테일(표면 질감·점도 같이)
       thr    : 문턱(밝기 0~255 단위). 경계 성분의 크기가 이보다 작은 곳은 강조하지 않음 → 평평한 면의 잡티가 덜 커짐.
                너무 크면 희미한 획도 빠짐. 0 = 모든 곳 강조
       계산은 실수(float32)로 해야 음수 경계 성분이 잘리지 않음. 끝에서 0~255 로 자름."""
    if amount <= 0:
        return g
    f = g.astype(np.float32)
    detail = f - cv2.GaussianBlur(f, (0, 0), sigma)           # (0, 0): 창 크기를 sigma 에 맞춰 자동으로
    if thr > 0:
        detail[np.abs(detail) < thr] = 0
    return np.clip(f + amount * detail, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# 5단계 점 제거 (획 보호 + 점 지우기) — 사용자 제안 방식, 마스크 없이 사진 전체에서
# ---------------------------------------------------------------------
SPOT_DEFAULT = dict(h=12, bh=13, min_len=18, dark=6, color=6, max_area=60, final_h=6, bright=False, protect=False)
# bright  : 획 보호에 '주변보다 밝은 선'(top-hat)도 추가 — 밝은 인쇄 글자(ANY, SPEED)는 black-hat 만으로는 보호를 못 받음 (5a 에서 확인)
# protect : 마무리 NLmeans 뒤 획 보호 구역을 NLmeans 전 픽셀로 되돌림 — 표면만 매끈하게, 글자는 그대로 (5b 에서 새로 시험)
# (5a 결과를 그대로 다시 만들 수 있도록 두 값의 기본은 False)


def _keep(binary, ok):
    """연결된 덩어리마다 ok(가로, 세로, 넓이) 가 참인 것만 남김 → bool 배열."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = ok(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA])
    return keep[lbl]


def _find_spots(L, a, b, dark, color, max_area):
    """점 후보: 주변(15×15 중앙값)보다 dark 이상 어둡고 '그리고' 색이 color 이상 튀는 작은 덩어리.
       색 튐 = √(Δa² + Δb²) (원래 제안의 '갈색' 조건을 일반화 → 파란 반점·검은 점도 잡힘).
       (색 조건만 쓰면 각인 테두리 색번짐이, 밝기 조건만 쓰면 각인 자체가 걸림)"""
    dark_dev = cv2.medianBlur(L, 15).astype(np.int16) - L.astype(np.int16)
    color_dev = np.hypot(a.astype(np.float32) - cv2.medianBlur(a, 15), b.astype(np.float32) - cv2.medianBlur(b, 15))
    cand = (dark_dev > dark) & (color_dev > color)
    return _keep(cand, lambda w, h, area: area <= max_area and max(w, h) <= 10)


def remove_spots(L, a, b, h, bh, min_len, dark, color, max_area, final_h, bright=False, protect=False):
    """점 제거. L = 밝기(흑백 400), a·b = 원본 색(400). → (밝기, a, b, 획 보호 마스크, 점 마스크)
       ① 획 보호: 강한 NLmeans(h)로 점을 뭉갠 임시 사진 → black-hat(bh, 주변보다 어두운 선 = 파인 각인) → Otsu →
          길이 min_len 이상 덩어리만 획 → 5 만큼 넓혀 획 테두리까지 보호
            h  : 키우면 점이 더 묻혀 획만 남지만, 너무 크면 희미한 획도 묻혀 보호를 못 받음
            bh : 획 굵기보다 커야 함. 키우면 넓은 그늘까지 획으로, 줄이면 굵은 획을 놓침
            min_len : 키우면 짧은 획(1, 점)이 보호를 못 받음, 줄이면 큰 점도 획으로 인정돼 안 지워짐
       ② 점: _find_spots(dark, color, max_area), 3 만큼 넓혀 옅은 테두리까지, 획 보호 구역은 제외
       ③ 메우기(inpaint): 밝기와 함께 a·b(색)도 메워 '색만 남은 유령 점'이 생기지 않게. 2회차는 기준을 조금 낮춤
       ④ 마무리 NLmeans(final_h, 0 이면 안 함): 남은 잔얼룩 정리. 키우면 깨끗하지만 획도 흐려짐
          (5a 에서 글씨를 흐리게 만든 주원인으로 확인 → protect=True 면 획 보호 구역은 NLmeans 전 픽셀로 되돌림)
       bright=True 면 ① 에서 top-hat(주변보다 밝은 선)도 Otsu 로 찾아 획에 더함 (밝은 글자 보호)"""
    tmp = cv2.fastNlMeansDenoising(L, None, h, 7, 21)                 # NLmeans: 비슷한 조각끼리 평균 → 흩어진 점은 묻히고 획은 남음
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh, bh))
    blackhat = cv2.morphologyEx(tmp, cv2.MORPH_BLACKHAT, se)          # 주변보다 어두운 선 (파인 각인, 어두운 인쇄)
    _, lines = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lines = lines > 0
    if bright:
        tophat = cv2.morphologyEx(tmp, cv2.MORPH_TOPHAT, se)          # 주변보다 밝은 선 (밝은 인쇄 글자)
        _, lb = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lines |= lb > 0
    stroke = _keep(lines, lambda w, hh, area: max(w, hh) >= min_len)
    stroke = cv2.dilate(stroke.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    spot = _find_spots(L, a, b, dark, color, max_area)
    spot = (cv2.dilate(spot.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~stroke   # ★ 획 보호 구역은 안 건드림
    m = spot.astype(np.uint8) * 255
    L2, a2, b2 = (cv2.inpaint(x, m, 3, cv2.INPAINT_TELEA) for x in (L, a, b))
    spot2 = _find_spots(L2, a, b, dark - 1, color / 2.0, max_area) & ~stroke   # 첫 번에 못 지운 옅은 점 (색은 원본 기준)
    if spot2.any():
        m2 = spot2.astype(np.uint8) * 255
        L2, a2, b2 = (cv2.inpaint(x, m2, 3, cv2.INPAINT_TELEA) for x in (L2, a2, b2))
    if final_h > 0:
        smooth = cv2.fastNlMeansDenoising(L2, None, final_h, 7, 21)
        if protect:                                                   # 획 보호 마무리: 표면만 매끈하게, 획은 NLmeans 전 그대로
            smooth[stroke] = L2[stroke]
        L2 = smooth
    return L2, a2, b2, stroke, spot | spot2


def overlay_masks(bgr, stroke, spot):
    """점 마스크 확인 그림: 초록 = 획 보호 구역, 빨강 = 지울 점 (반투명). 획이 빨갛게 칠해지면 글자를 지우는 위험한 설정."""
    out = bgr.astype(np.float32)
    out[stroke] = out[stroke] * 0.5 + np.array([0, 200, 0]) * 0.5
    out[spot] = out[spot] * 0.3 + np.array([0, 0, 255]) * 0.7
    return out.astype(np.uint8)


def clahe_rgb_each(bgr, clip=2.0, tile=8):
    """비교용 '나쁜 예': 컬러 사진의 B·G·R 를 각각 따로 CLAHE.
       채널마다 늘어나는 정도가 달라 색이 틀어지고 컬러 잡티·무지개 테두리가 생길 것으로 예상 →
       밝기(L)에만 처리하고 색을 다시 입히는 방식(recolor)이 왜 필요한지 보여 주는 용도."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return cv2.merge([c.apply(ch) for ch in cv2.split(bgr)])
