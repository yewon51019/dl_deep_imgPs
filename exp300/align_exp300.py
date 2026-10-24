# =====================================================================
#  align_exp300.py  —  각인 정렬 실험 2차: ①(PaddleOCR 상자) · ③ 단독 · ④ 극좌표 · ⑤ 각도 분류기 · ⑥ TPS 펴기
# =====================================================================
#
#  run_exp300.py --align <이름> 으로 불러 써요. (vote / conf / box 는 run_exp300.py 안에 있음)
#    pbox       : ① PaddleOCR 검출 상자의 기울기로 글줄 방향 → θ/θ+180 두 번 읽고 신뢰도로 고름 (상자 없으면 ③)
#    pbox_cls   : 위와 같고, 위아래(θ / θ+180)는 PaddleOCR 글줄 방향 분류기(0°/180°)로 고름
#    stroke     : ③ 획 방향만 (상자 안 씀)
#    polar      : ④ 알약 중심 기준으로 가장자리 띠를 펼쳐(cv2.warpPolar) 직선으로 만든 뒤 읽기
#    polar_conf : ④ 펼친 띠 2장 + 12방향 읽기를 모두 후보로 놓고 신뢰도 최고 (휜 글자 / 곧은 글자 둘 다 대응)
#    cnn        : ⑤ 작은 CNN(ResNet18)이 글자를 세울 각도를 예측 → 상위 2개 각도만 읽고 신뢰도로 고름
#    tps        : ⑥ TPS 펴기 네트워크로 휜·비스듬한 글자를 펴서 PARSeq 로 읽음. 12방향 × 펴기 → 신뢰도 최고
#
#  · 길쭉한 알약은 모든 방법이 지금처럼 0°/180° (차이는 둥근 알약에서만 남)
#  · ⑤⑥ 은 synth_pills.py 의 '가짜 알약'으로만 학습 (train_align_exp300.py). AI Hub 사진은 학습에 안 씀
#  · PaddleOCR 은 CPU 판. Windows 에서 Paddle 을 먼저 불러오면 PyTorch 가 깨져서 torch 를 먼저 불러옴
# =====================================================================

import os

import torch                                         # ← Paddle 보다 먼저 (Windows DLL 충돌 방지)

import cv2
import numpy as np

import run_exp300 as base

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models")


def _long(g, ocr):
    """길쭉한 알약: 지금 방식과 같이 0°/180° 를 읽고 신뢰도로 고름"""
    cands = base.read_angles(g, [0, 180], ocr)
    best = base.pick_by_conf(cands)
    return best[0], best[1], cands, best[2], {"src": "long", "theta": "", "n_box": ""}


def rotate_same(g, ang):
    """크기를 그대로 두고 가운데를 기준으로 돌림 (둥근 알약은 모서리가 배경이라 잘려도 괜찮음)"""
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rotate_matrix(g, ang):
    """run_exp300.rotate 와 같은 변환 행렬 (상자 좌표를 같이 옮길 때)"""
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return M


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ① PaddleOCR 상자 + 글줄 방향 분류기                                  ║
# ╚════════════════════════════════════════════════════════════════════╝
_PD = {}
PADDLE_MAX_AREA = 0.5      # 알약 사진 넓이의 절반이 넘는 상자는 '알약 전체'를 잡은 것 → 글자 상자로 안 씀


def _paddle(name):
    if name not in _PD:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import TextDetection, TextLineOrientationClassification
        cls = {"det": TextDetection, "cls": TextLineOrientationClassification}[name]
        for kw in ({"enable_mkldnn": False}, {}):
            try:
                _PD[name] = cls(**kw)
                break
            except (TypeError, ValueError):
                continue
    return _PD[name]


def paddle_rotated_boxes(g):
    """PaddleOCR 검출(기본 설정)의 4점 상자. 너무 작거나 알약 전체를 잡은 상자는 뺌"""
    res = list(_paddle("det").predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))[0]
    h, w = g.shape
    polys = [np.asarray(p, np.float32).reshape(-1, 2) for p in res["dt_polys"]]
    return [p for p in polys if 40 <= cv2.contourArea(p) <= PADDLE_MAX_AREA * h * w]


def paddle_flip_prob(g, ang, polys):
    """rotate(g, ang) 로 세운 사진에서 글자 부분을 잘라 PaddleOCR 글줄 방향 분류기에 넣음 → '180°' 확률"""
    M = rotate_matrix(g, ang)
    r = base.rotate(g, ang)
    H, W = r.shape
    if polys:
        pts = np.concatenate([cv2.transform(p[None], M)[0] for p in polys])
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
    else:                                               # 상자가 없으면(③) 가운데 띠
        x0, x1, y0, y1 = int(W * 0.2), int(W * 0.8), int(H * 0.3), int(H * 0.7)
    crop = r[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if crop.size < 16:
        return 0.5
    res = list(_paddle("cls").predict(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)))[0]
    names, scores = list(res["label_names"]), [float(s) for s in res["scores"]]
    p = scores[0] if names[0].startswith("180") else 1 - scores[0]
    return float(p)


def read_pbox(g, aspect, ocr, use_cls=False):
    """① PaddleOCR 상자 + ③ 대신. use_cls=True 면 θ/θ+180 중 하나를 분류기로 고름 (애매해서 4방향이면 축은 신뢰도로)"""
    if aspect > 1.3:
        return _long(g, ocr)
    t, c, cands, ang, info = base.read_aligned(g, aspect, ocr, boxes_fn=paddle_rotated_boxes)
    info["conf_angle"] = ang
    if use_cls:
        # 후보 순서: 2개면 (θ, θ+180), 4개면 (θ, θ+90, θ+180, θ+270) → 축마다 (정방향, 뒤집힘) 짝
        pairs = [(cands[0], cands[1])] if len(cands) == 2 else [(cands[0], cands[2]), (cands[1], cands[3])]
        c0, c1 = max(pairs, key=lambda pr: max(pr[0][1], pr[1][1]))
        p180 = paddle_flip_prob(g, c0[2], info["polys"])
        info["p180"] = round(p180, 3)
        t, c, ang = c1 if p180 > 0.5 else c0
    return t, c, cands, ang, info


def read_stroke(g, aspect, ocr):
    """③ 획 방향만"""
    if aspect > 1.3:
        return _long(g, ocr)
    return base.read_aligned(g, aspect, ocr, boxes_fn=lambda _g: [])


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ④ 극좌표 펼치기                                                      ║
# ╚════════════════════════════════════════════════════════════════════╝
POLAR_BAND = (0.40, 0.98)   # 펼칠 띠: 알약 반지름의 40%~98% (가장자리를 따라 새긴 글자가 있는 곳)


def polar_strip(g):
    """알약 중심 기준으로 가장자리 띠를 펼친 직사각형 사진 → (띠, 이음매 각도).
       글자 윗부분이 바깥(가장자리)을 향한다고 보고 바깥 반지름을 위로 놓음.
       띠를 어디서 자를지(이음매)는 획이 가장 적은 곳(글자 사이 빈 곳)으로 정함"""
    s = max(g.shape)
    g = cv2.resize(g, (s, s), interpolation=cv2.INTER_CUBIC)   # 카메라 60° 로 납작한 알약을 원에 가깝게
    R = s / 2 / 1.1                                         # crop_pill 이 10% 여유를 두고 자름
    n_ang = int(round(2 * np.pi * R * 0.75))                 # 띠 길이 = 반지름 75% 지점의 둘레
    P = cv2.warpPolar(g, (int(R), n_ang), (s / 2, s / 2), R, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    r0, r1 = int(R * POLAR_BAND[0]), int(R * POLAR_BAND[1])
    S = P[:, r0:r1].T[::-1]                                 # 행 = 반지름(위가 바깥), 열 = 각도
    k = max(9, int(S.shape[0] * 0.15)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.add(cv2.morphologyEx(S, cv2.MORPH_BLACKHAT, se), cv2.morphologyEx(S, cv2.MORPH_TOPHAT, se))
    hs = S.shape[0]
    col = resp[int(hs * 0.15):int(hs * 0.9)].astype(np.float32).sum(0)  # 가장자리 그늘 줄은 빼고
    win = hs | 1                                            # 글자 2~3개 폭으로 평균 → 가장 넓은 빈 곳
    col = np.convolve(np.tile(col, 3), np.ones(win) / win, "same")[len(col):2 * len(col)]   # 둥글게 이어서 평균
    seam = int(np.argmin(col))
    S = np.roll(S, -seam, axis=1)
    return np.ascontiguousarray(S), seam * 360.0 / n_ang


def read_polar(g, aspect, ocr, with_rot=False, step=30):
    """④ 펼친 띠를 그대로 / 180° 돌려 두 번 읽고 신뢰도로 고름. with_rot=True 면 12방향 읽기도 후보에 넣음(④+②)"""
    if aspect > 1.3:
        return _long(g, ocr)
    S, seam = polar_strip(g)
    cands = []
    for flip in (0, 1):
        img = S if not flip else S[::-1, ::-1].copy()
        try:
            text, conf = ocr(img)
        except Exception as e:
            text, conf = "", 0.0
            print(f"    [경고] 극좌표 읽기 실패: {e}")
        cands.append((base.norm(text), float(conf), f"P{int(seam)}" + ("r" if flip else "")))
    info = {"src": "polar", "theta": round(seam, 1), "n_box": "", "strip": S}
    if with_rot:
        cands += base.read_angles(g, list(range(0, 360, step)), ocr)
        info["src"] = "polar+rot"
    best = base.pick_by_conf(cands)
    return best[0], best[1], cands, best[2], info


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑤ 각도 분류기 (ResNet18, 36칸 = 10°씩)                               ║
# ╚════════════════════════════════════════════════════════════════════╝
CNN_SIZE, CNN_BINS = 128, 36


def build_angle_cnn(pretrained=True):
    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = torch.nn.Linear(m.fc.in_features, CNN_BINS)
    return m


def cnn_input(gs, dev):
    """흑백 사진 목록 → (N,3,128,128) 텐서"""
    x = np.stack([cv2.resize(g, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_AREA) for g in gs]).astype(np.float32)
    x = torch.from_numpy((x / 255.0 - 0.45) / 0.25)[:, None].repeat(1, 3, 1, 1)
    return x.to(dev)


_CNN = {}


def angle_cnn():
    if "m" not in _CNN:
        dev = "cuda" if base._gpu() else "cpu"
        m = build_angle_cnn(pretrained=False)
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, "angle_cnn.pt"), map_location=dev))
        _CNN["m"], _CNN["dev"] = m.eval().to(dev), dev
    return _CNN["m"], _CNN["dev"]


def cnn_angles(g, k=2):
    """글자를 세울 각도 상위 k 개 (서로 40° 이상 떨어진 봉우리). 각도는 rotate() 기준"""
    m, dev = angle_cnn()
    with torch.no_grad():
        p = m(cnn_input([g], dev)).softmax(-1)[0].cpu().numpy()
    p = (np.roll(p, 1) + 2 * p + np.roll(p, -1)) / 4                  # 이웃 칸과 살짝 섞기
    out = []
    for i in np.argsort(-p):
        a = float(i * 360 / CNN_BINS)
        if all(min(abs(a - b), 360 - abs(a - b)) >= 40 for b, _ in out):
            out.append((a, float(p[i])))
        if len(out) == k:
            break
    return out


def read_cnn(g, aspect, ocr):
    if aspect > 1.3:
        return _long(g, ocr)
    tops = cnn_angles(g)
    cands = base.read_angles(g, [a for a, _ in tops], ocr)
    best = base.pick_by_conf(cands)
    return best[0], best[1], cands, best[2], {"src": "cnn", "theta": tops[0][0], "n_box": "",
                                              "cnn_p": [round(p, 3) for _, p in tops]}


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑥ TPS 펴기 (RARE 방식의 STN + 얼려 둔 PARSeq)                         ║
# ╚════════════════════════════════════════════════════════════════════╝
TPS_IN, TPS_F, TPS_OUT = 160, 20, (32, 128)     # 입력 160×160, 기준점 20개(위 10 + 아래 10), 출력 32×128 (PARSeq 입력 크기)


class TPSGrid(torch.nn.Module):
    """기준점(fiducial) F개의 위치 → 출력 그림의 각 픽셀이 입력의 어디서 올지(grid). (Shi et al., RARE 2016 / clovaai 구현과 같은 식)"""
    def __init__(self, F=TPS_F, out_hw=TPS_OUT):
        super().__init__()
        H, W = out_hw
        cx = np.linspace(-1.0, 1.0, F // 2)
        C = np.concatenate([np.stack([cx, -np.ones(F // 2)], 1), np.stack([cx, np.ones(F // 2)], 1)])   # (F,2)
        hat = np.linalg.norm(C[:, None] - C[None], axis=2)
        np.fill_diagonal(hat, 1)
        hat = hat ** 2 * np.log(hat)
        delta = np.concatenate([np.concatenate([np.ones((F, 1)), C, hat], 1),
                                np.concatenate([np.zeros((2, 3)), C.T], 1),
                                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], 1)])
        gx = (np.arange(-W, W, 2) + 1.0) / W
        gy = (np.arange(-H, H, 2) + 1.0) / H
        P = np.stack(np.meshgrid(gx, gy), 2).reshape(-1, 2)                                          # (HW,2)
        d = np.linalg.norm(P[:, None] - C[None], axis=2)
        rbf = d ** 2 * np.log(d + 1e-6)
        P_hat = np.concatenate([np.ones((len(P), 1)), P, rbf], 1)
        self.register_buffer("inv_delta", torch.tensor(np.linalg.inv(delta), dtype=torch.float32))
        self.register_buffer("P_hat", torch.tensor(P_hat, dtype=torch.float32))
        self.H, self.W = H, W

    def forward(self, Cp):                                          # Cp: (B,F,2) 입력에서의 기준점 위치 (-1~1)
        T = self.inv_delta @ torch.cat([Cp, Cp.new_zeros(Cp.shape[0], 3, 2)], 1)
        return (self.P_hat @ T).view(-1, self.H, self.W, 2)


class TPSRectifier(torch.nn.Module):
    """작은 CNN(위치 찾기망)이 기준점 20개를 내고 → TPS 로 알약 사진에서 글자 띠를 펴서 32×128 로 꺼냄"""
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return torch.nn.Sequential(torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.BatchNorm2d(o),
                                       torch.nn.ReLU(True), torch.nn.MaxPool2d(2))
        self.loc = torch.nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128), blk(128, 256),
                                       torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                       torch.nn.Linear(256, 256), torch.nn.ReLU(True))
        self.fc = torch.nn.Linear(256, TPS_F * 2)
        cx = np.linspace(-0.75, 0.75, TPS_F // 2)                   # 처음엔 가운데 가로 띠를 그대로 꺼내도록
        init = np.concatenate([np.stack([cx, -0.3 * np.ones(TPS_F // 2)], 1), np.stack([cx, 0.3 * np.ones(TPS_F // 2)], 1)])
        torch.nn.init.zeros_(self.fc.weight)
        self.fc.bias.data = torch.tensor(init.reshape(-1), dtype=torch.float32)
        self.grid = TPSGrid()

    def forward(self, x):                                           # x: (B,1,160,160) 0~1
        small = torch.nn.functional.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        Cp = self.fc(self.loc(small)).view(-1, TPS_F, 2)
        g = self.grid(Cp)
        out = torch.nn.functional.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
        return out, Cp


def to_parseq(rect):
    """펴진 흑백(0~1) → PARSeq 입력 (3채널, -1~1)"""
    return ((rect - 0.5) / 0.5).repeat(1, 3, 1, 1)


_TPS = {}


def tps_model(parseq):
    if "m" not in _TPS:
        m = TPSRectifier()
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tps_stn.pt"), map_location=parseq.dev))
        _TPS["m"] = m.eval().to(parseq.dev)
    return _TPS["m"]


def tps_read_batch(gs, parseq):
    """흑백 사진 목록 → [(글자, 신뢰도 0~100, 펴진 그림 uint8)]"""
    m = tps_model(parseq)
    x = np.stack([cv2.resize(g, (TPS_IN, TPS_IN), interpolation=cv2.INTER_AREA) for g in gs]).astype(np.float32) / 255
    x = torch.from_numpy(x)[:, None].to(parseq.dev)
    with torch.no_grad():
        rect, _ = m(x)
        p = parseq.m(to_parseq(rect)).softmax(-1)
    labels, confs = parseq.m.tokenizer.decode(p)
    out = []
    for t, c, r in zip(labels, confs, rect):
        out.append(("".join(ch for ch in t if ch.isalnum()), float(c.mean()) * 100 if len(c) else 0.0,
                    (r[0].cpu().numpy() * 255).astype(np.uint8)))
    return out


def read_tps(g, aspect, ocr, step=30):
    """⑥ 12방향으로 돌린 알약을 각각 펴서 읽고 신뢰도 최고. ocr 은 PARSeq(또는 PARSeq 를 .m 으로 감싼 것)"""
    if aspect > 1.3:
        return _long(g, ocr)
    parseq = ocr if isinstance(ocr, base.PARSeq) else ocr.m
    angles = list(range(0, 360, step))
    res = tps_read_batch([rotate_same(g, a) for a in angles], parseq)
    cands = [(base.norm(t), c, a) for (t, c, _), a in zip(res, angles)]
    best = base.pick_by_conf(cands)
    rect = res[angles.index(best[2])][2]
    return best[0], best[1], cands, best[2], {"src": "tps", "theta": "", "n_box": "", "rect": rect}


# ----------------------------- run_exp300 에서 부르는 곳 -----------------------------
def read_pill_ext(g, aspect, ocr, step, align):
    if align == "pbox":
        return read_pbox(g, aspect, ocr)
    if align == "pbox_cls":
        return read_pbox(g, aspect, ocr, use_cls=True)
    if align == "stroke":
        return read_stroke(g, aspect, ocr)
    if align == "polar":
        return read_polar(g, aspect, ocr)
    if align == "polar_conf":
        return read_polar(g, aspect, ocr, with_rot=True, step=step)
    if align == "cnn":
        return read_cnn(g, aspect, ocr)
    if align == "tps":
        return read_tps(g, aspect, ocr, step)
    raise ValueError(f"모르는 --align: {align}")
