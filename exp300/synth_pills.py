# =====================================================================
#  synth_pills.py  —  학습용 '가짜 둥근 알약' 사진 만들기 (⑤ 각도 분류기, ⑥ TPS 펴기 학습용)
# =====================================================================
#
#  [왜 가짜 사진?]
#   규칙상 학습용 사진이 평가용(test)과 겹치면 안 되고, 개발용(dev)은 '고르기'에 써야 해서
#   AI Hub 사진은 학습에 쓰지 않아요. 대신 각인을 흉내 낸 사진을 무작위로 만들어서 학습해요.
#   → 정답(글자, 돌린 각도)을 우리가 정해서 만들기 때문에 라벨이 정확해요.
#
#  [만드는 법]  (실제 사진에서 본 특징을 흉내)
#   1) 둥근 알약 면: 기본 밝기 + 조명 기울기 + 얼룩(저주파 잡음) + 입자 잡음
#   2) 각인: Windows 글꼴로 글자 모양(마스크)을 그리고 → 흐려서 '홈 깊이'로 보고
#            → 한쪽에서 빛이 온다고 보고 홈의 기울기로 밝고 어두운 테두리를 만듦 (음각 75%, 양각 15%, 인쇄 10%)
#      배치: 한 줄 / 두 줄 / 가장자리를 따라 휜 글자(호)
#   3) 방해물: 가운데 분할선(15%)
#   4) 카메라 60° 흉내: 세로로 납작하게 + 아래쪽에 알약 옆면 띠
#   5) crop_pill 처럼 알약 바깥은 알약 중간 밝기로 칠함
#
#  [각도 규칙]  글자를 phi 도(반시계) 돌려서 그림 → 똑바로 세우려면 run_exp300.rotate(g, (360 - phi) % 360)
# =====================================================================

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
FONT_NAMES = ["arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "consola.ttf", "consolab.ttf",
              "tahoma.ttf", "tahomabd.ttf", "verdana.ttf", "verdanab.ttf", "times.ttf", "timesbd.ttf",
              "segoeui.ttf", "segoeuib.ttf", "impact.ttf", "cour.ttf", "courbd.ttf", "trebuc.ttf",
              "trebucbd.ttf", "georgia.ttf", "georgiab.ttf", "bahnschrift.ttf", "framd.ttf", "micross.ttf"]
FONTS = [os.path.join(FONT_DIR, f) for f in FONT_NAMES if os.path.exists(os.path.join(FONT_DIR, f))]
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_FONT_CACHE = {}


def font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def rand_text(rng, lo=1, hi=8):
    n = int(rng.choice(np.arange(lo, hi + 1), p=_len_p(lo, hi)))
    kind = rng.random()
    if kind < 0.35:                                   # 글자+숫자 섞기
        return "".join(rng.choice(list(CHARS), n))
    if kind < 0.65:                                   # 글자만
        return "".join(rng.choice(list(CHARS[:26]), n))
    if kind < 0.85:                                   # 숫자만
        return "".join(rng.choice(list(CHARS[26:]), n))
    k = int(rng.integers(1, n + 1))                   # 글자 뒤 숫자 (예: AB12)
    return "".join(rng.choice(list(CHARS[:26]), k)) + "".join(rng.choice(list(CHARS[26:]), n - k))


def _len_p(lo, hi):
    w = np.array([{1: .08, 2: .2, 3: .2, 4: .17, 5: .12, 6: .1, 7: .07, 8: .06}.get(i, .04) for i in range(lo, hi + 1)])
    return w / w.sum()


def text_mask(text, fpath, size):
    """글자 한 줄을 흰 글씨(1.0)로 그린 float 마스크 (여백 잘라냄)"""
    f = font(fpath, size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 8, b - t + 8), 0)
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, fill=255, font=f)
    return np.asarray(im, np.float32) / 255.0


def paste(canvas, m, cx, cy):
    """m 의 가운데를 (cx, cy)에 더해 넣기 (캔버스 밖은 잘라냄)"""
    h, w = m.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if X1 > X0 and Y1 > Y0:
        canvas[Y0:Y1, X0:X1] = np.maximum(canvas[Y0:Y1, X0:X1], m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0])


def rot_img(m, ang, center=None):
    h, w = m.shape
    M = cv2.getRotationMatrix2D(center or (w / 2, h / 2), ang, 1)
    return cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def band_points(top, bottom, F):
    """글자 띠의 위쪽 선·아래쪽 선 위에 왼→오른으로 F/2 개씩 점 (⑥ TPS 기준점 정답, TPSGrid 와 같은 순서)"""
    return np.concatenate([top(np.linspace(0, 1, F // 2)), bottom(np.linspace(0, 1, F // 2))])


def layout_mask(rng, S, R, layout, text=None, F=20):
    """각인 마스크 (S×S, 똑바로 선 상태), 읽을 글자, 글자 띠 기준점 (F,2) 를 돌려줌 (두 줄은 기준점 없음)"""
    D = np.zeros((S, S), np.float32)
    fp = FONTS[int(rng.integers(len(FONTS)))]
    if layout == "arc":
        text = text or rand_text(rng, 3, 9)
        rr = rng.uniform(0.58, 0.8) * R                       # 글자가 놓일 반지름
        size = int(np.clip(rng.uniform(0.2, 0.32) * R, 9, 60))
        ms = [text_mask(ch, fp, size) for ch in text]
        gap = rng.uniform(0.05, 0.25) * size
        widths = [m.shape[1] - 6 + gap for m in ms]
        span = sum(widths) / rr                                # 라디안
        if span > np.radians(300):                             # 한 바퀴를 넘으면 글자를 줄임
            return layout_mask(rng, S, R, "arc", text[: max(3, len(text) // 2)], F)
        a = -np.pi / 2 - span / 2                              # 위쪽 가운데를 기준으로 왼→오
        a0, a1, gh = a, a + span, max(m.shape[0] for m in ms) - 8
        arc = lambda r: (lambda t: np.stack([S / 2 + r * np.cos(a0 + (a1 - a0) * t),
                                             S / 2 + r * np.sin(a0 + (a1 - a0) * t)], 1))
        fid = band_points(arc(rr + gh * 0.6), arc(rr - gh * 0.6), F)   # 글자 윗부분이 바깥(큰 반지름)
        for m, wd in zip(ms, widths):
            a_mid = a + wd / 2 / rr
            cx, cy = S / 2 + rr * np.cos(a_mid), S / 2 + rr * np.sin(a_mid)
            big = np.zeros((max(m.shape) * 2,) * 2, np.float32)
            paste(big, m, big.shape[1] / 2, big.shape[0] / 2)
            big = rot_img(big, -np.degrees(a_mid + np.pi / 2))  # 글자 윗부분이 바깥을 향하게
            paste(D, big, cx, cy)
            a += wd / rr
        return D, text, fid
    lines = [text or rand_text(rng)] if layout == "line" else [rand_text(rng, 1, 5), rand_text(rng, 1, 5)]
    maxw = rng.uniform(0.6, 0.85) * 2 * R
    h_target = rng.uniform(0.18, 0.34) * 2 * R / len(lines)
    ms = []
    for t in lines:
        m = text_mask(t, fp, 64)
        s = min(h_target / m.shape[0], maxw / m.shape[1])
        m = cv2.resize(m, None, fx=s * rng.uniform(0.85, 1.15), fy=s, interpolation=cv2.INTER_AREA)
        ms.append(m)
    gap = rng.uniform(0.1, 0.4) * h_target
    tot = sum(m.shape[0] for m in ms) + gap * (len(ms) - 1)
    oy = rng.uniform(-0.12, 0.12) * R
    y = S / 2 - tot / 2 + oy
    xs = []
    for m in ms:
        cx = S / 2 + rng.uniform(-0.06, 0.06) * R
        paste(D, m, cx, y + m.shape[0] / 2)
        xs += [cx - m.shape[1] / 2, cx + m.shape[1] / 2]
        y += m.shape[0] + gap
    fid = None
    if len(ms) == 1:
        x0, x1 = min(xs) + 2, max(xs) - 2
        y0, y1 = S / 2 - tot / 2 + oy + 2, S / 2 + tot / 2 + oy - 2
        line = lambda yy: (lambda t: np.stack([x0 + (x1 - x0) * t, np.full_like(t, yy)], 1))
        fid = band_points(line(y0), line(y1), F)
    return D, "".join(lines), fid


def make_pill(rng, S=192, phi=0.0, layout=None, text=None, squash=None, return_fid=False):
    """가짜 둥근 알약 1장 → (흑백 uint8 S×S, 읽을 글자, 배치 이름)
       return_fid=True 면 글자 띠 기준점(⑥ TPS 정답, -1~1 좌표, (20,2)) 도 같이 (두 줄이면 None)"""
    R = S / 2 / 1.1
    layout = layout or rng.choice(["line", "line", "line", "two", "arc"], p=None)
    D, text, fid = layout_mask(rng, S, R, layout, text)
    if rng.random() < 0.15:                                    # 분할선 (글자는 아님)
        line = np.zeros_like(D)
        y = S / 2 + (rng.uniform(0.25, 0.4) * R * rng.choice([-1, 1]) if D[int(S / 2)].any() else 0)
        cv2.line(line, (0, int(y)), (S, int(y)), 1.0, max(1, int(R * 0.03)))
        D = np.maximum(D, rot_img(line, rng.choice([0, 90])))
    D = rot_img(D, phi)                                        # 글자 방향 (정답 각도)
    if fid is not None:                                        # 기준점도 똑같이 돌림
        fid = cv2.transform(fid[None].astype(np.float32), cv2.getRotationMatrix2D((S / 2, S / 2), phi, 1))[0]

    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    r = np.hypot(xx - S / 2, yy - S / 2)
    base = rng.uniform(120, 235)
    la = rng.uniform(0, 2 * np.pi)                             # 빛이 오는 방향
    lx, ly = np.cos(la), np.sin(la)
    surf = base + rng.uniform(0, 25) * ((xx - S / 2) * lx + (yy - S / 2) * ly) / R    # 조명 기울기
    low = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), S * 0.08)
    surf += low / (low.std() + 1e-6) * rng.uniform(0, 6)

    kind = rng.random()
    sg = rng.uniform(0.8, 2.2) * S / 192
    depth = cv2.GaussianBlur(D, (0, 0), sg)
    if kind < 0.9:                                             # 음각(0.75) / 양각(0.15): 홈의 기울기로 그늘
        sign = 1.0 if kind < 0.75 else -1.0
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        k = rng.uniform(25, 90)
        surf += sign * k * (gx * lx + gy * ly) - rng.uniform(0, 25) * depth * (sign > 0)
    else:                                                      # 인쇄(잉크)
        surf -= rng.uniform(40, 110) * np.clip(depth * 1.5, 0, 1)
    bevel = np.clip((r - 0.86 * R) / (0.14 * R), 0, 1)          # 가장자리 경사
    surf -= rng.uniform(0, 30) * bevel ** 2
    surf += rng.normal(0, rng.uniform(1, 5), (S, S))            # 입자 잡음

    sq = squash if squash is not None else (rng.uniform(0.78, 1.0) if rng.random() < 0.4 else 1.0)
    img = np.clip(surf, 0, 255).astype(np.float32)
    inner = (r <= 0.97 * R).astype(np.float32)
    if sq < 1.0:                                               # 카메라 60°: 세로로 납작 + 아래쪽 옆면 띠
        M = np.float32([[1, 0, 0], [0, sq, S / 2 * (1 - sq)]])
        img = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
        if fid is not None:
            fid = cv2.transform(fid[None].astype(np.float32), M)[0]
        inner = cv2.warpAffine(inner, M, (S, S))
        band = cv2.warpAffine((r <= 0.97 * R).astype(np.float32), np.float32([[1, 0, 0], [0, sq, S / 2 * (1 - sq) + R * (1 - sq) * 1.6]]), (S, S))
        side = (band > 0.5) & (inner < 0.5)
        img[side] = base * rng.uniform(0.55, 0.9) + rng.normal(0, 3, side.sum())
        inner = np.maximum(inner, side.astype(np.float32))
    m = inner > 0.5
    img[~m] = np.median(img[m])                                # 바깥은 알약 중간 밝기 (crop_pill 과 같게)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    out = np.clip(img, 0, 255).astype(np.uint8)
    if return_fid:
        fid = None if fid is None else (2 * fid + 1) / S - 1   # grid_sample(align_corners=False) 좌표
        return out, text, layout, fid
    return out, text, layout


if __name__ == "__main__":                                     # 견본 몇 장 저장해서 눈으로 확인
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_samples.png"
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(24):
        g, t, lay = make_pill(rng, 192, phi=float(rng.uniform(0, 360)))
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(g, f"{t} {lay}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(g)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, 24, 8)]
    cv2.imencode(".png", np.vstack(rows))[1].tofile(out)
    print("저장:", out, "글꼴", len(FONTS), "개")
