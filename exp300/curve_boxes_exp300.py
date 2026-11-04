# =====================================================================
#  curve_boxes_exp300.py  —  휜 각인(예: 아리셉트정)에서 놓친 글자 박스 되찾기
# =====================================================================
#
#  [왜]  CRAFT(EasyOCR 글자 찾기)는 글자를 '단어 박스'로 묶어서 돌려줘요.
#        각인이 알약 테두리를 따라 휘어 있으면 글자 사이 연결 점수(affinity)가 끊겨서
#        단어가 여러 조각으로 나뉘거나, 흐린 글자는 아예 박스가 안 생겨요.
#        또 지금 코드(run_exp300.craft_boxes)는 기울어진 박스를 '감싸는 사각형'으로 바꿔서
#        기울기 정보도 버려요.
#
#  [아이디어]  (루카스-카나데처럼 '찾은 곳 주변을 보고 다음 위치를 따라가기')
#    ① CRAFT 의 '글자 점수 지도(region score)'를 직접 꺼내서 단어가 아니라 '글자 하나하나'의
#       중심·크기·점수를 얻는다.
#    ② 점수가 높은 글자(확실한 글자)끼리 이웃을 묶어 '글자 줄(chain)'을 만든다.
#    ③ 줄의 글자 중심들에 원(또는 직선)을 맞춘다 → 글자마다 접선 방향 = 그 글자의 기울기.
#       괄호 ( 처럼 휜 줄이면 위 글자와 아래 글자의 기울기가 반대로 나온다.
#    ④ 줄의 양 끝에서 원호를 따라 '다음 글자가 있을 자리'를 예측하고, 그 근처에
#       약한 글자 점수가 있으면 글자로 받아들여 줄을 늘린다 (반복).
#    ⑤ 너무 겹치는 글자 박스는 IoU 로 하나만 남긴다.
#    ⑥ 글자마다 접선 방향으로 똑바로 세워 잘라서 가로로 이어 붙인 '펴진 줄'을 만들고
#       PARSeq 로 한 번에 읽는다 (0°/180° 둘 다 읽고 확신도 높은 쪽).
#
#  [실행 예]  (개발용 둥근 알약만, 0° 한 번 읽기 기준과 비교)
#    python exp300\curve_boxes_exp300.py --manifest exp300\results\manifest.csv --split dev ^
#        --out exp300\results\curve_boxes_dev.csv --viz exp300\results\viz_curve_dev
#    python exp300\curve_boxes_exp300.py --selftest        # 합성 곡선 글자로 동작만 확인 (데이터 불필요)
#
#  ※ 평가용(test)은 방법을 개발용으로 다 정한 뒤 마지막에 한 번만 돌리세요.
#  ※ --viz 그림에는 원본 사진이 들어가므로 GitHub 에 올리지 않습니다 (results/viz_* 는 .gitignore).
# =====================================================================

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp300 as base                                     # noqa: E402


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ① CRAFT 점수 지도 꺼내기                                            ║
# ╚════════════════════════════════════════════════════════════════════╝

def craft_maps(g, canvas=2560, mag=1.0):
    """EasyOCR 안의 CRAFT 를 직접 돌려 (글자 점수 지도, 글자 사이 연결 점수 지도)를 원본 크기로 돌려줌.
       easyocr.Reader.detect 는 이 지도로 단어 박스만 만들고 지도 자체는 버려요."""
    import torch
    from easyocr import imgproc
    r = base.easy_reader()
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) if g.ndim == 2 else g
    img, ratio, _ = imgproc.resize_aspect_ratio(rgb, canvas, interpolation=cv2.INTER_LINEAR, mag_ratio=mag)
    x = torch.from_numpy(imgproc.normalizeMeanVariance(img)).permute(2, 0, 1).unsqueeze(0).to(r.device)
    with torch.no_grad():
        y, _ = r.detector(x)
    y = y[0].cpu().numpy()                                     # (h/2, w/2, 2)
    H, W = g.shape[:2]
    s = 2.0 / ratio                                            # 지도 1칸 = 원본 s 픽셀
    up = cv2.resize(y, (int(round(y.shape[1] * s)), int(round(y.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
    up = up[:H, :W]
    if up.shape[0] < H or up.shape[1] < W:                     # 반올림으로 모자라면 0으로 채움
        up = np.pad(up, ((0, H - up.shape[0]), (0, W - up.shape[1]), (0, 0)))
    return up[..., 0], up[..., 1]


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ② 글자 하나하나 찾기                                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def find_chars(region, min_peak=0.2, rel=0.5, min_area=6):
    """글자 점수 지도에서 '산봉우리' 하나 = 글자 하나.
       봉우리끼리 붙어 있어도 나눌 수 있게, 먼저 봉우리(지역 최댓값)를 찾고
       봉우리 높이의 rel 배 이상인 픽셀을 가장 가까운 봉우리에 나눠 줌.
       돌려주는 것: [{cx, cy, w, h, ang, peak, box}]  (box = 바깥 사각형 x0,y0,x1,y1)"""
    sm = cv2.GaussianBlur(region, (0, 0), 1.0)
    k = max(5, int(min(region.shape) * 0.02) | 1)
    is_max = (sm >= cv2.dilate(sm, np.ones((k, k), np.uint8)) - 1e-6) & (sm >= min_peak)
    ys, xs = np.nonzero(is_max)
    order = np.argsort(-sm[ys, xs])
    peaks = []
    for i in order:                                            # 너무 가까운 봉우리는 하나로
        if all((xs[i] - px) ** 2 + (ys[i] - py) ** 2 > (k * 0.8) ** 2 for px, py in peaks):
            peaks.append((int(xs[i]), int(ys[i])))
    if not peaks:
        return []
    P = np.array(peaks, float)
    mask = sm >= min_peak * rel
    my, mx = np.nonzero(mask)
    d2 = (mx[:, None] - P[None, :, 0]) ** 2 + (my[:, None] - P[None, :, 1]) ** 2
    owner = d2.argmin(1)
    out = []
    for j, (px, py) in enumerate(peaks):
        pk = float(sm[py, px])
        sel = (owner == j) & (sm[my, mx] >= pk * rel)
        xs_j, ys_j = mx[sel], my[sel]
        if len(xs_j) < min_area:
            continue
        w8 = sm[ys_j, xs_j]
        cx, cy = float((xs_j * w8).sum() / w8.sum()), float((ys_j * w8).sum() / w8.sum())
        (_, _), (rw, rh), ang = cv2.minAreaRect(np.stack([xs_j, ys_j], 1).astype(np.float32))
        out.append(dict(cx=cx, cy=cy, w=float(rw), h=float(rh), ang=float(ang), peak=pk,
                        box=(int(xs_j.min()), int(ys_j.min()), int(xs_j.max()) + 1, int(ys_j.max()) + 1)))
    return out


def peak_in_window(region, x, y, rad, used_mask=None):
    """(x,y) 둘레 rad 픽셀 안에서 글자 점수가 가장 높은 자리와 점수"""
    H, W = region.shape
    x0, x1 = max(0, int(x - rad)), min(W, int(x + rad) + 1)
    y0, y1 = max(0, int(y - rad)), min(H, int(y + rad) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    win = region[y0:y1, x0:x1].copy()
    if used_mask is not None:
        win[used_mask[y0:y1, x0:x1]] = 0
    k = int(np.argmax(win))
    py, px = divmod(k, win.shape[1])
    return float(x0 + px), float(y0 + py), float(win[py, px])


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ③ 원(또는 직선) 맞추기 → 글자마다 기울기                            ║
# ╚════════════════════════════════════════════════════════════════════╝

def fit_circle(pts):
    """최소제곱 원 맞추기 (Kasa). 돌려줌: (중심x, 중심y, 반지름) 또는 None"""
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    try:
        (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = a1 / 2, a2 / 2
    r2 = a3 + cx * cx + cy * cy
    return (float(cx), float(cy), float(math.sqrt(r2))) if r2 > 0 else None


class Curve:
    """글자 줄의 모양. 원이면 접선 = 반지름에 수직, 직선이면 접선 = 주성분 방향.
       straight_ratio: 반지름이 글자 줄 길이의 이 배수보다 크면 사실상 직선으로 봄"""
    def __init__(self, pts, straight_ratio=4.0):
        self.pts = pts
        mean = pts.mean(0)
        if len(pts) >= 2:
            _, _, vt = np.linalg.svd(pts - mean)
            d = vt[0]
        else:
            d = np.array([1.0, 0.0])
        self.mean, self.dir = mean, d / np.linalg.norm(d)
        span = float(np.ptp((pts - mean) @ self.dir)) if len(pts) >= 2 else 1.0
        c = fit_circle(pts)
        resid = None
        if c is not None:
            resid = float(np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2]).mean())
        self.circle = c if (c is not None and c[2] < straight_ratio * max(span, 1.0) and resid < 0.25 * max(span, 1.0)) else None
        self.span = span

    def tangent(self, x, y):
        """(x,y) 에서 글자 줄 방향 단위벡터 (줄 순서 방향과 같게 맞춤)"""
        if self.circle is None:
            return self.dir.copy()
        cx, cy, _ = self.circle
        t = np.array([-(y - cy), x - cx])
        t /= max(np.linalg.norm(t), 1e-6)
        return t if t @ self.dir >= 0 else -t

    def order(self, pts):
        """글자를 줄을 따라 순서대로 놓는 인덱스"""
        if self.circle is None:
            return np.argsort((pts - self.mean) @ self.dir)
        cx, cy, _ = self.circle
        a = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        a0 = math.atan2(self.mean[1] - cy, self.mean[0] - cx)
        rel = (a - a0 + np.pi) % (2 * np.pi) - np.pi            # 줄 가운데 기준 각도 (끊김 없게)
        idx = np.argsort(rel)
        if len(idx) >= 2 and (pts[idx[-1]] - pts[idx[0]]) @ self.dir < 0:   # 순서 방향을 self.dir 쪽으로
            idx = idx[::-1]
        return idx

    def curvature(self):
        """곡률 = 1/반지름 (직선이면 0)"""
        return 0.0 if self.circle is None else 1.0 / self.circle[2]


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ④ 이웃 묶기 + 원호 따라 늘리기                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_chars(chars, iou_th=0.4):
    """너무 겹치는 글자 박스는 점수 높은 것 하나만 (IoU 합치기)"""
    keep = []
    for c in sorted(chars, key=lambda c: -c["peak"]):
        if all(box_iou(c["box"], k["box"]) < iou_th for k in keep):
            keep.append(c)
    return keep


def char_size(chars):
    """글자 크기 대표값 (CRAFT 점수 봉우리는 실제 글자보다 작으므로 중앙값 사용)"""
    return float(np.median([max(c["w"], c["h"]) for c in chars])) if chars else 10.0


def group_chains(chars, link, gap=2.2):
    """가까운 글자끼리(글자 크기의 gap 배 이내) 묶음. 두 글자 사이 연결 점수가 너무 낮고
       거리도 멀면 끊음. 돌려줌: [[글자 인덱스...], ...]"""
    n = len(chars)
    if n == 0:
        return []
    sz = char_size(chars)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chars[i], chars[j]
            d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
            if d > gap * sz:
                continue
            mx, my = int((a["cx"] + b["cx"]) / 2), int((a["cy"] + b["cy"]) / 2)
            lk = float(link[min(my, link.shape[0] - 1), min(mx, link.shape[1] - 1)])
            if d <= 1.3 * sz or lk > 0.1:                      # 아주 가깝거나 조금이라도 이어져 있으면
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def grow_chain(chain, pool, region, max_add=12, search=0.6, accept=0.12):
    """줄 양 끝에서 원호(또는 직선)를 따라 다음 글자 자리를 예측하고, 그 주변에 글자 점수가
       있으면 붙임 (= 놓친 박스 추가). pool 에 있는 약한 글자를 먼저 쓰고, 없으면 지도에서 직접 찾음.
       돌려줌: (늘어난 글자 목록, Curve, 추가된 글자 수)"""
    chain = list(chain)
    added = 0
    sz = char_size(chain)
    used = np.zeros(region.shape, bool)
    for c in chain:
        x0, y0, x1, y1 = c["box"]
        used[y0:y1, x0:x1] = True
    for _ in range(max_add):
        pts = np.array([[c["cx"], c["cy"]] for c in chain])
        cur = Curve(pts)
        idx = cur.order(pts)
        chain = [chain[i] for i in idx]
        pts = pts[idx]
        if len(chain) >= 2:
            step = float(np.median(np.hypot(*np.diff(pts, axis=0).T)))
        else:
            step = 1.6 * sz
        step = float(np.clip(step, 0.8 * sz, 3.0 * sz))
        grown = False
        for end, sign in ((len(chain) - 1, +1), (0, -1)):
            ex, ey = pts[end]
            t = cur.tangent(ex, ey) * sign
            if cur.circle is not None:                          # 원호 위에서 한 걸음 (각도로 이동)
                cx, cy, R = cur.circle
                a = math.atan2(ey - cy, ex - cx)
                da = step / R * (1 if (np.array([-(ey - cy), ex - cx]) @ t) > 0 else -1)
                px, py = cx + R * math.cos(a + da), cy + R * math.sin(a + da)
            else:
                px, py = ex + t[0] * step, ey + t[1] * step
            rad = search * step
            # (가) 미리 찾아 둔 약한 글자 중 예측 자리 근처
            best = None
            for c in pool:
                if any(c is k for k in chain):
                    continue
                d = math.hypot(c["cx"] - px, c["cy"] - py)
                if d <= rad and (best is None or c["peak"] > best["peak"]):
                    best = c
            # (나) 없으면 점수 지도에서 직접
            if best is None:
                pk = peak_in_window(region, px, py, rad, used)
                if pk is None or pk[2] < accept:
                    continue
                bx, by, sc = pk
                h = sz / 2
                best = dict(cx=bx, cy=by, w=sz, h=sz, ang=0.0, peak=sc, new=True,
                            box=(int(bx - h), int(by - h), int(bx + h) + 1, int(by + h) + 1))
            if any(math.hypot(best["cx"] - k["cx"], best["cy"] - k["cy"]) < 0.6 * step for k in chain):
                continue                                        # 이미 있는 글자의 꼬리를 또 잡은 것
            x0, y0, x1, y1 = best["box"]
            used[max(0, y0):y1, max(0, x0):x1] = True
            chain.append(best)
            added += 1
            grown = True
        if not grown:
            break
    pts = np.array([[c["cx"], c["cy"]] for c in chain])
    cur = Curve(pts)
    idx = cur.order(pts)
    return [chain[i] for i in idx], cur, added


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑤ 글자마다 똑바로 세워서 한 줄로 펴기                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def straighten(g, chain, cur, h_scale=1.8, w_scale=1.0):
    """글자마다 접선 방향으로 회전해 잘라서 가로로 이어 붙임 → 휜 줄이 곧은 줄이 됨"""
    pts = np.array([[c["cx"], c["cy"]] for c in chain])
    if len(chain) >= 2:
        step = float(np.median(np.hypot(*np.diff(pts, axis=0).T)))
    else:
        step = 1.6 * char_size(chain)
    ph = int(max(12, h_scale * max(step, char_size(chain))))
    pw = int(max(8, w_scale * step))
    tiles = []
    for c in chain:
        t = cur.tangent(c["cx"], c["cy"])
        ang = math.degrees(math.atan2(t[1], t[0]))
        M = cv2.getRotationMatrix2D((c["cx"], c["cy"]), ang, 1.0)
        rot = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        tiles.append(cv2.getRectSubPix(rot, (pw, ph), (c["cx"], c["cy"])))
    pad = np.full((ph, pw // 2), int(np.median(g)), np.uint8)
    return np.hstack([pad] + tiles + [pad])


def tilt_list(chain, cur):
    """글자마다 기울기(도). 0 = 가로, +면 오른쪽이 아래로. 휜 줄이면 양 끝 부호가 반대"""
    out = []
    for c in chain:
        t = cur.tangent(c["cx"], c["cy"])
        out.append(round(math.degrees(math.atan2(t[1], t[0])), 1))
    return out


def detect_curved(g, strong=0.5, weak=0.2, min_chain=2):
    """한 알약 이미지 → 글자 줄 목록. 각 줄 = dict(chars, curve, added, tilts, strip)"""
    region, link = craft_maps(g)
    chars = nms_chars(find_chars(region, min_peak=weak))
    seeds = [c for c in chars if c["peak"] >= strong]
    pool = [c for c in chars if c["peak"] < strong]
    lines = []
    for grp in group_chains(seeds, link):
        chain = [seeds[i] for i in grp]
        chain, cur, added = grow_chain(chain, pool, region)
        if len(chain) < min_chain:
            continue
        pool = [p for p in pool if not any(p is c for c in chain)]
        lines.append(dict(chars=chain, curve=cur, added=added, tilts=tilt_list(chain, cur),
                          strip=straighten(g, chain, cur)))
    # 줄끼리 같은 글자를 잡았으면(IoU) 긴 줄만 남김
    lines.sort(key=lambda L: -len(L["chars"]))
    kept = []
    for L in lines:
        sz = char_size(L["chars"])
        dup = sum(1 for c in L["chars"] if any(box_iou(c["box"], k["box"]) > 0.4 or
                                              math.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"]) < 0.5 * sz
                                              for K in kept for k in K["chars"]))
        if dup < 0.5 * len(L["chars"]):
            kept.append(L)
    return kept, region, link, chars


def read_lines(lines, reader):
    """펴진 줄마다 0°/180° 둘 다 읽고 확신도 높은 쪽. 줄이 여러 개면 위→아래로 이어 붙임"""
    items = []
    for L in lines:
        best = ("", 0.0)
        for s in (L["strip"], cv2.rotate(L["strip"], cv2.ROTATE_180)):
            t, c = reader.read_crop(s)
            t = "".join(ch for ch in t if ch.isalnum())
            if t and c > best[1]:
                best = (t, c)
        L["read"], L["conf"] = best
        if best[0]:
            items.append((float(L["curve"].mean[0]), float(L["curve"].mean[1]), best[0], best[1]))
    text, conf = base.join_boxes(items)
    return base.norm(text), conf


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑥ 그림 (확인용, GitHub 에 올리지 않음)                              ║
# ╚════════════════════════════════════════════════════════════════════╝

def draw(g, lines, craft_boxes, chars):
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in craft_boxes:                         # 파랑: 원래 CRAFT 단어 박스
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 120, 0), 1)
    for c in chars:                                            # 회색 점: 찾은 글자 봉우리 전부
        cv2.circle(vis, (int(c["cx"]), int(c["cy"])), 2, (160, 160, 160), -1)
    for L in lines:
        cur = L["curve"]
        if cur.circle is not None:                             # 노랑: 맞춘 원
            cx, cy, R = cur.circle
            cv2.circle(vis, (int(cx), int(cy)), int(R), (0, 220, 255), 1)
        for c in L["chars"]:
            col = (0, 0, 255) if c.get("new") else (0, 200, 0)  # 초록: 원래 글자 / 빨강: 늘려서 새로 찾은 글자
            x0, y0, x1, y1 = c["box"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
            t = cur.tangent(c["cx"], c["cy"]) * 12
            cv2.arrowedLine(vis, (int(c["cx"]), int(c["cy"])), (int(c["cx"] + t[0]), int(c["cy"] + t[1])), col, 1)
    strips = [cv2.cvtColor(L["strip"], cv2.COLOR_GRAY2BGR) for L in lines]
    if strips:
        w = vis.shape[1]
        rows = [cv2.resize(s, (w, max(1, int(s.shape[0] * w / s.shape[1])))) if s.shape[1] > w else
                cv2.copyMakeBorder(s, 0, 0, 0, w - s.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for s in strips]
        vis = np.vstack([vis] + rows)
    return vis


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑦ 합성 곡선 글자로 동작 확인 (데이터 없이)                          ║
# ╚════════════════════════════════════════════════════════════════════╝

def synth_curved(text="ARICEPT", size=400, R=120, start_deg=200, step_deg=20, seed=0):
    """둥근 알약 위에 원호를 따라 글자를 새긴 합성 이미지 (글자 아래쪽이 원 중심을 향함)"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 90, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.45), 200, -1)
    for k, ch in enumerate(text):
        a = math.radians(start_deg + k * step_deg)
        x, y = c + R * math.cos(a), c + R * math.sin(a)
        tile = np.zeros((60, 60), np.uint8)
        cv2.putText(tile, ch, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((30, 30), -(math.degrees(a) + 90), 1.0)
        tile = cv2.warpAffine(tile, M, (60, 60))
        x0, y0 = int(x - 30), int(y - 30)
        roi = img[y0:y0 + 60, x0:x0 + 60]
        roi[:] = np.where(tile > 60, np.clip(roi.astype(int) - (tile.astype(int) * 0.45), 0, 255), roi).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img.astype(int) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def selftest(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = synth_curved()
    t0 = time.time()
    lines, region, link, chars = detect_curved(g)
    cb = base.craft_boxes(g)
    print(f"CRAFT 단어 박스 {len(cb)}개, 글자 봉우리 {len(chars)}개, 글자 줄 {len(lines)}개 ({time.time() - t0:.1f}s)")
    for L in lines:
        cur = L["curve"]
        circ = "직선" if cur.circle is None else f"원 중심({cur.circle[0]:.0f},{cur.circle[1]:.0f}) 반지름 {cur.circle[2]:.0f}"
        print(f"  글자 {len(L['chars'])}개 (늘려서 추가 {L['added']}), {circ}, 글자별 기울기 {L['tilts']}")
    cv2.imwrite(os.path.join(out_dir, "selftest_input.png"), g)
    cv2.imwrite(os.path.join(out_dir, "selftest_result.png"), draw(g, lines, cb, chars))
    cv2.imwrite(os.path.join(out_dir, "selftest_region.png"), (np.clip(region, 0, 1) * 255).astype(np.uint8))
    try:
        reader = base.PARSeq()
        print("PARSeq 펴진 줄 읽기:", read_lines(lines, reader))
        boxes = cb or [(0, 0, g.shape[1], g.shape[0])]
        items = [(x0, y0, reader.read_crop(g[y0:y1, x0:x1])[0], 1.0) for x0, y0, x1, y1 in boxes]
        print("PARSeq 기존 방식(CRAFT 박스별) 읽기:", base.norm(base.join_boxes(items)[0]))
    except Exception as e:
        print("PARSeq 를 불러오지 못해 읽기는 건너뜀:", e)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑧ 실제 사진 실행                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def judge(read, truth):
    if not truth:
        return "-"
    return "O" if read == truth else ("~" if truth in read else "X")


def main():
    ap = argparse.ArgumentParser(description="휜 각인 글자 박스 늘리기 실험 (CRAFT 글자 지도 + 원호 따라가기)")
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--shape", default="round", choices=["round", "all"], help="round = 둥근 알약만")
    ap.add_argument("--prep", default="P0")
    ap.add_argument("--out", default="curve_boxes_dev.csv")
    ap.add_argument("--viz", default="", help="그림 저장 폴더 (비우면 안 그림)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.viz or "curve_selftest")
        return
    if not a.manifest:
        ap.error("--manifest 가 필요해요 (또는 --selftest)")

    rows = base.load_manifest(a.manifest, a.split)
    reader = base.PARSeq()
    prep = base.PREPS[a.prep][1]
    if a.viz:
        os.makedirs(a.viz, exist_ok=True)
    out_rows, n = [], 0
    for r in rows:
        img = base.imread_any(r["image_path"])
        if img is None:
            continue
        crop, aspect, _, _ = base.crop_pill(img)
        if a.shape == "round" and aspect > 1.3:
            continue
        n += 1
        if a.limit and n > a.limit:
            break
        g = prep(crop)
        truth = base.norm(r["truth"])
        t0 = time.time()
        # 기준: 지금 방식 0° 한 번 읽기 (CRAFT 단어 박스마다 PARSeq)
        cb = base.craft_boxes(g)
        boxes = cb or [(0, 0, g.shape[1], g.shape[0])]
        items = []
        for x0, y0, x1, y1 in boxes:
            t, c = reader.read_crop(g[y0:y1, x0:x1])
            t = "".join(ch for ch in t if ch.isalnum())
            if t:
                items.append((x0, y0, t, c))
        read_base = base.norm(base.join_boxes(items)[0])
        # 새 방식: 글자 줄 늘리기 + 펴서 읽기
        lines, region, link, chars = detect_curved(g)
        read_new, conf_new = read_lines(lines, reader)
        L0 = lines[0] if lines else None
        out_rows.append({
            "file_name": r["file_name"], "split": r["split"], "drug_N": r.get("drug_N", ""),
            "rot_deg": r.get("rot_deg", ""), "camera_la": r.get("camera_la", ""), "truth": truth,
            "n_craft_boxes": len(cb), "n_chars": len(chars), "n_lines": len(lines),
            "n_line_chars": "|".join(str(len(L["chars"])) for L in lines),
            "n_added": sum(L["added"] for L in lines),
            "radius": "" if (L0 is None or L0["curve"].circle is None) else round(L0["curve"].circle[2], 1),
            "tilts": json.dumps(L0["tilts"]) if L0 else "",
            "line_angle": "" if L0 is None else round(math.degrees(math.atan2(*L0["curve"].dir[::-1])), 1),
            "read_base": read_base, "judge_base": judge(read_base, truth),
            "read_new": read_new, "conf_new": round(conf_new, 1), "judge_new": judge(read_new, truth),
            "line_reads": "|".join(L.get("read", "") for L in lines), "ms": int((time.time() - t0) * 1000),
        })
        print(f"{n:3d} {r['file_name'][:28]:28s} 정답 {truth:10s} 기존 {read_base:12s}{out_rows[-1]['judge_base']}  "
              f"새 {read_new:12s}{out_rows[-1]['judge_new']}  (줄 {len(lines)}, 추가 {out_rows[-1]['n_added']})")
        if a.viz:
            cv2.imwrite(os.path.join(a.viz, f"curve_{n:03d}_{r.get('rot_deg', '')}.png"), draw(g, lines, cb, chars))

    if not out_rows:
        print("결과 없음")
        return
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    m = len(out_rows)
    for k in ("base", "new"):
        o = sum(1 for x in out_rows if x[f"judge_{k}"] == "O")
        t = sum(1 for x in out_rows if x[f"judge_{k}"] in "O~")
        print(f"[{k:4s}] 정확히 일치 {o}/{m} ({100 * o / m:.1f}%), 포함까지 {t}/{m} ({100 * t / m:.1f}%)")
    print(f"※ 1장 = {100 / m:.1f}%p. 차이가 몇 장 수준이면 우연 범위일 수 있어요.  저장: {a.out}")


if __name__ == "__main__":
    main()
