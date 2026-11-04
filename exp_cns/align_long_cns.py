# =====================================================================
#  align_long_cns.py  —  ① 긴 알약 축 정렬: 알약의 긴 축을 찾아 가로로 눕히기
# =====================================================================
#
#  [왜]  1라운드 오답 원인을 세어 보니 긴 알약 오답의 대부분(Paddle 94장 중 71장)이
#        '글자는 다 맞게 읽었는데 붙이는 순서만 틀림' (예: JEILTRS → TRSJEIL).
#        알약이 세로로 선 방향에서는 분할선 양쪽 글자 상자가 위아래로 쌓여 '위→아래' 순으로 붙기 때문.
#        긴 축을 가로로 눕힌 뒤 0°/180° 두 방향만 읽으면 상자가 좌→우로 놓여 순서 문제가 줄어들 것으로 기대.
#  [방법]
#    1) 사진을 긴 변 400 정사각형으로 맞춤 (ocr_cns 의 '원본' 과 같은 좌표)
#    2) 배경색 = 네 모서리 작은 칸의 중앙값.  크롭 사진은 알약 박스에 딱 맞게 잘려서 테두리 가운데는 알약이 닿지만,
#       모서리는 알약(둥근 끝)이 잘 안 닿으므로 모서리만 씀  (exp300 segment_pill 은 테두리 띠 전체를 써서 여기선 안 맞음)
#    3) 배경색과의 Lab 색 거리 → Otsu 문턱 → 가장 큰 덩어리 = 알약 마스크
#    4) cv2.minAreaRect 로 '알약을 감싸는 가장 작은 기울어진 사각형' → 긴 변/짧은 변 비율과 긴 축 각도
#    5) 비율 > 1.3 이면 '긴 알약' (exp300 shape_group 과 같은 기준), 이하면 '원형' → 지금처럼 12방향
#       모양은 라벨이 아니라 사진에서 판정 (실제로 쓸 때는 라벨이 없으므로)
#  [v2 (--v 2) 에서 바꾼 것]  v1 시트에서 흰 알약 + 연파랑 배경 + 한쪽 조명이면 그늘진 반쪽만 알약으로 잡혀
#     원형 18장(K-000166, K-000135)이 '긴 알약' 으로 잘못 판정됨 → 두 가지를 더함
#    a) GrabCut 으로 알약 영역 다시 잡기 (pill_mask_v2): 색 분포(가우시안 혼합)로 알약/배경을 나누고 경계를 다듬는 방법.
#       - v1 영역(조금 깎은 것) = '확실한 알약' 씨앗,  모서리 칸 중 배경색에 가까운 픽셀 = '확실한 배경'
#       - 나머지(그늘진 반쪽·두 색 캡슐의 다른 반쪽 등) = '아마 알약' 으로 두고 GrabCut 이 정함
#       - 정사각형으로 맞추며 채운 여백(fit_side 의 패딩)은 '확실한 배경'
#    b) 안전장치 (cover): 크롭 사진은 알약 박스에 딱 맞게 잘려 있으므로 정상이라면 알약을 감싸는 상자가
#       사진(패딩 뺀 부분)의 가로·세로를 거의 다 덮어야 함. min(가로 덮은 비율, 세로 덮은 비율) < 0.85 이면
#       '영역을 덜 잡음' → 긴 알약이라도 원형처럼 12방향으로 읽음 (지금 방식 그대로 = 손해 없음)
#  [각도 약속]  돌리는 각도는 ocr_common_cns.rotate_color(img, ang) 와 같은 약속(OpenCV getRotationMatrix2D, 반시계 +).
#               rotate_color(img, axis_angle) 하면 긴 축이 가로가 됨. 읽을 방향 = axis_angle, axis_angle + 180
#  [이번 단계]  --sheet : 확인용 그림만 (OCR 안 함)
#     exp_cns\results\align_v1\sheet_align_long.png   : 개발용 '라벨상 긴 알약' 20장 (고르게 뽑음)
#                                                       칸 = 원본 | 알약 영역(초록) + 긴 축 선(빨강) | 가로로 돌린 결과
#     exp_cns\results\align_v1\long_axis_dev.csv      : 개발용 468장 전부의 비율·각도·사진 판정 vs 라벨 모양 (그림 없음)
#     --v 2 : exp_cns\results\align_v1\sheet_align_long_v2.png (긴 20장 + v1 에서 잘못 판정된 원형 6장 + 두 색 캡슐 4장)
#             exp_cns\results\align_v1\long_axis_dev_v2.csv (cover 열 추가)  — v1 파일은 덮어쓰지 않음
#  [실행 예]  python exp_cns\align_long_cns.py --sheet           (v1)
#             python exp_cns\align_long_cns.py --sheet --v 2    (v2)
# =====================================================================

import argparse
import csv
import os
import time

import cv2
import numpy as np

import common_cns as cc
import ocr_common_cns as oc

OUT = cc.RESULTS / "align_v1"
LONG_RATIO = 1.3                     # 긴 변/짧은 변 > 1.3 → 긴 알약 (exp300 과 같은 기준)
# 안전장치: 감싸는 상자가 사진 가로·세로를 이만큼은 덮어야 '영역을 제대로 잡음'
#   v2 (GrabCut) 는 배경까지 넘쳐 잡아 각도가 0°/90° 로 굳고 cover 가 늘 1.00 → 실패, 기록으로만 남김
#   v3 (= v1 영역 + 안전장치, OCR 에 쓰는 판) : 개발용에서 v1 오판 원형 18장의 cover 0.50~0.72, 0.80 이상만 받으면
#   원형 오판 0장 · 라벨 긴 알약 189/216 장 정렬 (0.75 는 194장이지만 원형 최댓값 0.72 와 여유가 작아 0.80)
MIN_COVER = {2: 0.85, 3: 0.80}


def content_rect(src_hw, side=400):
    """cc.fit_side(원본, side) 결과에서 실제 사진이 놓인 사각형 (x, y, w, h). 나머지는 정사각형을 맞추려 채운 여백."""
    h, w = src_hw
    s = side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return (side - nw) // 2, (side - nh) // 2, nw, nh


def pill_mask(img):
    """긴 변 400 정사각형 컬러 사진 → 알약 마스크(0/255). 못 찾으면 None."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    c = max(6, int(min(h, w) * 0.06))                                    # 모서리 칸 크기 (6%)
    corners = np.concatenate([lab[:c, :c].reshape(-1, 3), lab[:c, -c:].reshape(-1, 3),
                              lab[-c:, :c].reshape(-1, 3), lab[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)                                      # 배경색 추정
    dist8 = np.clip(np.linalg.norm(lab - bg, axis=2) * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (dist8 > max(thr, 20)).astype(np.uint8) * 255                    # 너무 작은 색 차이는 배경으로
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # 작은 점 지우기
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # 각인 홈 때문에 생긴 구멍 메우기
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))                     # 가장 큰 덩어리
    if st[k, cv2.CC_STAT_AREA] < 0.05 * h * w:                           # 사진의 5% 도 안 되면 실패로 봄
        return None
    return (lab_cc == k).astype(np.uint8) * 255


def pill_mask_v2(img, crect):
    """v2: v1 영역을 씨앗으로 GrabCut 을 돌려 알약 영역을 다시 잡음 → 마스크(0/255) 또는 None.
       crect = content_rect (패딩 뺀 실제 사진 위치)."""
    m1 = pill_mask(img)
    if m1 is None:
        return None
    h, w = img.shape[:2]
    x0, y0, cw, ch = crect
    gc = np.full((h, w), cv2.GC_PR_FGD, np.uint8)                        # 기본 = '아마 알약'
    # 모서리 칸(실제 사진 기준 6%) 중 v1 이 배경이라고 본 픽셀 = '확실한 배경'
    c = max(6, int(min(cw, ch) * 0.06))
    corner = np.zeros((h, w), bool)
    for yy, xx in ((y0, x0), (y0, x0 + cw - c), (y0 + ch - c, x0), (y0 + ch - c, x0 + cw - c)):
        corner[yy:yy + c, xx:xx + c] = True
    gc[corner & (m1 == 0)] = cv2.GC_BGD
    # 실제 사진 테두리 한 줄(2px) 중 v1 배경 = '아마 배경' (알약이 테두리에 닿는 곳은 '아마 알약' 그대로)
    edge = np.zeros((h, w), bool)
    edge[y0:y0 + ch, x0:x0 + cw] = True
    inner = np.zeros((h, w), bool)
    inner[y0 + 2:y0 + ch - 2, x0 + 2:x0 + cw - 2] = True
    gc[edge & ~inner & (m1 == 0) & ~corner] = cv2.GC_PR_BGD
    # v1 영역을 조금 깎은 것 = '확실한 알약' (경계 근처는 GrabCut 이 다시 정하게)
    gc[cv2.erode(m1, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    # 정사각형 여백(패딩) = '확실한 배경'
    gc[~edge] = cv2.GC_BGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, None, bgm, fgm, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:                                                    # 배경 씨앗이 없는 등 GrabCut 이 못 돌면 v1 그대로
        return m1
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return m1
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))                     # 가장 큰 덩어리만
    return (lab_cc == k).astype(np.uint8) * 255


def long_axis(img, crect=None, v=1):
    """→ dict(ok, ratio, angle, rect, mask, cover).  angle = rotate_color 로 돌리면 긴 축이 가로가 되는 각도 (0~180).
       v=2 면 GrabCut 영역(pill_mask_v2) + cover(감싸는 상자가 실제 사진 가로·세로를 덮은 비율 중 작은 쪽) 계산.
       마스크를 못 찾으면 ok=False, ratio=1 (→ 원형 취급, 12방향)."""
    if crect is None:
        crect = (0, 0, img.shape[1], img.shape[0])
    m = pill_mask_v2(img, crect) if v == 2 else pill_mask(img)          # v1·v3 는 색 거리 영역
    if m is None:
        return dict(ok=False, ratio=1.0, angle=0.0, rect=None, mask=None, cover=0.0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    if rw < rh:                                                          # 긴 쪽이 rw 가 되게 (exp300 crop_pill 과 같은 처리)
        rw, rh, ang = rh, rw, ang + 90
    bx, by, bw, bh = cv2.boundingRect(cv2.boxPoints(rect).astype(np.float32))
    cover = min(bw / crect[2], bh / crect[3])                            # 1 에 가까울수록 알약 전체를 잡은 것
    return dict(ok=True, ratio=rw / max(rh, 1.0), angle=float(ang % 180), rect=rect, mask=m,
                cover=float(min(cover, 1.0)), v=v)


def is_long(info):
    """긴 알약으로 보고 축 정렬할지. v2·v3 는 영역을 덜 잡은 것(cover < MIN_COVER)은 원형처럼 12방향으로 (안전장치)."""
    if not (info["ok"] and info["ratio"] > LONG_RATIO):
        return False
    v = info.get("v", 1)
    return v == 1 or info["cover"] >= MIN_COVER[v]


def align_angles(src):
    """OCR 용 (--align long): 원본 컬러 사진 → (읽을 각도 목록, 축 정보).  축은 '원본'을 긴 변 400 정사각형으로 맞춘 것에서 구함
       (전처리 결과도 같은 400 정사각형 좌표라 같은 각도를 그대로 씀).
       - 긴 알약(v3 판정) : [축 각도, 축 각도 + 180]  (소수 첫째 자리로 반올림 — boxes.json 의 키가 이 숫자의 글자)
       - 그 밖             : 지금처럼 30° 간격 12방향 (정수 → 1라운드 boxes.json 상자를 그대로 다시 쓸 수 있음)"""
    img = cc.fit_side(src, 400)
    info = long_axis(img, content_rect(src.shape[:2]), 3)
    if is_long(info):
        a = round(info["angle"], 1)
        return [a, round((a + 180) % 360, 1)], info
    return list(oc.ANGLES), info


# ---------------------------------------------------------------------
# 확인용 그림
# ---------------------------------------------------------------------
def draw_axis(img, info):
    """알약 영역(초록 반투명) + 감싸는 사각형(노랑) + 긴 축 선(빨강) + 비율·각도 글자."""
    out = img.copy()
    if not info["ok"]:
        cv2.putText(out, "mask fail", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    green = np.zeros_like(out)
    green[..., 1] = 255
    sel = info["mask"] > 0
    out[sel] = (0.65 * out[sel] + 0.35 * green[sel]).astype(np.uint8)
    box = cv2.boxPoints(info["rect"]).astype(np.int32)
    cv2.polylines(out, [box], True, (0, 220, 255), 2)
    (cx, cy), (rw, rh), _ = info["rect"]
    L = max(rw, rh) / 2
    # 긴 축 방향: 사진을 angle 만큼 반시계로 돌리면 가로가 되므로, 원래 축은 화면에서 시계 방향으로 angle 기운 선
    # (영상 좌표는 y 가 아래로 + 라서 시계 방향 = dy 가 + 쪽)
    t = np.deg2rad(info["angle"])
    dx, dy = L * np.cos(t), L * np.sin(t)
    cv2.line(out, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (0, 0, 255), 3)
    cv2.putText(out, f"ratio {info['ratio']:.2f}  ang {info['angle']:.0f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    judge = "LONG" if is_long(info) else ("12 dirs (low cover)" if info["ratio"] > LONG_RATIO else "ROUND (12 dirs)")
    cv2.putText(out, judge + (f"  cover {info['cover']:.2f}" if info.get("v") == 2 else ""), (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def make_sheet(items, path, title):
    """행마다 (원본 | 축 그림 | 가로로 돌린 결과) 를 matplotlib 으로 한 장에 (한글 제목·파일명 표시)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    n = len(items)
    cols = 2                                                             # 한 줄에 사진 2장 (각 3칸) → 6칸
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 3 * 2.6, rows * 2.8))
    for ax in axes.ravel():
        ax.axis("off")
    for i, (it, img, info) in enumerate(items):
        r, c0 = i // cols, (i % cols) * 3
        rot = oc.rotate_color(img, info["angle"]) if is_long(info) else img
        panels = [(img, f"{it['file_name'][:-4]}\n라벨 {it['shape']} / 정답 {it['truth']}" + (f"  [{it['why']}]" if it.get("why") else "")),
                  (draw_axis(img, info), "알약 영역 + 긴 축"),
                  (rot, "가로로 돌림" if is_long(info) else "12방향 그대로 → 안 돌림")]
        for j, (im, t) in enumerate(panels):
            ax = axes[r, c0 + j]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(t, fontsize=7)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(path.parent, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="확인용 그림 + 개발용 전부의 축 표 (OCR 안 함)")
    ap.add_argument("--n", type=int, default=20, help="그림에 넣을 긴 알약 수")
    ap.add_argument("--v", type=int, default=1, choices=[1, 2, 3],
                    help="1 = 색 거리 영역, 2 = GrabCut 영역 + 안전장치(실패), 3 = 색 거리 영역 + 안전장치 0.80 (OCR 용)")
    a = ap.parse_args()
    if not a.sheet:
        ap.error("지금은 --sheet 만 있습니다 (OCR 연결은 확인 뒤 ocr_cns.py 에 추가)")
    tag = "" if a.v == 1 else f"_v{a.v}"                                 # v2·v3 는 파일 이름에 _v2·_v3 → v1 결과를 덮어쓰지 않음

    items = oc.load_split("dev", 0)
    rows, cache, times = [], {}, []
    for it in items:
        src = cc.imread(it["image_path"])
        img = cc.fit_side(src, 400)
        t0 = time.perf_counter()
        info = long_axis(img, content_rect(src.shape[:2]), a.v)
        times.append((time.perf_counter() - t0) * 1000)
        cache[it["file_name"]] = (it, img, info)
        rows.append(dict(file_name=it["file_name"], drug_N=it["drug_N"], label_shape=it["shape"], camera_la=it["camera_la"],
                         rot=it["rot"], color2=it["color2"], mask_ok=int(info["ok"]), ratio=round(info["ratio"], 3),
                         angle=round(info["angle"], 1), cover=round(info["cover"], 3),
                         judged="긴" if is_long(info) else "원형(12방향)"))

    os.makedirs(OUT, exist_ok=True)
    with open(OUT / f"long_axis_dev{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    longs = [v for v in cache.values() if v[0]["shape"] != "원형"]
    step = max(1, len(longs) // a.n)                                     # 라벨상 긴 알약에서 고르게 n 장 (v1 과 같은 20장)
    pick = longs[::step][:a.n]
    if a.v >= 2:
        # v1 에서 원형인데 '긴' 으로 잘못 판정된 사진 중 고르게 6장 (v1 표가 있으면)
        f1 = OUT / "long_axis_dev.csv"
        if f1.exists():
            bad = [r["file_name"] for r in csv.DictReader(open(f1, encoding="utf-8-sig"))
                   if r["label_shape"] == "원형" and r["judged"] == "긴"]
            for fn in bad[::max(1, len(bad) // 6)][:6]:
                it, img, info = cache[fn]
                pick.append(({**it, "why": "v1 원형→긴 오판"}, img, info))
        # 두 색 캡슐(라벨 color2 있음) 중 아직 안 고른 것에서 고르게 4장
        chosen = {p[0]["file_name"] for p in pick}
        two = [v for v in cache.values() if v[0]["color2"] and v[0]["file_name"] not in chosen]
        for it, img, info in two[::max(1, len(two) // 4)][:4]:
            pick.append(({**it, "why": "두 색 캡슐"}, img, info))
    title = f"긴 알약 정렬 확인 (개발용 {len(pick)}장)" if a.v == 1 else f"긴 알약 정렬 확인 v{a.v} (개발용 {len(pick)}장)"
    make_sheet(pick, OUT / f"sheet_align_long{tag}.png", title)

    # 사진 판정 vs 라벨 모양 (긴 = 장방형·타원형)
    lab_long = [r for r in rows if r["label_shape"] != "원형"]
    lab_round = [r for r in rows if r["label_shape"] == "원형"]
    print(f"저장: {OUT / f'sheet_align_long{tag}.png'}, {OUT / f'long_axis_dev{tag}.csv'}")
    print(f"  마스크 실패 {sum(1 - r['mask_ok'] for r in rows)}장,  축 찾기 평균 {np.mean(times):.0f}ms/장")
    print(f"  라벨 긴 알약 {len(lab_long)}장 중 사진도 '긴' 판정 {sum(r['judged'] == '긴' for r in lab_long)}장")
    print(f"  라벨 원형 {len(lab_round)}장 중 사진이 '긴' 으로 잘못 판정 {sum(r['judged'] == '긴' for r in lab_round)}장")
    if a.v >= 2:
        low = [r for r in rows if r["ratio"] > LONG_RATIO and r["cover"] < MIN_COVER[a.v]]
        print(f"  비율은 긴데 덮은 비율 < {MIN_COVER[a.v]} 이라 12방향으로 돌린 사진 {len(low)}장 "
              f"(라벨 긴 {sum(r['label_shape'] != '원형' for r in low)} / 라벨 원형 {sum(r['label_shape'] == '원형' for r in low)})")


if __name__ == "__main__":
    main()
