# =====================================================================
#  deblur_check_cns.py  —  흐림 개선 비교 시트 (크기 맞추기 단계에서 무엇을 할지), 평가용 사진으로 그림만
# =====================================================================
#
#  [왜]  5b 결과에서 '글씨가 흐리다'는 사진을 확인해 보니, 점 제거가 아니라 원본 초점 흐림 + 작은 사진(긴 변 250px 미만
#        60/432장)을 400 으로 늘린 흐림이 주원인으로 보였다. 그래서 크기 맞추기 단계에서 쓸 방법을 비교한다.
#  [비교 (열)]
#     ① 앞 단계만 본 모습 (400 맞춤 직후, 컬러)
#        now   : 지금 방식 (비율 유지 보간)
#        SR    : 초해상도 ESPCN 으로 키운 뒤 400 (긴 변 400 미만인 사진만 달라짐)
#        RL1.5 : 지금 방식 + 디컨볼루션 Richardson-Lucy (번짐 σ1.5, 10회)
#        RL2.5 : 같은 것, 번짐 σ2.5
#     ② 그 위에 지금까지 고른 경로 하나(F4-P1-D3-R3-…: auto → 가우시안 k7σ1 → 평균5 → CLAHE c8 t12 → 샤프닝 → 끝에서 점 제거·획 보호)를
#        똑같이 적용한 최종 모습: now / SR / RL1.5 / RL2.5 / now 에서 평균 필터만 뺀 것(흐림 단계 줄이기)
#  [행]  흐리다고 하신 3장 + 샘플 8장 = 11장
#  [저장]  exp_cns\results\prep_v2\step0_deblur\sheet_step0_deblur.png (+ _zoom: 가운데 2배 확대), times.csv (한 장당 시간)
#  [실행 예]  python exp_cns\deblur_check_cns.py
# =====================================================================

import csv
import time

import cv2
import numpy as np

import common_cns as cc
import prep_steps as ps
import sweep_cns as sw

OUT = sw.OUT / "step0_deblur"
USER_BLURRY = ["K-000122_0_0_0_2_75_240_200.png", "K-000208_0_0_0_2_90_060_200.png", "K-002275_0_0_0_2_75_000_200.png"]
PRE = {"now": lambda bgr: cc.fit_side(bgr, ps.SIDE),
       "SR": ps.sr_fit400,
       "RL1.5": lambda bgr: ps.deconv_fit400(bgr, 1.5, 10),
       "RL2.5": lambda bgr: ps.deconv_fit400(bgr, 2.5, 10)}


def full_path(pre400, p):
    """앞 단계 결과(컬러 400 정사각형)에 고른 경로를 그대로 적용. apply_spot 은 원본을 받아 400 으로 맞추므로,
       이미 400 정사각형인 사진을 넣으면 크기 맞추기는 그대로 통과함 (fit_side 가 같은 크기를 돌려줌)."""
    return sw.apply_spot(pre400, p)


def main():
    p = {**dict(sw.picked("5a"))["P1-D3-R3-A5-T3-H8-M2-C6-G12"], **ps.SPOT_DEFAULT,
         "bright": True, "final_h": 6, "protect": True}                     # = 5b 채택 F4-P1-D3-R3-… 와 같은 설정
    items = {it["file_name"]: it for it in cc.load_items(cc.EVAL_ROOT)}
    samples = [r["file_name"] for r in csv.DictReader(open(sw.OUT / "samples8.csv", encoding="utf-8-sig"))]
    rows = [(fn, "user: blurry") for fn in USER_BLURRY] + [(fn, "sample8") for fn in samples]
    cols = [f"pre {k}" for k in PRE] + [f"full {k}" for k in PRE] + ["full no-mean"]
    times = {c: [] for c in cols}
    grid_full, grid_zoom = [], []
    for fn, why in rows:
        it = items[fn]
        bgr = cc.imread(it["image_path"])
        cells = []
        for k, f in PRE.items():
            t0 = time.perf_counter()
            pre = f(bgr)
            times[f"pre {k}"].append((time.perf_counter() - t0) * 1000)
            cells.append((f"pre {k}", pre))
        for k, f in PRE.items():
            pre = f(bgr)
            t0 = time.perf_counter()
            cells.append((f"full {k}", full_path(pre, p)))
            times[f"full {k}"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        cells.append(("full no-mean", full_path(PRE["now"](bgr), {**p, "mean_k": 1})))
        times["full no-mean"].append((time.perf_counter() - t0) * 1000)
        h, w = bgr.shape[:2]
        head = [f"{it['drug_N']}  truth={it['truth']}", f"rot{it['rot']} cam{it['camera_la']}  {max(h, w)}px", why]
        for zoom, grid in ((False, grid_full), (True, grid_zoom)):
            row = [sw._text_box(260, sw.CELL + sw.BAR, ["", "", *head], 0.45, 1)]
            row += [sw._cell(sw._zoom(im) if zoom else im, t) for t, im in cells]
            grid.append(np.hstack(row))
    OUT.mkdir(parents=True, exist_ok=True)
    cc.imwrite(str(OUT / "sheet_step0_deblur.png"), np.vstack(grid_full))
    cc.imwrite(str(OUT / "sheet_step0_deblur_zoom.png"), np.vstack(grid_zoom))
    with open(OUT / "times.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["column", "ms_mean", "n", "note"])
        for c in cols:
            w.writerow([c, round(float(np.mean(times[c][1:])), 2), len(times[c]),
                        "pre = 크기 맞추기만 / full = 그 뒤 고른 경로 전체(점 제거 포함)"])
    print("저장:", OUT)
    print("한 장당 평균 ms (첫 장 제외):", {c: round(float(np.mean(times[c][1:])), 1) for c in cols})


if __name__ == "__main__":
    main()
