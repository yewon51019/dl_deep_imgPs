# =====================================================================
#  dbpp_diag_cns.py  —  DBNet++ 가 '박스 0개' 인 사진 진단 (Paddle 전용 GPU 환경에서 실행, OCR 안 함)
# =====================================================================
#
#  [왜]  잘 못 찾은 사진 비교(hard_det_compare.csv)에서 DBNet++ 가 원본 86장 중 34장에서 박스를 하나도 못 찾음.
#        원인 후보: ① 기준을 낮춰 알약 전체가 한 덩어리 → '사진 넓이의 절반 초과는 버림' 거르기에 걸림
#                   ② 각인을 글자로 거의 안 봐서 확률이 처음부터 낮음   ③ 입력 크기(짧은 변 736)가 안 맞음
#  [대상]  dbpp_th0.1\orig_alignlong_line_hard\pred.csv 에서 fallback=1 인 사진, 방향은 OCR 때와 같은 축 정렬 방향
#  [하는 일]
#    (가) 방향마다 기준 t=0.1 에서 상자 후보가 버려진 이유를 셈
#         후보 = DBPostProcess(thresh=t, box_thresh=0) 가 낸 다각형 (넓히기까지 끝난 것)
#         → 넓이 > 사진 50% (너무 큼) / 넓이 < 40 (너무 작음) / 상자 안 평균 확률 < t (점수 낮음) / 남음
#         + 확률 지도 최댓값 (② 확인: 최댓값 자체가 낮으면 각인을 글자로 안 본 것)
#    (나) 그림 10장: 돌린 사진 | 확률 지도 | 확률 ≥ 0.3 / 0.1 / 0.05 이진화 (+ 거르기 통과 상자 초록, 너무 커서 버린 상자 빨강)
#    (다) 입력 크기(짧은 변) 400 / 736 / 1024 별로 '모든 방향에서 남은 상자 0개' 인 사진 수 (t = 0.1)
#  [저장]  exp_cns\results\ocr_v1\dev\dbpp_diag\diag_reasons.csv  (사진 × 방향 × 입력 크기)
#          exp_cns\results\ocr_v1\dev\dbpp_diag\sheet_dbpp_probmap.png  (그림 제목: DBNet++ 확률 지도 진단 (박스 0개 10장))
#  [실행 예]  .\.venv_paddle_gpu\Scripts\python.exe exp_cns\dbpp_diag_cns.py
# =====================================================================

import argparse
import csv
import os

import cv2
import numpy as np

import align_long_cns as al
import common_cns as cc
import ocr_common_cns as oc
from det2x_boxes_cns import Det2x

OUT = oc.OCR_OUT / "dev" / "dbpp_diag"
T = 0.1
SIZES = [400, 736, 1024]


def prob_map(d2, img, limit):
    """DBNet++ 확률 지도 (모델 입력 크기 그대로) + 크기 정보 shape."""
    x, shape = d2.prepare(img, limit)
    with d2.paddle.no_grad():
        out = d2.m(d2.paddle.to_tensor(x))
    return out["maps"].numpy()[0, 0], shape, out


def classify(d2, img, limit, post0):
    """한 방향 사진 → (이유별 개수 dict, 확률 지도, 크기, 다각형 목록과 각 이유)."""
    pm, shape, out = prob_map(d2, img, limit)
    polys = list(post0(out, shape[None])[0]["points"])                  # box_thresh=0: 점수로는 안 거른 후보 전부
    H, W = img.shape[:2]
    sy, sx = shape[2], shape[3]                                        # 원본 → 확률 지도 배율
    cnt = dict(cand=len(polys), too_big=0, too_small=0, low_score=0, kept=0)
    tagged = []
    for p in polys:
        p = np.asarray(p, np.float32).reshape(-1, 2)
        area = cv2.contourArea(p)
        if area > oc.PADDLE_MAX_AREA * H * W:
            why = "too_big"
        elif area < 40:
            why = "too_small"
        else:
            m = np.zeros(pm.shape, np.uint8)
            cv2.fillPoly(m, [np.round(p * [sx, sy]).astype(np.int32)], 1)
            why = "low_score" if m.any() and pm[m > 0].mean() < T else "kept"
        cnt[why] += 1
        tagged.append((p, why))
    return cnt, pm, shape, tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sheet", type=int, default=10)
    a = ap.parse_args()
    src_csv = oc.out_dir("dev", f"dbpp_th{T:g}", "orig", 0, 12, "long", "_line_hard") / "pred.csv"
    names = [r["file_name"] for r in csv.DictReader(open(src_csv, encoding="utf-8-sig")) if r["fallback"] == "1"]
    items = [it for it in oc.load_split("dev", 0) if it["file_name"] in set(names)]
    print(f"DBNet++ 박스 0개 사진 {len(items)}장 (원본, t={T})")

    d2 = Det2x("dbpp", T)                                              # PaddleOCR 2.x 소스 경로도 여기서 sys.path 에 들어감
    from ppocr.postprocess import build_post_process
    post0 = build_post_process({"name": "DBPostProcess", "thresh": T, "box_thresh": 0.0, "max_candidates": 1000,
                                "unclip_ratio": 1.5, "det_box_type": "quad"})
    prep = oc.Prep("orig", "dev")
    rows, sheet = [], []
    for it in items:
        src = cc.imread(it["image_path"])
        img, _ = prep(it, src)
        for ang in al.align_angles(src)[0]:
            rimg = oc.rotate_color(img, ang)
            for lim in SIZES:
                cnt, pm, shape, tagged = classify(d2, rimg, lim, post0)
                rows.append(dict(file_name=it["file_name"], truth=it["truth"], angle=ang, size=lim,
                                 prob_max=round(float(pm.max()), 3), prob_p99=round(float(np.percentile(pm, 99)), 3),
                                 frac_ge_t=round(float((pm >= T).mean()), 3), **cnt))
                if lim == 736 and len(sheet) < a.n_sheet and ang == al.align_angles(src)[0][0]:
                    sheet.append((it, rimg, pm, shape, tagged))

    os.makedirs(OUT, exist_ok=True)
    with open(OUT / "diag_reasons.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # (가)(다) 요약
    for lim in SIZES:
        rs = [r for r in rows if r["size"] == lim]
        tot = {k: sum(r[k] for r in rs) for k in ("cand", "too_big", "too_small", "low_score", "kept")}
        zero_img = sum(1 for fn in {r["file_name"] for r in rs} if sum(r["kept"] for r in rs if r["file_name"] == fn) == 0)
        print(f"  입력 {lim:4d}: 방향 {len(rs)}개, 후보 {tot['cand']} = 너무 큼 {tot['too_big']} + 너무 작음 {tot['too_small']} "
              f"+ 점수 낮음 {tot['low_score']} + 남음 {tot['kept']}  /  모든 방향 남은 상자 0개 사진 {zero_img}/{len(items)}  "
              f"/ 확률 최댓값 중앙값 {np.median([r['prob_max'] for r in rs]):.2f}, t 이상 픽셀 비율 중앙값 {np.median([r['frac_ge_t'] for r in rs]):.2f}")

    # (나) 그림 — Paddle 환경에는 matplotlib 이 없어서 OpenCV 로 (칸 제목은 영어만 가능)
    #   칸: 돌린 사진(초록 = 남은 상자, 빨강 = 너무 커서 버린 상자) | 확률 지도(밝을수록 높음, 0~1) | 확률 ≥ 0.3 / 0.1 / 0.05
    S = 260
    lines = []
    for it, rimg, pm, shape, tagged in sheet:
        H, W = rimg.shape[:2]
        pmr = cv2.resize(pm, (W, H))
        vis = rimg.copy()
        for p, why in tagged:
            col = {"kept": (0, 200, 0), "too_big": (0, 0, 255)}.get(why)
            if col:
                cv2.polylines(vis, [p.astype(np.int32)], True, col, 2)
        heat = cv2.applyColorMap((np.clip(pmr, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        bins = [cv2.cvtColor(((pmr >= th) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for th in (0.3, 0.1, 0.05)]
        titles = [f"{it['file_name'][:15]} truth {it['truth']}", f"prob map (max {pm.max():.2f})",
                  "prob >= 0.3", "prob >= 0.1", "prob >= 0.05"]
        cells = []
        for im, t in zip([vis, heat] + bins, titles):
            c = cc.fit_side(im, S, fill=(255, 255, 255))
            c = np.vstack([np.full((24, S, 3), 255, np.uint8), c])
            cv2.putText(c, t, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cells.append(c)
        lines.append(np.hstack(cells))
    cc.imwrite(OUT / "sheet_dbpp_probmap.png", np.vstack(lines))
    print(f"저장: {OUT / 'diag_reasons.csv'}, {OUT / 'sheet_dbpp_probmap.png'}")


if __name__ == "__main__":
    main()
