# =====================================================================
#  diag_paddle.py — 기본 검출 기준(paddleocr)이 왜 전부 '빈 글자'인지 확인 (몇 분)
# =====================================================================
#  개발용 사진 몇 장을 '자른 알약(P0)' 에 넣고, 검출 기준을 바꿔 가며 박스/글자 개수를 비교해요.
#  · 기본값에서 0개, 낮춘 값에서 N개  → 검출 기준이 너무 엄격한 것 (정상 동작, 결과 그대로 보고)
#  · 모든 설정에서 0개 / 오류        → 입력 이미지나 설치 문제 (코드 쪽 확인 필요)
#  사용:  python diag_paddle.py --manifest manifest.csv --n 4
import argparse
import cv2
import numpy as np
import run_exp300 as base
from run_exp300_paddle import build_paddle

SETTINGS = [
    ("기본값",                   {}, {}),
    ("box_thresh 0.3",           dict(text_det_box_thresh=0.3), dict(det_db_box_thresh=0.3)),
    ("thresh 0.1 / box 0.3",     dict(text_det_thresh=0.1, text_det_box_thresh=0.3), dict(det_db_thresh=0.1, det_db_box_thresh=0.3)),
    ("전부 0.0 (t0)",            dict(text_det_thresh=0.0, text_det_box_thresh=0.0, text_rec_score_thresh=0.0),
                                 dict(det_db_thresh=0.0, det_db_box_thresh=0.0, drop_score=0.0)),
]

def read(ocr, g):
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if hasattr(ocr, "predict"):
        out = []
        for r in ocr.predict(rgb):
            out += [(t, round(float(s), 2)) for t, s in zip(r["rec_texts"], r["rec_scores"])]
            n_poly = len(r.get("dt_polys", []))
        return out, n_poly
    lines = ocr.ocr(rgb, cls=False)
    out = [(t, round(float(c), 2)) for _, (t, c) in (lines[0] if lines and lines[0] else [])]
    return out, len(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    items = base.load_manifest(a.manifest, "dev")[: a.n]
    crops = []
    for it in items:
        g, *_ = base.crop_pill(base.imread_any(it["image_path"]))
        crops.append((it, base.resize_long(g, base.BASE)))
    for name, v3, v2 in SETTINGS:
        ocr = build_paddle(dict(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                                use_textline_orientation=False, **v3),
                           dict(use_angle_cls=False, lang="en", show_log=False, **v2))
        print(f"\n=== {name} ===")
        for it, g in crops:
            try:
                texts, n_poly = read(ocr, g)
                print(f"  정답 {base.norm(it['truth']):<12} 검출박스 {n_poly:2d}개 → 글자 {texts}")
            except Exception as e:
                print(f"  정답 {base.norm(it['truth']):<12} 오류: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
