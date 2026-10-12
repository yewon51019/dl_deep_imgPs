# =====================================================================
#  run_exp300_paddle.py  —  PaddleOCR 을 A실험과 '똑같은 조건'으로 돌리는 파일
# =====================================================================
#
#  [이 파일이 하는 일]
#   PaddleOCR 모델 하나만 넣어서 run_exp300.py 의 main() 을 그대로 실행합니다.
#   → 알약 자르기, 전처리 P0~P4, 회전 투표, 매칭, CSV 형식이 A실험과 100% 같아요.
#   → 결과 CSV 는 evaluate_exp300.py 로 그대로 채점됩니다. (CSV 형식을 따로 맞출 필요 없음)
#
#  [준비]  (Colab 기준)
#     !pip install paddlepaddle paddleocr
#     (그 밖의 준비는 run_exp300.ipynb 의 1~4번 셀과 같아요: git clone, manifest.csv, search_db.csv)
#
#  [PaddleOCR 두 설정]  (모델 이름)
#     paddleocr    : PaddleOCR 기본 검출 기준
#     paddleocr_t0 : 검출 기준 0.0 (팀원 B 실험과 같은 설정: text_det_thresh / text_det_box_thresh / text_rec_score_thresh = 0.0)
#   → 둘 다 A실험 파이프라인(자르기·전처리 P0~P10·회전 투표·DB 매칭)으로 돌려서 설정만 비교합니다.
#
#  [실행 순서]  — 할 일 3가지  (두 설정을 같이 돌려요)
#   ① 개발용 사진으로 두 설정 × 전처리 11개 돌리기
#        python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv \
#               --split dev --out pred_dev_paddle.csv
#   ② 점수를 보고 '설정마다' 가장 좋은 전처리 고르기  (예: paddleocr:P3, paddleocr_t0:P9)
#        python evaluate_exp300.py --pred pred_dev_paddle.csv --top 5
#   ③ 고른 조합으로 평가용 사진을 '한 번만' 돌리고 채점
#        python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv \
#               --split test --combos paddleocr:P3,paddleocr_t0:P9 --out pred_test_paddle.csv
#        python evaluate_exp300.py --pred pred_test_paddle.csv --cm all
#
#  [꼭 지킬 것]
#   · manifest.csv 는 남예원(A실험)과 '같은 파일'을 써야 개발용/평가용 사진이 같아서 비교가 돼요.
#   · 평가용 점수를 보고 전처리를 다시 고르면 안 돼요. (평가용은 마지막에 한 번만)
#   · 끊기면 같은 명령을 다시 실행하세요. 이미 끝낸 건 건너뛰고 이어서 합니다.
#
#  [PaddleOCR 버전]  v2.x (.ocr) 와 v3.x (.predict) 를 둘 다 지원해요. 설치된 쪽을 자동으로 씁니다.
# =====================================================================

import os
import sys

import cv2
import numpy as np

import run_exp300 as base
from extra_preps import PREPS_EXTRA
from run_exp300 import join_boxes


os.environ.setdefault("FLAGS_use_mkldnn", "0")      # Colab CPU 에서 oneDNN 오류(ConvertPirAttribute2RuntimeAttribute) 방지


def build_paddle(v3_kwargs, v2_kwargs):
    """PaddleOCR 만들기. v3 이면 oneDNN(enable_mkldnn)을 꺼서 Colab CPU 의 NotImplementedError 를 피하고,
       그 인자를 모르는 v3 면 빼고, v2 면 v2 이름의 인자로 만듭니다."""
    from paddleocr import PaddleOCR
    for extra in ({"enable_mkldnn": False}, {}):
        try:
            return PaddleOCR(**v3_kwargs, **extra)
        except (TypeError, ValueError):
            continue
    return PaddleOCR(**v2_kwargs)


class PaddleOCRModel:
    """PaddleOCR (글자 위치 찾기 + 읽기). 흑백 이미지 -> (읽은 글자, 신뢰도 0~100)
       기본 설정 = PaddleOCR 기본 검출 기준 (paddleocr)"""
    V3_EXTRA = {}          # 자식 클래스가 v3 용 추가 설정을 채움
    V2_EXTRA = {}          # 자식 클래스가 v2 용 추가 설정을 채움

    def __init__(self):
        self.ocr = build_paddle(
            dict(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                 use_textline_orientation=False, **self.V3_EXTRA),
            dict(use_angle_cls=False, lang="en", show_log=False, **self.V2_EXTRA))
        self.v3 = hasattr(self.ocr, "predict")             # v3: .predict / v2: .ocr


    def __call__(self, g):
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)      # PaddleOCR 은 3채널 이미지를 받음
        items = []
        if self.v3:
            for res in self.ocr.predict(rgb):
                texts, scores = res["rec_texts"], res["rec_scores"]
                boxes = res.get("rec_boxes")
                if boxes is None or len(boxes) == 0:
                    boxes = res.get("rec_polys") or res.get("dt_polys") or []
                for t, c, b in zip(texts, scores, boxes):
                    b = np.asarray(b).reshape(-1, 2) if np.asarray(b).size != 4 else np.asarray(b).reshape(2, 2)
                    items.append((float(b[:, 0].min()), float(b[:, 1].min()), t, float(c)))
        else:
            lines = self.ocr.ocr(rgb, cls=False)
            for box, (t, c) in (lines[0] if lines and lines[0] else []):
                items.append((min(p[0] for p in box), min(p[1] for p in box), t, float(c)))
        items = [(x, y, "".join(ch for ch in t if ch.isalnum()), c) for x, y, t, c in items]
        return join_boxes([it for it in items if it[2]])


class PaddleOCRModelT0(PaddleOCRModel):
    """검출 기준 0.0 설정 (팀원 B 실험과 같은 설정, paddleocr_t0):
       흐린 글자도 박스로 잡지만 쓸모없는 박스도 늘어남. 다른 부분은 기본 설정과 똑같음"""
    V3_EXTRA = dict(text_det_thresh=0.0, text_det_box_thresh=0.0, text_rec_score_thresh=0.0)
    V2_EXTRA = dict(det_db_thresh=0.0, det_db_box_thresh=0.0, drop_score=0.0)

# A실험의 모델 표에 PaddleOCR 한 줄 추가
base.MODELS["paddleocr"] = ("PaddleOCR (기본 검출 기준)", PaddleOCRModel)
base.MODELS["paddleocr_t0"] = ("PaddleOCR (검출 기준 0.0, B실험 설정)", PaddleOCRModelT0)
# 전처리 표에 추가 6종(P5~P10) 추가 → 기본값으로 P0~P10 모두 실행. 일부만 쓰려면 --preps P0,P5 처럼 지정
base.PREPS.update(PREPS_EXTRA)


def main():
    # --models / --combos 를 따로 지정하지 않으면 PaddleOCR 두 설정(paddleocr, paddleocr_t0)만 돌림 (다른 모델은 A실험에서 끝남)
    if not any(a.startswith(("--models", "--combos")) for a in sys.argv[1:]):
        sys.argv += ["--models", "paddleocr,paddleocr_t0"]
    base.main()


if __name__ == "__main__":
    main()
