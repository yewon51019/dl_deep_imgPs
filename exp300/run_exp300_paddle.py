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
#  [실행 순서]  — 할 일 3가지
#   ① 개발용 사진으로 PaddleOCR × 전처리 5종 돌리기
#        python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv \
#               --split dev --out pred_dev_paddle.csv
#   ② 점수를 보고 가장 좋은 전처리 고르기  (예: paddleocr:P3)
#        python evaluate_exp300.py --pred pred_dev_paddle.csv --top 1
#   ③ 고른 조합으로 평가용 사진을 '한 번만' 돌리고 채점
#        python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv \
#               --split test --combos paddleocr:P3 --out pred_test_paddle.csv
#        python evaluate_exp300.py --pred pred_test_paddle.csv --cm all
#
#  [꼭 지킬 것]
#   · manifest.csv 는 남예원(A실험)과 '같은 파일'을 써야 개발용/평가용 사진이 같아서 비교가 돼요.
#   · 평가용 점수를 보고 전처리를 다시 고르면 안 돼요. (평가용은 마지막에 한 번만)
#   · 끊기면 같은 명령을 다시 실행하세요. 이미 끝낸 건 건너뛰고 이어서 합니다.
#
#  [PaddleOCR 버전]  v2.x (.ocr) 와 v3.x (.predict) 를 둘 다 지원해요. 설치된 쪽을 자동으로 씁니다.
# =====================================================================

import sys

import cv2
import numpy as np

import run_exp300 as base
from run_exp300 import join_boxes


class PaddleOCRModel:
    """PaddleOCR (글자 위치 찾기 + 읽기). 흑백 이미지 -> (읽은 글자, 신뢰도 0~100)"""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.v3 = False
        try:                                           # v3.x : 문서 회전·펴기 기능은 끄고, 글자만 읽게 함
            self.ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                                 use_doc_unwarping=False, use_textline_orientation=False)
            self.v3 = hasattr(self.ocr, "predict")
        except (TypeError, ValueError):                # v2.x
            self.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

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


# A실험의 모델 표에 PaddleOCR 한 줄 추가
base.MODELS["paddleocr"] = ("PaddleOCR (DB 검출 + SVTR 읽기)", PaddleOCRModel)


def main():
    # --models / --combos 를 따로 지정하지 않으면 PaddleOCR 하나만 돌림 (다른 모델은 이미 A실험에서 끝남)
    if not any(a.startswith(("--models", "--combos")) for a in sys.argv[1:]):
        sys.argv += ["--models", "paddleocr"]
    base.main()


if __name__ == "__main__":
    main()
