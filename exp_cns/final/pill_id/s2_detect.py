"""2단계: 글자 상자 찾기 — 파인튜닝 PaddleOCR 검출기(det_ft1, unclip 2.0)로 방향마다 글자 상자를 찾는다.

⚠ Paddle 전용 환경(.venv_paddle_gpu)에서 실행한다. 본 환경(.venv)에 Paddle GPU 판을 깔면 OpenCV·numpy 가 바뀌어
  결과가 달라질 수 있어서, 이 단계만 따로 돌려 boxes.json 으로 저장하고 3단계가 그 파일을 읽는다.
  이 파일은 torch 를 import 하지 않는다 (Windows 에서 Paddle 을 torch 보다 먼저 부르면 torch DLL 오류가 나는 문제와 무관하게).
검출 기준은 모델 기본값(0.2 / 0.45). 상자는 감싸는 반듯한 사각형으로 바꾸고, 너무 작은 것·알약 전체를 잡은 것은 뺀다.
"""
import json
import os
import time

import cv2
import numpy as np

from . import config as C
from .imageio import fit_side, rotate_color


def polys_to_rects(polys, H, W):
    """검출 다각형 → 감싸는 사각형 [(x0, y0, x1, y1)]. 넓이 40 미만·사진 절반 초과는 뺌, 4px 미만도 뺌."""
    out = []
    for p in polys:
        p = np.asarray(p, np.float32).reshape(-1, 2)
        if not (C.MIN_BOX_AREA <= cv2.contourArea(p) <= C.MAX_BOX_AREA * H * W):
            continue
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append((int(x0), int(y0), int(x1), int(y1)))
    return out


class Detector:
    """파인튜닝 검출기. GPU 판 Paddle 이면 GPU, 아니면 CPU (oneDNN 오류 대비해 enable_mkldnn=False 먼저 시도)."""

    def __init__(self, model_dir=C.DET_MODEL, unclip=C.UNCLIP):
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        import paddle
        from paddleocr import TextDetection
        dev = "gpu:0" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
        th = dict(model_name="PP-OCRv6_medium_det", model_dir=str(model_dir), unclip_ratio=unclip)
        last = None
        for kw in ({"enable_mkldnn": False, "device": dev}, {"device": dev}, {}):
            try:
                self.m = TextDetection(**th, **kw)
                self.device = kw.get("device", "default")
                return
            except (TypeError, ValueError) as e:      # 이 Paddle 판이 모르는 인자 → 인자를 줄여 다시
                last = e
        raise RuntimeError(f"검출기를 만들지 못했습니다 (모델 폴더: {model_dir}). 마지막 오류: {last!r}") from last

    def boxes(self, img):
        res = list(self.m.predict(img))[0]
        return polys_to_rects(res["dt_polys"], *img.shape[:2])


def detect_all(det, pills, angles_of):
    """알약마다 모든 방향의 상자 → {이름: {"각도": {"boxes": [...], "det_ms": ms}}} (boxes.json 형식).
    pills = [(이름, 자른 원본)], angles_of = s1_orient.angles. 사진은 3단계와 똑같이 400 맞춤 후 돌림."""
    res = {}
    for name, src in pills:
        img = fit_side(src, C.SIDE)
        res[name] = {}
        for ang in angles_of(src)[0]:
            t0 = time.perf_counter()
            b = det.boxes(rotate_color(img, ang))
            res[name][str(ang)] = {"boxes": b, "det_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return res


def load_boxes(path):
    """boxes.json → 함수 get(이름, 각도) → [(x0, y0, x1, y1)]."""
    d = json.load(open(path, encoding="utf-8"))
    return lambda name, ang: [tuple(b) for b in d[name][str(ang)]["boxes"]]
