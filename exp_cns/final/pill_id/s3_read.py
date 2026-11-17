"""3단계: 글자 읽기 + 방향 고르기 — 상자마다 파인튜닝 PARSeq 로 읽고, 방향마다 이어 붙인 뒤 확신도가 가장 높은 방향을 고른다.

  - 상자가 0개인 방향은 사진 전체를 한 줄로 읽음 (fallback)
  - 이어 붙이기: 긴 알약(축 2방향)은 'line' (세로로 반 이상 겹치는 상자끼리 한 줄, 줄 안은 왼→오른, 줄끼리 위→아래),
                 그 밖(12방향)은 'tb' (위→아래, 그다음 왼→오른)
  - 방향 확신도 = 상자별 PARSeq 확신도 평균 × 100. 글자가 하나도 안 나온 방향은 뒤로.
(exp_cns/ocr_cns.py 의 Reader·join_line·read_direction·pick_best, exp300/run_exp300.py 의 norm·join_boxes 와 같은 계산)
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .imageio import fit_side, rotate_color


def norm(text):
    """읽은 글자 정리: 대문자로, 영문·숫자만 남김."""
    text = str(text or "")
    if text.strip().lower() == "nan":
        return ""
    return "".join(ch for ch in text.replace("마크", " ").upper() if ch.isalnum())


class Reader:
    """PARSeq (baudm/parseq 사전학습 구조 + 파인튜닝 가중치). 조각은 BGR 로 받아 RGB 로 바꿔 넣음."""

    def __init__(self, weights=None):
        import torch
        from torchvision import transforms as T
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # 모델 구조 코드는 torch.hub 로 baudm/parseq 에서 받음 (처음 한 번 인터넷 필요, 약 1MB).
        # 가중치는 아래에서 파인튜닝 파일로 전부 덮어쓰므로 사전학습 가중치(약 95MB)는 받지 않음 (pretrained=False).
        local = Path(torch.hub.get_dir()) / "baudm_parseq_main"
        try:
            self.m = torch.hub.load("baudm/parseq", "parseq", pretrained=not weights, trust_repo=True)
        except Exception:                             # GitHub 이 안 될 때: 전에 받아 둔 사본 (같은 코드)
            if not local.exists():
                raise
            self.m = torch.hub.load(str(local), "parseq", pretrained=not weights, source="local")
        self.m = self.m.eval().to(self.dev)
        if weights:
            self.m.load_state_dict(torch.load(weights, map_location=self.dev))
            self.m.eval()
        self.tf = T.Compose([T.Resize(self.m.hparams.img_size, T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(0.5, 0.5)])

    def read(self, crop_bgr):
        """조각 하나 → (글자, 확신도 0~1 = 글자별 확률 평균)."""
        from PIL import Image
        x = self.tf(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            p = self.m(x).softmax(-1)
        labels, confs = self.m.tokenizer.decode(p)
        c = confs[0]
        return labels[0], float(c.mean()) if len(c) else 0.0


def join_tb(items):
    """[(x0, y0, x1, y1, 글자, 확신도)] → 위→아래, 왼→오른. 줄 간격 = max(10, (가장 높은 y − 가장 낮은 y) / 3)."""
    if not items:
        return "", 0.0
    hs = [it[1] for it in items]
    gap = max(10, (max(hs) - min(hs)) / 3 if len(hs) > 1 else 10)
    items = sorted(items, key=lambda it: (round(it[1] / gap), it[0]))
    return "".join(it[4] for it in items), float(np.mean([it[5] for it in items]) * 100)


def join_line(items):
    """[(x0, y0, x1, y1, 글자, 확신도)] → 세로로 '두 높이 중 작은 쪽의 절반 이상' 겹치면 같은 줄."""
    if not items:
        return "", 0.0
    lines = []
    for it in sorted(items, key=lambda it: (it[1] + it[3]) / 2):
        y0, y1 = it[1], it[3]
        if lines:
            top, bot, mem = lines[-1]
            if min(bot, y1) - max(top, y0) >= 0.5 * min(y1 - y0, bot - top):
                lines[-1] = [min(top, y0), max(bot, y1), mem + [it]]
                continue
        lines.append([y0, y1, [it]])
    order = [it for _, _, mem in lines for it in sorted(mem, key=lambda it: it[0])]
    return "".join(it[4] for it in order), float(np.mean([it[5] for it in order]) * 100)


def read_direction(img, boxes, reader, join):
    """돌린 사진 한 장 + 상자 → dict(text, conf, boxes, texts, fallback)."""
    fallback = not boxes
    if fallback:
        boxes = [(0, 0, img.shape[1], img.shape[0])]
    items, texts = [], []
    for x0, y0, x1, y1 in boxes:
        t, c = reader.read(img[y0:y1, x0:x1])
        t = norm(t)
        texts.append(t)
        if t:
            items.append((x0, y0, x1, y1, t, c))
    text, conf = (join_line if join == "line" else join_tb)(items)
    return dict(text=norm(text), conf=conf, boxes=boxes, texts=texts, fallback=fallback)


def read_pill(src, angs, boxes_of, reader):
    """자른 원본 + 각도 목록 + 상자 함수(각도 → 상자) → (방향별 결과, 고른 방향, 400 사진)."""
    img = fit_side(src, C.SIDE)
    join = "line" if len(angs) == 2 else "tb"
    dirs = []
    for ang in angs:
        d = read_direction(rotate_color(img, ang), boxes_of(ang), reader, join)
        d["ang"] = ang
        dirs.append(d)
    best = max(dirs, key=lambda d: (bool(d["text"]), d["conf"]))
    return dirs, best, img
