# =====================================================================
#  run_exp300.py  —  A실험: 모델 5개 × 전처리 5개로 각인 읽고 약 찾기
# =====================================================================
#
#  [한 장의 사진이 지나가는 길]
#
#    사진 ─▶ ① 알약 자르기 ─▶ ② 전처리(P0~P4) ─▶ ③ 돌려가며 OCR(모델 5개 중 1개)
#         ─▶ ④ 다수결로 각인 1개 고르기 ─▶ ⑤ 검색 DB에서 가장 비슷한 약 찾기
#         ─▶ ⑥ predictions.csv 에 한 줄 저장  (채점은 evaluate_exp300.py 가 함)
#
#  [모델 5개]   (MODELS 표에서 켜고 끌 수 있어요)
#    tesseract : 문서 OCR. 기준선(baseline)
#    easyocr   : 글자 찾기(CRAFT) + 읽기(CRNN). A실험 주력
#    doctr     : 글자 찾기(DBNet) + 읽기(CRNN). 다른 딥러닝 OCR 비교용
#    parseq    : 읽기 전용 트랜스포머. 글자 찾기는 EasyOCR의 CRAFT를 빌려 씀
#    trocr     : 읽기 전용 트랜스포머(인쇄 문서용). 글자 찾기는 CRAFT를 빌려 씀
#
#  [전처리 5개]  (PREPS 표)
#    P0 : 없음 (자르기 + 배경 지우기만)
#    P1 : 조명 평탄화                  (A실험의 B)
#    P2 : 반사광 제거 + 조명 평탄화     (A실험의 D)
#    P3 : CLAHE                         (B실험 최고)
#    P4 : 2배 확대 + CLAHE              (작은 글자 대응, 새로 추가)
#
#  [실행 예]
#    1단계(개발용, 25조합 전부):  python run_exp300.py --manifest manifest.csv --db search_db.csv --split dev
#    2단계(평가용, 상위 조합만): python run_exp300.py ... --split test --combos easyocr:P2,doctr:P3
#
#  [이어하기]  Colab이 끊겨도 같은 --out 으로 다시 실행하면, 이미 한 것은 건너뛰고 이어서 해요.
# =====================================================================

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ╔════════════════════════════════════════════════════════════════════╗
# ║ 0. 공통 도구                                                        ║
# ╚════════════════════════════════════════════════════════════════════╝

def imread_any(path):
    """한글 경로도 읽을 수 있는 이미지 읽기"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


MARK_WORDS = ("마크",)   # AI Hub 라벨에서 '글자 대신 그림(로고)'이라는 뜻. 읽을 수 없으니 지움


def norm(text):
    """각인 글자 정리 규칙 (채점·검색 모두 이 규칙 하나만 써요)
       1) '마크'(그림 표시)라는 단어를 지우고
       2) 대문자로 바꾸고, 글자·숫자만 남김 (공백, 하이픈, 기호 삭제)
       예) 'Sp eed-1' -> 'SPEED1',  '마크' -> '' (채점에서 빠짐)"""
    text = str(text or "")
    if text.strip().lower() == "nan":
        return ""
    for w in MARK_WORDS:
        text = text.replace(w, " ")
    return "".join(ch for ch in text.upper() if ch.isalnum())


ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"   # OCR에 허용할 글자


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ① 알약 자르기 (AI Hub 사진용)                                       ║
# ╚════════════════════════════════════════════════════════════════════╝
#
#  AI Hub 사진은 단색 배경(검정/파랑/연회색) 위에 알약 1개가 작게 찍혀 있어요.
#  "사진 테두리의 색 = 배경색"이라고 보고, 배경색과 색이 많이 다른 부분을 알약으로 봐요.
#  (연회색 배경 위의 흰 알약처럼 밝기만으로는 구분이 안 되는 경우도 색 차이로 잡기 위해)

def segment_pill(bgr):
    """알약 마스크(흰색=알약)를 돌려줌. 못 찾으면 None"""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(5, int(min(h, w) * 0.04))                       # 테두리 두께 (4%)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)                           # 배경색 추정
    dist = np.linalg.norm(lab - bg, axis=2)                  # 배경색과의 색 차이
    dist8 = np.clip(dist * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist8 > max(thr, 20)).astype(np.uint8) * 255     # 너무 작은 차이는 무시
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.002 * h * w or area > 0.8 * h * w:            # 너무 작거나(먼지) 너무 크면(실패) 포기
        return None
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [cv2.convexHull(c)], -1, 255, -1)  # 볼록하게 채워서 글자 홈도 포함
    return out


def crop_pill(bgr):
    """알약을 잘라서 (흑백 이미지, 긴변/짧은변 비율, 색, 성공여부)를 돌려줌.
       - 길쭉한 알약은 긴 쪽이 가로가 되게 돌려서 잘라요.
       - 알약 바깥(배경)은 알약의 중간 밝기로 칠해서 OCR이 배경을 글자로 착각하지 않게 해요."""
    mask = segment_pill(bgr)
    if mask is None:                                         # 실패하면 가운데 50%만 사용
        h, w = bgr.shape[:2]
        crop = bgr[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return g, 1.0, detect_color(crop, None), False

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if rw < rh:                                              # 긴 쪽을 가로로
        rw, rh, ang = rh, rw, ang + 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    H, W = bgr.shape[:2]
    rot_img = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rot_msk = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
    size = (int(rw * 1.1) + 2, int(rh * 1.1) + 2)            # 10% 여유
    crop = cv2.getRectSubPix(rot_img, size, (cx, cy))
    cm = cv2.getRectSubPix(rot_msk, size, (cx, cy)) > 127

    k = max(3, int(min(crop.shape[:2]) * 0.03))              # 테두리 그림자는 빼고 안쪽만
    inner = cv2.erode(cm.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if inner.sum() < 50:
        inner = cm
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    out = np.full_like(g, int(np.median(g[inner])) if inner.any() else int(np.median(g)))
    out[inner] = g[inner]
    return out, rw / max(rh, 1.0), detect_color(crop, inner), True


def detect_color(bgr, m):
    """알약의 대표 색을 거칠게 판정 (조명에 따라 바뀌는 하양/회색은 '무채색'으로 묶음)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = m if m is not None and m.any() else np.ones(hsv.shape[:2], bool)
    H, S = int(np.median(hsv[..., 0][sel])), int(np.median(hsv[..., 1][sel]))
    if S < 50:
        return "무채색"
    for upper, name in [(8, "빨강"), (15, "주황"), (35, "노랑"), (85, "초록"), (130, "파랑"), (160, "보라")]:
        if H < upper:
            return name
    return "빨강"


def shape_group(aspect):
    """잘라낸 알약의 비율로 모양을 2가지로만 나눔 (자세히 나누면 오히려 자주 틀려요)"""
    return "길쭉" if aspect > 1.3 else "둥근"


# 검색 DB(라벨)의 색·모양 이름을 위의 거친 묶음으로 바꾸는 표
COLOR_GROUP = {"하양": "무채색", "흰색": "무채색", "회색": "무채색", "검정": "무채색",
               "노랑": "노랑", "주황": "주황", "갈색": "주황", "빨강": "빨강", "분홍": "빨강",
               "자주": "보라", "보라": "보라", "초록": "초록", "연두": "초록",
               "청록": "파랑", "파랑": "파랑", "남색": "파랑"}          # '투명'은 판단에서 뺌
SHAPE_GROUP = {"원형": "둥근", "타원형": "길쭉", "장방형": "길쭉"}     # 그 외(삼각형 등)는 판단에서 뺌


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ② 전처리 P0 ~ P4                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def resize_long(g, long_side):
    """긴 변을 long_side 픽셀로 맞춤 (부드러운 바이큐빅 확대/축소)"""
    s = long_side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def flatten_light(g):
    """조명 평탄화: 크게 흐린 사진(조명 지도)으로 나눠서 한쪽 그늘을 없앰"""
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, min(g.shape) * 0.08)).astype(np.float32)
    out = g.astype(np.float32) / (bg + 1) * float(np.median(g))
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_glare(g):
    """반사광 제거: 하얗게 날아간 작은 부분만 주변 값으로 메움"""
    med = float(np.median(g))
    m = ((g >= 240) & (g > med + 25)).astype(np.uint8) * 255
    if m.sum() == 0:
        return g
    return cv2.inpaint(g, cv2.dilate(m, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)


def clahe(g):
    """CLAHE: 사진을 8×8 칸으로 나눠 칸마다 대비를 키움 (음각의 그림자를 진하게)"""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


BASE = 400   # P0~P3 는 긴 변 400px, P4 는 그 2배(800px)

PREPS = {
    "P0": ("없음",              lambda g: resize_long(g, BASE)),
    "P1": ("조명평탄화",         lambda g: flatten_light(resize_long(g, BASE))),
    "P2": ("반사광제거+평탄화",  lambda g: flatten_light(remove_glare(resize_long(g, BASE)))),
    "P3": ("CLAHE",             lambda g: clahe(resize_long(g, BASE))),
    "P4": ("2배확대+CLAHE",      lambda g: clahe(resize_long(g, BASE * 2))),   # 확대한 다음 CLAHE
}


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ③ OCR 모델 5개                                                      ║
# ╚════════════════════════════════════════════════════════════════════╝
#  모든 모델을 "흑백 이미지 → (읽은 글자, 신뢰도 0~100)" 모양으로 맞춰서 바꿔 끼우기 쉽게 했어요.

def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class Tesseract:
    def __init__(self):
        import pytesseract
        self.t = pytesseract
        self.cfg = f"--psm 7 -c tessedit_char_whitelist={ALLOW}"     # 한 줄 글자로 읽기

    def __call__(self, g):
        g = cv2.copyMakeBorder(g, 25, 25, 25, 25, cv2.BORDER_REPLICATE)
        d = self.t.image_to_data(g, config=self.cfg, output_type=self.t.Output.DICT)
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return "".join(w for w, _ in words), float(np.mean([c for _, c in words]))


_EASY_READER = None


def easy_reader():
    """EasyOCR 리더는 무거워서 한 번만 만들고 같이 써요 (easyocr, parseq, trocr 공용)"""
    global _EASY_READER
    if _EASY_READER is None:
        import easyocr
        _EASY_READER = easyocr.Reader(["en"], gpu=_gpu(), verbose=False)
    return _EASY_READER


def join_boxes(items):
    """여러 글자 상자를 위→아래, 왼→오른 순서로 이어 붙임. items = [(x, y, 글자, 신뢰도0~1)]"""
    if not items:
        return "", 0.0
    hs = [it[1] for it in items]
    line_gap = max(10, (max(hs) - min(hs)) / 3 if len(hs) > 1 else 10)
    items = sorted(items, key=lambda it: (round(it[1] / line_gap), it[0]))
    return "".join(t for _, _, t, _ in items), float(np.mean([c for *_, c in items]) * 100)


class EasyOCR:
    def __init__(self):
        self.r = easy_reader()

    def __call__(self, g):
        res = self.r.readtext(g, allowlist=ALLOW, detail=1)
        return join_boxes([(min(p[0] for p in box), min(p[1] for p in box), t, c) for box, t, c in res])


class DocTR:
    def __init__(self):
        from doctr.models import ocr_predictor
        self.m = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
        if _gpu():
            try:
                self.m = self.m.cuda()
            except Exception:
                pass

    def __call__(self, g):
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        doc = self.m([rgb])
        items = []
        for block in doc.pages[0].blocks:
            for line in block.lines:
                for wd in line.words:
                    (x0, y0), _ = wd.geometry
                    items.append((x0 * 1000, y0 * 1000, wd.value, float(wd.confidence)))
        return join_boxes(items)


def craft_boxes(g):
    """EasyOCR의 글자 찾기(CRAFT)만 빌려서 글자 상자 좌표를 돌려줌 [(x0, y0, x1, y1)]"""
    hl, fl = easy_reader().detect(g)
    boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in hl[0]]
    for poly in fl[0]:                                       # 기울어진 상자는 감싸는 사각형으로
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    H, W = g.shape[:2]
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append((x0, y0, x1, y1))
    return out


class _CraftThenRead:
    """읽기 전용 모델 공통 틀: CRAFT로 상자를 찾고 → 상자마다 읽기 → 이어 붙이기.
       상자를 하나도 못 찾으면 알약 전체를 한 줄로 보고 읽어요."""
    def read_crop(self, crop_gray):          # 자식 클래스가 채움: (글자, 신뢰도0~1)
        raise NotImplementedError

    def __call__(self, g):
        boxes = craft_boxes(g) or [(0, 0, g.shape[1], g.shape[0])]
        items = []
        for x0, y0, x1, y1 in boxes:
            t, c = self.read_crop(g[y0:y1, x0:x1])
            t = "".join(ch for ch in t if ch.isalnum())
            if t:
                items.append((x0, y0, t, c))
        return join_boxes(items)


class PARSeq(_CraftThenRead):
    def __init__(self):
        import torch
        from torchvision import transforms as T
        self.torch = torch
        self.dev = "cuda" if _gpu() else "cpu"
        self.m = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(self.dev)
        self.tf = T.Compose([T.Resize(self.m.hparams.img_size, T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(0.5, 0.5)])

    def read_crop(self, crop_gray):
        from PIL import Image
        x = self.tf(Image.fromarray(crop_gray).convert("RGB")).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            p = self.m(x).softmax(-1)
        labels, confs = self.m.tokenizer.decode(p)
        c = confs[0]
        return labels[0], float(c.mean()) if len(c) else 0.0


class TrOCR(_CraftThenRead):
    def __init__(self):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.dev = "cuda" if _gpu() else "cpu"
        name = "microsoft/trocr-base-printed"
        self.proc = TrOCRProcessor.from_pretrained(name)
        self.m = VisionEncoderDecoderModel.from_pretrained(name).eval().to(self.dev)

    def read_crop(self, crop_gray):
        from PIL import Image
        px = self.proc(images=Image.fromarray(crop_gray).convert("RGB"), return_tensors="pt").pixel_values.to(self.dev)
        with self.torch.no_grad():
            out = self.m.generate(px, max_new_tokens=16, output_scores=True, return_dict_in_generate=True)
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        probs = [float(s.softmax(-1).max()) for s in out.scores]
        return text, float(np.mean(probs)) if probs else 0.0


#  모델 표: 이름 → (설명, 만드는 방법).  안 쓰려면 줄 앞에 # 를 붙이세요.
MODELS = {
    "tesseract": ("문서 OCR (기준선)",            Tesseract),
    "easyocr":   ("CRAFT + CRNN",                EasyOCR),
    "doctr":     ("DBNet + CRNN",                DocTR),
    "parseq":    ("CRAFT + PARSeq(트랜스포머)",   PARSeq),
    "trocr":     ("CRAFT + TrOCR(트랜스포머)",    TrOCR),
}


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ④ 돌려가며 읽고 다수결로 고르기                                     ║
# ╚════════════════════════════════════════════════════════════════════╝

def rotate(g, ang):
    """사진을 ang도 돌림 (모서리가 잘리지 않게 캔버스를 키우고, 빈 곳은 테두리 색으로)"""
    if ang == 0:
        return g
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def read_with_rotation(g, aspect, ocr, step):
    """둥근 알약은 각인 방향을 모르니 step도씩 360도 돌며 읽고(30도면 12번),
       길쭉한 알약은 이미 가로로 폈으니 0도/180도 2번만 읽음.
       → 가장 많이 나온 글자(다수결), 동점이면 신뢰도가 높은 것을 고름.
       ※ 파일 이름의 회전각(정답 정보)은 쓰지 않아요. 실제 앱에서는 모르는 값이니까요."""
    angles = [0, 180] if aspect > 1.3 else list(range(0, 360, step))
    cands = []
    for a in angles:
        try:
            text, conf = ocr(rotate(g, a))
        except Exception as e:                       # 한 각도에서 오류가 나도 나머지는 계속
            text, conf = "", 0.0
            print(f"    [경고] {a}도 읽기 실패: {e}")
        cands.append((norm(text), float(conf)))
    votes = Counter(t for t, _ in cands if 1 <= len(t) <= 12)
    pick = max(cands, key=lambda c: (votes.get(c[0], 0) if c[0] else -1, c[1]))
    return pick[0], pick[1], cands


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑤ 검색 DB에서 약 찾기                                               ║
# ╚════════════════════════════════════════════════════════════════════╝

def levenshtein(a, b):
    """편집거리: a를 b로 바꾸려면 글자를 최소 몇 번 고쳐야 하나"""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def similarity(a, b):
    """각인 유사도 (Heo et al., 2023): (길이-편집거리)/(길이+편집거리). 같으면 1"""
    if not a or not b:
        return 0.0
    L, ed = max(len(a), len(b)), levenshtein(a, b)
    return (L - ed) / (L + ed)


MIN_SIM = 0.3   # 이보다 덜 비슷하면 '판정 없음'


def load_db(path):
    """search_db.csv 읽기 + 검색에 쓸 값 준비 (앞/뒷면 각인, 모양 묶음, 색 묶음)"""
    db = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            colors = {COLOR_GROUP[c.strip()] for c in f"{r.get('color1', '')},{r.get('color2', '')}".split(",")
                      if c.strip() in COLOR_GROUP}
            db.append({"drug_N": r["drug_N"], "front": norm(r.get("print_front")), "back": norm(r.get("print_back")),
                       "shape": SHAPE_GROUP.get(r.get("shape", "").strip()), "colors": colors})
    return db


def identify(pred, shape, color, db):
    """읽은 각인(pred)으로 약을 찾음. 3가지 방식의 결과를 한꺼번에 돌려줌 ('' = 판정 없음)
       exact : DB의 앞면 또는 뒷면 각인과 '완전히 같을 때'만 인정
       edit  : 편집거리로 가장 비슷한 약 (1등 동점이면 판정 없음)     ← 모델 비교의 기준
       attr  : 모양·색이 맞는 후보로 먼저 좁힌 뒤 edit
               (후보가 1개만 남으면 글자를 못 읽어도 그 약으로 판정)"""
    def best(cands):
        if not pred or not cands:
            return "", 0.0
        sc = sorted(((max(similarity(pred, d["front"]), similarity(pred, d["back"])), d["drug_N"]) for d in cands),
                    reverse=True)
        if sc[0][0] < MIN_SIM or (len(sc) > 1 and sc[0][0] == sc[1][0]):
            return "", sc[0][0]
        return sc[0][1], sc[0][0]

    hits = [d["drug_N"] for d in db if pred and pred in (d["front"], d["back"])]
    exact = hits[0] if len(hits) == 1 else ""
    edit, sim = best(db)
    narrowed = [d for d in db if (d["shape"] is None or d["shape"] == shape)
                and (not d["colors"] or color in d["colors"])] or db
    attr = narrowed[0]["drug_N"] if len(narrowed) == 1 else best(narrowed)[0]
    return exact, edit, attr, sim, len(narrowed)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ⑥ 실행: 조합마다 사진을 돌리고 predictions.csv 에 쌓기               ║
# ╚════════════════════════════════════════════════════════════════════╝

OUT_COLS = ["file_name", "split", "model", "prep", "combo", "truth_drug", "truth_imprint",
            "pred_imprint", "conf", "pred_exact", "pred_edit", "pred_attr", "sim", "n_attr_cands",
            "det_shape", "det_color", "crop_ok", "candidates", "ms",
            "form", "drug_dir", "camera_la", "rot_deg"]


def load_manifest(path, split):
    with open(path, encoding="utf-8-sig") as f:
        allrows = list(csv.DictReader(f))
    if split != "all":
        allrows = [r for r in allrows if r["split"] == split]
    rows = [r for r in allrows if norm(r["truth"])]           # 정답 각인이 비거나 '마크'뿐인 사진은 제외
    if len(rows) < len(allrows):
        marks = sum(1 for r in allrows if not norm(r["truth"]) and str(r["truth"]).strip())
        print(f"채점 제외: {len(allrows) - len(rows)}장 (그 중 정답이 '마크'뿐인 사진 {marks}장)")
    return rows


def parse_combos(text, models, preps):
    """'easyocr:P2,doctr:P3' → [('easyocr','P2'), ('doctr','P3')].  없으면 전체 조합"""
    if not text:
        return [(m, p) for m in models for p in preps]
    out = []
    for part in text.split(","):
        m, p = part.strip().split(":")
        out.append((m.strip(), p.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description="A실험: 모델×전처리 조합으로 각인 읽고 약 찾기")
    ap.add_argument("--manifest", required=True, help="make_manifest.py 가 만든 채점표")
    ap.add_argument("--db", required=True, help="make_search_db.py 가 만든 검색 DB")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--models", default=",".join(MODELS), help="쓸 모델 (쉼표로)")
    ap.add_argument("--preps", default=",".join(PREPS), help="쓸 전처리 (쉼표로)")
    ap.add_argument("--combos", default="", help="조합을 직접 지정 (예: easyocr:P2,doctr:P3). 지정하면 위 두 옵션 무시")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--step", type=int, default=30, help="둥근 알약 회전 간격(도)")
    ap.add_argument("--seed", type=int, default=0, help="사진 순서 섞기 시드")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞에서 N장만 (0=전부)")
    a = ap.parse_args()

    items = load_manifest(a.manifest, a.split)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(items)                                   # 사진 순서를 섞어서 진행 (약제별로 몰려 있지 않게)
    if a.limit:
        items = items[: a.limit]
    db = load_db(a.db)
    combos = parse_combos(a.combos, [m.strip() for m in a.models.split(",")], [p.strip() for p in a.preps.split(",")])
    print(f"사진 {len(items)}장 ({a.split}) × 조합 {len(combos)}개, 검색 DB {len(db)}종")

    # 이어하기: 이미 저장된 (사진, 모델, 전처리)는 건너뜀
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8-sig") as f:
            done = {(r["file_name"], r["model"], r["prep"]) for r in csv.DictReader(f)}
        print(f"이어하기: 이미 {len(done)}건 완료")
    new_file = not os.path.exists(a.out)
    fout = open(a.out, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fout, fieldnames=OUT_COLS)
    if new_file:
        writer.writeheader()

    # 알약 자르기는 조합과 상관없이 같으니, 사진마다 한 번만 해서 저장해 둠
    crops = {}

    for model_name in dict.fromkeys(m for m, _ in combos):           # 모델 순서대로 (모델을 한 번만 불러오게)
        todo = [(m, p) for m, p in combos if m == model_name]
        need = [(it, p) for _, p in todo for it in items if (it["file_name"], model_name, p) not in done]
        if not need:
            continue
        print(f"\n[{model_name}] {MODELS[model_name][0]} 불러오는 중...")
        try:
            ocr = MODELS[model_name][1]()
        except Exception as e:
            print(f"  [건너뜀] {model_name} 을(를) 불러오지 못했어요: {e}")
            continue

        for k, (it, prep) in enumerate(need, 1):
            fn = it["file_name"]
            if fn not in crops:
                img = imread_any(it["image_path"])
                if img is None:
                    print(f"  [경고] 이미지를 못 읽음: {it['image_path']}")
                    continue
                crops[fn] = crop_pill(img)
            gray, aspect, color, ok = crops[fn]
            t0 = time.time()
            g = PREPS[prep][1](gray)
            pred, conf, cands = read_with_rotation(g, aspect, ocr, a.step)
            shape = shape_group(aspect)
            exact, edit, attr, sim, n_c = identify(pred, shape, color, db)
            writer.writerow({
                "file_name": fn, "split": it["split"], "model": model_name, "prep": prep,
                "combo": f"{model_name}:{prep}", "truth_drug": it["drug_N"], "truth_imprint": norm(it["truth"]),
                "pred_imprint": pred, "conf": round(conf, 1), "pred_exact": exact, "pred_edit": edit,
                "pred_attr": attr, "sim": round(sim, 3), "n_attr_cands": n_c, "det_shape": shape,
                "det_color": color, "crop_ok": int(ok), "candidates": json.dumps([c[0] for c in cands]),
                "ms": int((time.time() - t0) * 1000), "form": it.get("form", ""), "drug_dir": it.get("drug_dir", ""),
                "camera_la": it.get("camera_la", ""), "rot_deg": it.get("rot_deg", ""),
            })
            fout.flush()                                  # 끊겨도 여기까지는 남도록 바로 저장
            if k % 20 == 0 or k == len(need):
                print(f"  {k}/{len(need)}  (최근: {fn[:24]}… {prep} → '{pred}' / 정답 '{norm(it['truth'])}')")
    fout.close()
    print(f"\n저장 완료: {a.out}   → 다음: python evaluate_exp300.py --pred {a.out}")


if __name__ == "__main__":
    main()
