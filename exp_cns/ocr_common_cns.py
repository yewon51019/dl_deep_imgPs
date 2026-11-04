# =====================================================================
#  ocr_common_cns.py  —  OCR 실험 공통 부분 (torch 를 쓰지 않음)
# =====================================================================
#
#  [왜 따로]  PaddleOCR 검출을 GPU 로 돌리려고 Paddle 전용 가상환경(.venv_paddle_gpu)을 따로 만들었다.
#             그 환경에는 torch(PARSeq)가 없으므로, 두 환경이 같이 쓰는 부분(사진 고르기·전처리·돌리기·Paddle 상자)을 여기에 모음.
#             - ocr_cns.py (본 환경 .venv)          : 이 파일 + PARSeq 읽기 + 채점 + dt 그림
#             - paddle_boxes_cns.py (.venv_paddle_gpu): 이 파일로 사진 × 12방향의 Paddle 상자만 계산해 boxes.json 으로 저장
#             두 쪽이 같은 함수로 사진을 고르고 돌리므로 상자 좌표가 그대로 맞음.
# =====================================================================

import csv
import json
import os
import time

import cv2
import numpy as np

import common_cns as cc
import sweep_cns as sw

def angles(n=12):
    """n 방향 각도 목록. 12 → 30° 간격 (1라운드 기본), 36 → 10° 간격 (방향을 잘게 쪼개 보는 시험)."""
    return list(range(0, 360, 360 // n))


ANGLES = angles(12)                           # 12방향 (자른 사진 속 알약이 기울어져 있어 모든 알약을 12방향으로)
FINAL = sw.OUT / "final"                      # 최종 전처리(평가용) 폴더
FINAL_DEV = sw.OUT / "final_dev"              # 같은 설정으로 만든 개발용 전처리 저장 위치
FINAL_TRAIN500 = sw.OUT / "final_train500"    # 검출기 파인튜닝 라벨용 학습 500장의 전처리 저장 위치 (개발용과 섞이지 않게 따로)
FINAL_HARD = sw.OUT / "final_hardset"         # 흐린 사진 시험(hardset)의 전처리 저장 위치
HARDSET_CSV = cc.RESULTS / "simple_prep_v2" / "groups.csv"    # hard_prep_cns.py --step select 가 고른 사진 목록
FINAL_VALDEV = sw.OUT / "final_valdev"        # 섞는 비율 튜닝(valdev = 검증 828 + 개발 468)의 전처리 저장 위치
OCR_OUT = cc.RESULTS / "ocr_v1"
TRAIN500_CSV = cc.RESULTS / "det_label_v1" / "train500.csv"   # make_det_labels_cns.py --step select 가 고른 학습용 500장
LAMBDA_DRUGS = {"K-000027", "K-001061", "K-005676", "K-025922", "K-030177"}   # Λ 각인 약 (정신과 약에 없음, 학습에서 뺌)
EXP300_TEST = [cc.ROOT / "exp300" / "data" / "약제각인인식_테스트셋", cc.ROOT / "exp300" / "data" / "약제각인인식_테스트셋2"]
EXP300_MANIFEST = cc.ROOT / "exp300" / "results" / "manifest.csv"   # exp300 의 dev/test 분할
CROP51 = {"exp300dev51_crop1": "crop_v1", "exp300dev51_crop2": "crop_v2"}   # 사후 비교용 사진 목록 이름 → crop_cns.py 결과 폴더
PADDLE_MAX_AREA = 0.5                        # 사진 넓이의 절반이 넘는 상자는 '알약 전체'를 잡은 것 (align_exp300.PADDLE_MAX_AREA 와 같은 값)


def load_split(split, limit):
    """split 에 해당하는 사진 목록. limit 가 있으면 고르게 띄엄띄엄 limit 장 (시험 운전용, 약·각도가 섞이게).
       split='train500' → 학습용(train) 중 TRAIN500_CSV 에 적힌 500장 (검출기 파인튜닝 자동 상자용, 개발용·평가용과 안 겹침).
       split='trainall' → 학습용(train) 전체 (개발용 36종·Λ 각인 약 5종 제외, PARSeq 파인튜닝 조각 라벨용).
       split='exp300test' → exp300/data 의 AI Hub 테스트셋 두 폴더 (300장·27종, 학습·개발용과 약이 안 겹침) 중
                            정답 각인이 있는 288장 (마크뿐 12장은 exp300 과 같이 채점에서 뺌) — exp_cns 최종 평가용, 한 번만
       split='trainval' → trainall 중 attr_cnn_v1 labels.csv 의 part=val (검증 약 65종 828장, 약 판별 λ·w 고르기용).
       split='exp300dev51_crop1' / 'exp300dev51_crop2' → exp300test 중 exp300 분할에서 dev 인 51장만, 사진은 crop_cns.py 가 자른 것
                            (crop1 = 기존 영역 crop_v1, crop2 = 넉넉 자르기 crop_v2). 두 자르기 중 무엇을 쓸지 고르는 사후 비교용.
       split='hardset' → hard_prep_cns.py --step select 가 고른 흐린 사진·참고 약 (HARDSET_CSV). 평가용 237장은 들어가지 않음."""
    if split == "valdev":                                             # 검증 828 + 개발 468 (blur_mix_cns.py 튜닝용, 평가용 없음)
        items = load_split("trainval", 0) + load_split("dev", 0)
    elif split == "hardset":
        names ={r["file_name"] for r in csv.DictReader(open(HARDSET_CSV, encoding="utf-8-sig"))}
        items, seen = [], set()
        for src in ("dev", "trainall", "exp300dev51_crop2"):          # 사진 하나가 여러 묶음에 있어도 한 번만
            for it in load_split(src, 0):
                if it["file_name"] in names and it["file_name"] not in seen:
                    items.append(it)
                    seen.add(it["file_name"])
        assert seen == names, names - seen
    elif split in CROP51:
        dev51 = {r["file_name"] for r in csv.DictReader(open(EXP300_MANIFEST, encoding="utf-8-sig")) if r["split"] == "dev"}
        items = [it for it in load_split("exp300test", 0) if it["file_name"] in dev51]
        assert len(items) == 51, len(items)
        for it in items:                                              # 라벨은 그대로, 사진 경로만 자른 사진으로 바꿈
            it["image_path"] = str(cc.RESULTS / CROP51[split] / "exp300test" / "images" / it["file_name"])
    elif split == "test":
        items = cc.load_items(cc.EVAL_ROOT)
    elif split == "train500":
        names = {r["file_name"] for r in csv.DictReader(open(TRAIN500_CSV, encoding="utf-8-sig"))}
        items = [it for it in cc.load_items(cc.TRAIN_ROOT) if it["file_name"] in names]
    elif split == "exp300test":
        items = [it for root in EXP300_TEST for it in cc.load_items(root) if it["truth"]]
    elif split == "trainval":
        val = {r["file_name"] for r in csv.DictReader(open(cc.RESULTS / "attr_cnn_v1" / "labels.csv", encoding="utf-8-sig"))
               if r["part"] == "val"}
        items = [it for it in load_split("trainall", 0) if it["file_name"] in val]
    elif split == "trainall":
        allt = cc.load_items(cc.TRAIN_ROOT)
        sp = cc.split_train(allt)
        items = [it for it in allt if sp[it["file_name"]] == "train" and it["drug_N"] not in LAMBDA_DRUGS]
    else:
        allt = cc.load_items(cc.TRAIN_ROOT)
        sp = cc.split_train(allt)
        items = [it for it in allt if sp[it["file_name"]] == "dev"]
    items = sorted(items, key=lambda it: it["file_name"])
    if limit:
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    return items


def hard_names(split, prep):
    """'박스를 잘 못 찾은 사진' 이름 집합 — 검출기 비교(det2x_boxes_cns / ocr_cns --hard)에 씀.
       기준 결과 = Paddle 기본 + 축 정렬 + 줄 묶기 (paddle_t0\\<prep>_alignlong_line\\pred.csv) 에서
       ① 박스를 하나도 못 찾음(fallback) ② 아무것도 못 읽음 ③ 읽은 글자가 정답의 일부(정답에 포함되지만 같지 않음)."""
    f = OCR_OUT / split / "paddle_t0" / f"{prep}_alignlong_line" / "pred.csv"
    out = set()
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        rd, t = r["read"], r["truth"]
        if r["fallback"] == "1" or not rd or (rd != t and rd in t):
            out.add(r["file_name"])
    return out


def out_dir(split, det, prep, limit, n_ang=12, align="none", tail=""):
    """결과 폴더. 시험 운전(limit)은 '__tryN' 을 붙여 본 실행과 섞이지 않게.
       12방향이 아니면 '_a{방향 수}' 를 붙임 (예: orig_a36) → 1라운드(12방향) 결과를 덮어쓰지 않음.
       긴 알약 축 정렬(align='long')이면 '_alignlong' (예: orig_alignlong). tail 은 그 뒤에 더 붙일 말 (예: '_line_hard')."""
    name = prep if n_ang == 12 else f"{prep}_a{n_ang}"
    if align == "long":
        name += "_alignlong"
    name += tail
    return OCR_OUT / split / det / (name if not limit else f"{name}__try{limit}")


class Prep:
    """prep='orig' → 비율 유지로 긴 변 400 맞춤만 (기준선). 그 밖 → final 폴더 이름: params.json 설정을 sweep_cns.apply6 으로 적용.
       평가용은 final 폴더에 저장된 결과를 그대로 읽고, 개발용은 final_dev 에 없으면 만들어 저장 (처리 시간도 prep_ms.csv 에 기록)."""

    def __init__(self, name, split):
        self.name, self.split = name, split
        if name == "orig":
            return
        self.p = json.load(open(FINAL / name / "params.json", encoding="utf-8"))["params"]
        if split in ("trainall", "trainval", "exp300test"):
            raise ValueError("trainall·trainval 은 orig 전처리만 (전처리 저장 폴더 없음)")
        self.cache = {"test": FINAL, "train500": FINAL_TRAIN500, "hardset": FINAL_HARD, "valdev": FINAL_VALDEV}.get(split, FINAL_DEV) / name
        self.tfile = self.cache / "prep_ms.csv"
        self.times = {}
        if self.tfile.exists():
            self.times = {r["file_name"]: float(r["prep_ms"]) for r in csv.DictReader(open(self.tfile, encoding="utf-8-sig"))}

    def __call__(self, it, bgr):
        """→ (전처리 결과 컬러 400×400, 처리 시간 ms). 저장된 결과를 읽은 경우 시간은 처음 만들 때 잰 값 (없으면 빈칸)."""
        if self.name == "orig":
            t0 = time.perf_counter()
            out = cc.fit_side(bgr, 400)
            return out, (time.perf_counter() - t0) * 1000
        f = self.cache / it["file_name"]
        if f.exists():
            return cc.imread(str(f)), self.times.get(it["file_name"], "")
        t0 = time.perf_counter()
        out = sw.apply6(bgr, self.p)
        ms = (time.perf_counter() - t0) * 1000
        cc.imwrite(str(f), out)
        new = not self.tfile.exists()
        with open(self.tfile, "a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["file_name", "prep_ms"])
            w.writerow([it["file_name"], round(ms, 2)])
        self.times[it["file_name"]] = ms
        return out, ms


def rotate_color(img, ang):
    """컬러 사진을 ang 도 돌림 (모서리가 잘리지 않게 캔버스를 키우고, 빈 곳은 가장자리 색으로 채움). run_exp300.rotate 의 컬러판."""
    if ang == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


_PD = {}


def polys_to_rects(polys, H, W):
    """검출기가 낸 다각형들 → 감싸는 사각형 [(x0, y0, x1, y1)].
       너무 작은(넓이 40 미만) 상자와 알약 전체를 잡은(사진 넓이의 절반 초과) 상자는 뺌 — boost_exp300.paddle_t0_polys 와 같은 기준.
       (검출기 비교에서 모든 검출기에 같은 거르기를 씀)"""
    out = []
    for p in polys:
        p = np.asarray(p, np.float32).reshape(-1, 2)
        if not (40 <= cv2.contourArea(p) <= PADDLE_MAX_AREA * H * W):
            continue
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append((int(x0), int(y0), int(x1), int(y1)))
    return out


def paddle_t0_polys(img):
    """paddle_t0_boxes 와 같은 검출·같은 거르기를 하되, 감싸는 사각형과 함께 검출기가 낸 원래 네 꼭짓점도 돌려줌.
       → [((x0, y0, x1, y1), [[x, y] × 4]), ...]. 검출기 학습 라벨(det_polys_cns.py)용: 사각형은 저장된 상자와 짝 맞추기에,
         꼭짓점은 글자 기울기를 따라가는 라벨로 씀 (반듯한 사각형은 기울어진 글자에서 넉넉해짐)."""
    paddle_boxes(img, 0.0, 0.0)                                       # 모델 준비 (paddle_t0_boxes 와 같은 설정 열쇠)
    res = list(_PD[(0.0, 0.0)].predict(img))[0]
    H, W = img.shape[:2]
    out = []
    for p in res["dt_polys"]:
        r = polys_to_rects([p], H, W)                                 # 한 개씩 넣어 같은 거르기 (빠지면 [])
        if r:
            out.append((r[0], np.asarray(p, np.float32).reshape(-1, 2).tolist()))
    return out


def paddle_boxes(img, thresh=None, box_thresh=None, model_dir=None, unclip=None):
    """PaddleOCR 검출(TextDetection) 상자 → 감싸는 사각형. thresh·box_thresh 가 None 이면 모델 기본값.
       model_dir: 파인튜닝해서 내보낸 검출기 폴더 (예: det_ft1\\infer, 구조는 기본과 같은 PP-OCRv6_medium_det). None 이면 기본 모델.
       unclip: 상자 넓히기 비율 (None 이면 모델 설정값 1.4).
       ⚠ paddleocr 3.7.0(PaddleX)은 내부에서 'thresh or 기본값' 으로 처리해서 0.0 을 넣으면 기본값(0.2 / 0.45)이 됨
         (2026-11-01 사진 1장으로 확인: 0.0 과 기본 설정의 상자가 똑같음). 그래서 '낮춘판'은 0.05·0.1 처럼 0 이 아닌 값으로.
       설정마다 모델을 따로 만들어 둠. GPU 판 Paddle 이면 GPU 로, 아니면 CPU 로 (oneDNN 오류 대비해 enable_mkldnn=False 먼저 시도)."""
    key = (thresh, box_thresh) if model_dir is None and unclip is None else (thresh, box_thresh, str(model_dir), unclip)
    if key not in _PD:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        import paddle
        from paddleocr import TextDetection
        dev = "gpu:0" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
        th = {} if thresh is None else {"thresh": thresh, "box_thresh": box_thresh}
        if model_dir is not None:
            th.update(model_name="PP-OCRv6_medium_det", model_dir=str(model_dir))
        if unclip is not None:
            th["unclip_ratio"] = unclip
        for kw in ({"enable_mkldnn": False, "device": dev}, {"device": dev}, {}):
            try:
                _PD[key] = TextDetection(**th, **kw)
                _PD["dev"] = kw.get("device", "default")
                break
            except (TypeError, ValueError):
                continue
    res = list(_PD[key].predict(img))[0]
    return polys_to_rects(res["dt_polys"], *img.shape[:2])


def paddle_t0_boxes(img):
    """1라운드 'paddle_t0' (thresh·box_thresh 에 0.0 을 넣은 것). 위 ⚠ 때문에 실제로는 기본 설정과 같은 상자.
       결과를 그대로 재현하려고 이름과 넣는 값은 바꾸지 않음."""
    return paddle_boxes(img, 0.0, 0.0)
