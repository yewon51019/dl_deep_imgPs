# =====================================================================
#  common_cns.py  —  정신계 약 실험(exp_cns) 공통 기능
# =====================================================================
#
#  [데이터]  AI Hub 경구약제 '크롭' 이미지 (알약 박스(bbox)에 딱 맞게 잘린 PNG + 라벨 JSON)
#     - 학습용  trainset\...\약제각인인식_학습셋_crop   : 4,836장, 364종 (평가용과 약이 하나도 안 겹침)
#     - 평가용  pill_image_testset\...\테스트셋_리뉴얼_crop : 432장, 36종 (맨 마지막에 한 번만 점수용으로 씀)
#  [이 파일이 하는 일]
#     경로 상수 / 라벨 JSON → 사진 한 장당 한 줄 표 / 정답 각인 정리(norm_label) /
#     학습용을 약 단위로 나누기 / 한글 경로 읽기·저장 / 227×227 맞추기
#  [주의]  norm_label 은 '정답 라벨'에만 쓴다. OCR 이 읽은 글자에는 쓰지 않는다
#          (OCR 이 아무것도 못 읽은 것은 '빈 결과'로 따로 센다 — 정답 종류와 섞지 않음).
# =====================================================================

import csv
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent                          # D:\dl_deep_imgps
sys.path.insert(0, str(ROOT / "exp300"))
from run_exp300 import imread_any                                       # 한글 경로도 읽는 기존 함수 재사용 (noqa: E402)

TRAIN_ROOT = ROOT / "trainset" / "약제각인인식_학습셋_crop" / "약제각인인식_학습셋_crop"
EVAL_ROOT = ROOT / "pill_image_testset" / "약제각인인식_테스트셋_리뉴얼_crop" / "약제각인인식_테스트셋_리뉴얼_crop"
RESULTS = ROOT / "exp_cns" / "results"

# 라벨에서 '글자가 아닌 것'을 뜻하는 한글 단어. 긴 것부터 지워야 '십자분할선'이 '십자' + '분할선'으로 남지 않음
MARK_WORDS = ("십자분할선", "분할선", "마크")


def norm_label(text):
    """정답 각인 정리 → (정리한 정답, 정답 종류).
       - 정리: '마크'·'분할선' 단어를 지우고, 대문자로 바꾼 뒤 영문·숫자만 남김 (공백·기호 '-', '.', 'Λ' 등도 지워짐)
       - 종류: '보통'(영문·숫자 있음) / '마크뿐'(마크·분할선 단어만) / '한글 포함'(그 밖의 한글이 섞임) /
               '빈 라벨'(각인 칸이 비어 있음) / '기호뿐'(영문·숫자 없이 기호만)
       ※ 라벨만 보고 정하므로 OCR 결과와 무관하다."""
    if not text or not str(text).strip():
        return "", "빈 라벨"
    t = str(text)
    had_mark = any(w in t for w in MARK_WORDS)
    for w in MARK_WORDS:
        t = t.replace(w, " ")
    has_hangul = bool(re.search(r"[가-힣]", t))                         # 마크 단어를 지우고도 남은 한글 = 진짜 한글 각인
    norm = re.sub(r"[^A-Z0-9]", "", t.upper())                          # 대문자·영문·숫자만 (한글·기호·공백 제거)
    if has_hangul:
        return norm, "한글 포함"
    if not norm:
        return "", "마크뿐" if had_mark else "기호뿐"
    return norm, "보통"


def load_items(root):
    """라벨 JSON 폴더 → 사진 한 장당 dict 목록 (파일명 순서로 정렬).
       rot(회전각)은 파일명 7번째 칸(예: K-000069_0_0_1_2_75_000_200 → '000'). 정답 정보라서 학습·분석에만 쓰고 예측에는 안 씀."""
    items = []
    for p in sorted(glob.glob(str(Path(root) / "labels" / "*.json"))):
        im = json.load(open(p, encoding="utf-8"))["images"][0]
        face = im["drug_dir"]                                           # '앞면' / '뒷면'
        own = im["print_front"] if face == "앞면" else im["print_back"]   # 이 사진 면의 각인 (라벨 기준)
        other = im["print_back"] if face == "앞면" else im["print_front"]
        truth, kind = norm_label(own)
        items.append({
            "file_name": im["file_name"],
            "image_path": str(Path(root) / "images" / im["file_name"]),
            "drug_N": im["drug_N"],
            "dl_name": im["dl_name"],
            "face": face,
            "camera_la": int(im["camera_la"]),
            "rot": im["file_name"].split("_")[6],
            "back_color": im["back_color"],
            "color1": im["color_class1"],
            "color2": im["color_class2"],
            "form": im["dl_custom_shape"],                              # 제형 (예: 경질캡슐제, 연질캡슐제)
            "shape": im["drug_shape"],                                  # 모양 (예: 원형, 장방형)
            "truth_raw": own or "",
            "truth": truth,
            "truth_kind": kind,
            # 반대 면이 '마크'·한글이면 라벨 앞뒤가 뒤바뀌었을 때 사진에 글자가 없을 수 있음 → 틀렸을 때 사람이 그림으로 확인
            "other_face_mark": int(bool(re.search(r"[가-힣]", other or ""))),
        })
    return items


def write_items(items, path):
    """사진 목록을 CSV 로 저장 (점수표의 기준이 되는 '정답표')."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)


def split_train(items, dev_drugs=36, seed=42):
    """학습용을 '약 단위'로 학습/개발로 나눔 → {파일명: 'train' 또는 'dev'}.
       같은 약(= 같은 알약의 회전·카메라만 다른 거의 같은 사진들)은 반드시 한쪽에만 들어가 누수가 없음."""
    drugs = sorted({it["drug_N"] for it in items})
    dev = set(random.Random(seed).sample(drugs, dev_drugs))
    return {it["file_name"]: ("dev" if it["drug_N"] in dev else "train") for it in items}


def imread(path):
    """한글 경로 읽기 (BGR 컬러)."""
    return imread_any(path)


def imwrite(path, img):
    """한글 경로 저장. cv2.imwrite 는 한글 경로에서 실패하므로 메모리에서 PNG 로 압축한 뒤 파일로 씀."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".png", img)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    buf.tofile(str(path))


def fit_side(img, side=227, fill=None, interp_mask=False):
    """비율을 유지해 긴 변을 side 로 맞추고, 남는 곳을 fill 로 채워 side×side 정사각형으로 만듦 (논문 입력 227×227).
       - 줄일 때는 INTER_AREA(모아서 평균, 계단 현상 적음), 키울 때는 INTER_CUBIC(부드럽게)
       - 마스크(interp_mask=True)는 0/255 값이 섞이지 않게 INTER_NEAREST
       - fill 이 없으면 이미지 네 모서리 픽셀의 중앙값 (= 배경과 비슷한 색, 경계가 덜 생김)"""
    h, w = img.shape[:2]
    s = side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_NEAREST if interp_mask else (cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    if fill is None:
        corners = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
        fill = np.median(corners, axis=0)
    out = np.empty((side, side) + img.shape[2:], img.dtype)
    out[...] = np.asarray(fill, dtype=img.dtype)
    y, x = (side - nh) // 2, (side - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out
