# =====================================================================
#  ocr_cns.py  —  2단계 OCR 1라운드: 전처리별 순수 텍스트 정답률 (보정 없음)
# =====================================================================
#
#  [하는 일]
#    사진 → 전처리(원본 400 맞춤 / final 폴더의 설정) → 30°씩 12방향으로 돌림
#         → 방향마다 글자 찾기(CRAFT 또는 PaddleOCR t0) → 박스마다 PARSeq 로 읽기 → 위→아래, 왼→오른 순으로 이어 붙임
#         → 12방향 중 '확신도(박스별 PARSeq 확신도 평균)가 가장 높은 방향' 하나를 답으로 → 정답과 비교
#  [채점]  정답 = 찍힌 면의 라벨 각인(앞면 print_front / 뒷면 print_back)에서 대문자·영숫자만 남긴 것 (common_cns.norm_label)
#          주 지표 = 완전일치 정답률. 보조 = 글자 오류율(CER), 제형·카메라별, '정답에 없는 O·0 이 끼어든' 사진 수(테두리 오인식 확인)
#          DB 보정·부분일치·헷갈리는 글자 통일은 하지 않음
#  [사진]  --split dev  : 학습용 364종 중 약 단위로 떼어 낸 36종 468장 (common_cns.split_train, 시드 42) — 고르는 데 씀
#          --split test : 평가용 432장 — 마지막에 한 번만
#          final 폴더(prep_v2\final)의 전처리는 평가용에만 저장돼 있으므로, 개발용은 같은 params.json 으로 새로 만들어
#          prep_v2\final_dev\<같은 이름>\ 에 저장해 두고 다시 씀 (한 장당 처리 시간도 prep_ms.csv 에 같이 저장)
#  [모델]  글자 찾기: craft = EasyOCR 의 CRAFT 기본 설정 / paddle_t0 = PaddleOCR 검출, 검출 기준 thresh·box_thresh 0.0 (사전 기록한 대표 설정)
#          읽기: PARSeq (사전학습 그대로). 박스를 하나도 못 찾으면 사진 전체를 한 줄로 읽음 (fallback 표시)
#          입력은 컬러 그대로 (전처리 결과가 컬러이므로). Windows 에서 torch 를 Paddle 보다 먼저 import 해야 함
#          paddle_t0 는 결과 폴더에 boxes.json (paddle_boxes_cns.py 가 Paddle GPU 환경에서 미리 계산)이 있으면 그 상자를 씀
#          (이 환경의 Paddle 은 CPU 판이라 12방향에 한 장 약 20초, GPU 판은 약 0.5~0.7초)
#  [공통]  사진 고르기·전처리·돌리기·Paddle 상자는 ocr_common_cns.py (Paddle 전용 환경과 같은 코드)
#  [저장]  exp_cns\results\ocr_v1\<split>\<det>\<prep>\pred.csv  (사진마다 한 줄)
#                                                     \dt\<파일명>.png (원본 | 고른 방향 + 박스·읽은 글자 | 정답·결과)
#  [방향 수]  --angles 36 → 10° 간격 36방향, 결과 폴더 '<prep>_a36' (paddle_t0 는 같은 이름 폴더의 boxes.json 을 씀).
#             열 이름 det_ms_12dir·rec_ms_12dir 은 요약 코드와 맞추려고 그대로 두었고, 값은 '모든 방향 합'
#  [축 정렬]  --align long → 긴 알약(align_long_cns v3: 비율 > 1.3 이고 덮은 비율 ≥ 0.80)은 축 각도·+180° 두 방향만 읽고
#             나머지는 12방향. 결과 폴더 '<prep>_alignlong', pred.csv 에 aligned·axis_ratio·axis_cover 열 추가
#  [다시 찾기]  --retry-box-prep <전처리 이름> → 고른 방향이 박스 0개(fallback, 사진 통째로 읽음)인 사진만
#             같은 --prep 사진을 '그 전처리 사진에서 찾은 저장된 상자'로 모든 방향 다시 읽고, 그 결과를 답으로 씀.
#             (원본에서 못 찾은 흐린 각인을 각인을 진하게 만든 전처리(K0L17R6)에서는 찾는 경우가 있어서)
#             상자는 paddle_t0\<전처리>_alignlong\boxes.json (Paddle 재실행 없음). 결과 폴더 끝 '_retry<앞 7글자>', pred.csv 에 retry 열
#  [실행 예]
#    python exp_cns\ocr_cns.py --split dev --det craft --prep orig --limit 20
#    python exp_cns\ocr_cns.py --split dev --det paddle_t0 --prep orig --align long
#    python exp_cns\ocr_cns.py --split dev --det paddle_t0 --prep orig --align long --join lr --suffix _lr
#      (상자는 '<prep>_alignlong\boxes.json' 을 그대로 읽고, 결과는 '<prep>_alignlong_lr' 에 저장)
#    python exp_cns\ocr_cns.py --det dbpp --thresh 0.1 --prep orig --align long --join line --hard
#      (검출기 비교: 잘 못 찾은 사진만, 상자는 det2x_boxes_cns.py 가 만든 'dbpp_th0.1\orig_alignlong_line_hard\boxes.json')
#    python exp_cns\ocr_cns.py --split dev --det paddle_t0 --prep orig --angles 36
#    python exp_cns\ocr_cns.py --split dev --det paddle_t0 --prep K0L17-c3-Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12_auto_link17_low85
#    python exp_cns\ocr_cns.py --split dev --det paddle_t0 --prep orig --align long --join line --suffix _line ^
#        --retry-box-prep K0L17R6-q2-Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12_auto_link17_low85_rim0.6
#      (결과는 'orig_alignlong_line_retryK0L17R6')
# =====================================================================

import argparse
import csv
import json
import os
import time

import torch  # noqa: F401  (Paddle 보다 먼저 import — Windows 에서 순서가 바뀌면 torch DLL 오류)
import cv2
import numpy as np

import common_cns as cc
import align_long_cns as al                   # 긴 알약 축 정렬 (--align long)
import ocr_common_cns as oc                   # 사진 고르기·전처리·돌리기·Paddle 상자 (Paddle 전용 환경과 같이 씀, torch 없음)
import run_exp300 as base                     # norm, join_boxes, easy_reader, craft_boxes, PARSeq (기존 파이프라인과 같은 함수)

ANGLES = oc.ANGLES
rotate_color = oc.rotate_color
# 저장된 상자(det2x_boxes_cns.py 가 Paddle 환경에서 계산한 boxes.json)로만 돌리는 검출기 — 이 환경에서 직접 검출하지 않음
SAVED_ONLY = ["paddle_low", "dbpp", "pse"]
# 파인튜닝 검출기 (paddle_boxes_cns.py --name <이름> --det-model ... 가 저장한 boxes.json 만 씀, 검출 기준은 기본 0.2/0.45 라 --thresh 없음)
SAVED_FT = ["paddle_ft1", "paddle_ft1_u2.0", "paddle_ft2e13_u2.0", "paddle_ft2e50_u2.0",    # ft2: 13 에폭(검증 최고) · 50 에폭(마지막)
            "paddle_ft3_u2.0"]                                                             # ft3: K0L17 사진으로 학습한 검출기


class SavedBoxes:
    """paddle_boxes_cns.py (Paddle GPU 환경)가 저장한 boxes.json 을 읽어 (파일명, 각도) → 상자·검출 시간을 돌려줌.
       파일이 없으면 None → 이 환경에서 Paddle(CPU 판)을 직접 돌림 (한 장 12방향 약 20초)."""

    def __init__(self, path):
        self.d = json.load(open(path, encoding="utf-8")) if path.exists() else None

    def get(self, fn, ang):
        r = self.d[fn][str(ang)]
        return [tuple(b) for b in r["boxes"]], r["det_ms"]


def detect(det, img):
    """글자 상자 찾기. craft 는 EasyOCR 이 RGB 를 기대하므로 바꿔서 넣음, Paddle 은 BGR 그대로."""
    if det == "craft":
        return base.craft_boxes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return oc.paddle_t0_boxes(img)


PAD_RATIO = 2.0                                                      # --pad: 가로/세로가 이보다 작은 조각만 양옆을 채움


class Reader:
    """PARSeq 읽기 (run_exp300.PARSeq 모델·입력 변환을 그대로 쓰고, 컬러 조각을 받도록만 바꿈)."""

    def __init__(self, weights=None, pad=False):
        self.ps = base.PARSeq()
        self.pad = pad                                               # True = 정사각형에 가까운 조각은 옆으로 늘리지 않고 양옆을 채움 (pad_crop)
        if weights:                                                  # 파인튜닝 가중치 (parseq_ft_cns.py 의 best.pt)
            self.ps.m.load_state_dict(torch.load(weights, map_location=self.ps.dev))
            self.ps.m.eval()

    def pad_crop(self, crop_bgr):
        """가로/세로 < PAD_RATIO 인 조각 → 비율 그대로 입력 높이(32)에 맞추고 양옆을 조각의 중간 색으로 채운 128×32.
           [왜] 기본 입력 변환은 비율을 무시하고 128×32 로 늘려서, 215×190 상자는 세로보다 가로가 약 3.5배 더 늘어남.
           그보다 넓은 조각은 지금과 똑같이 둠 (이 함수는 그대로 돌려줌, 뒤의 입력 변환이 늘림)."""
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0 or w / h >= PAD_RATIO:
            return crop_bgr
        H, W = self.ps.m.hparams.img_size
        nw = max(1, min(W, round(w * H / h)))
        small = cv2.resize(crop_bgr, (nw, H), interpolation=cv2.INTER_CUBIC)
        fill = np.median(crop_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
        canvas = np.full((H, W, 3), fill, np.uint8)
        x0 = (W - nw) // 2                                           # 가운데에 놓음
        canvas[:, x0:x0 + nw] = small
        return canvas

    def read(self, crop_bgr):
        """박스 조각 하나 → (글자, 확신도 0~1). 확신도 = 글자별 확률 평균 (기존과 같음)."""
        from PIL import Image
        m = self.ps
        if self.pad:
            crop_bgr = self.pad_crop(crop_bgr)
        x = m.tf(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(m.dev)
        with m.torch.no_grad():
            p = m.m(x).softmax(-1)
        labels, confs = m.m.tokenizer.decode(p)
        c = confs[0]
        return labels[0], float(c.mean()) if len(c) else 0.0


def join_lr(items):
    """상자 글자를 왼→오른(상자 왼쪽 x 순서)으로만 이어 붙임. items = [(x, y, 글자, 확신도0~1)] → (글자, 확신도 평균 × 100).
       base.join_boxes 는 '위→아래, 그다음 왼→오른' 인데, 줄 간격 기준이 (가장 높은 y − 가장 낮은 y) / 3 이라
       상자가 2개면 높이가 10px 남짓만 달라도 다른 줄로 봐서 높은 상자를 먼저 붙임
       → 가로로 눕힌 긴 알약에서도 CHO-A | SPA 가 'SPACHOA' 로 붙는 문제. 축 정렬된 사진은 글자 줄이 하나라고 보고 x 순서만 씀."""
    if not items:
        return "", 0.0
    items = sorted(items, key=lambda it: it[0])
    return "".join(t for _, _, t, _ in items), float(np.mean([c for *_, c in items]) * 100)


def join_line(items):
    """상자를 '세로로 겹치는 정도'로 줄로 묶어 붙임. items = [(x0, y0, x1, y1, 글자, 확신도0~1)] → (글자, 확신도 평균 × 100).
       - 위(가운데 y)부터 차례로 보며, 지금 줄의 세로 범위와 '두 높이 중 작은 쪽의 절반 이상' 겹치면 같은 줄, 아니면 새 줄
       - 줄 안은 왼→오른, 줄끼리는 위→아래
       join_lr(왼→오른만)은 한 줄 분할 각인(CHO-A | SPA)은 고치지만 진짜 두 줄 각인(NVR / 25MG, HANMI / 04)을 망쳐서 (개발용 +6 / −13)
       두 경우를 모두 맞게 하려는 규칙."""
    if not items:
        return "", 0.0
    lines = []                                                       # [줄 위 y, 줄 아래 y, [상자들]]
    for it in sorted(items, key=lambda it: (it[1] + it[3]) / 2):
        y0, y1 = it[1], it[3]
        if lines:
            top, bot, mem = lines[-1]
            overlap = min(bot, y1) - max(top, y0)
            if overlap >= 0.5 * min(y1 - y0, bot - top):
                lines[-1] = [min(top, y0), max(bot, y1), mem + [it]]
                continue
        lines.append([y0, y1, [it]])
    order = [it for _, _, mem in lines for it in sorted(mem, key=lambda it: it[0])]
    return "".join(it[4] for it in order), float(np.mean([it[5] for it in order]) * 100)


def read_direction(img, det, reader, saved=None, join="tb"):
    """한 방향 사진 → dict(text, conf, boxes, texts, fallback, det_ms, rec_ms).
       text = 박스 글자를 위→아래, 왼→오른 순으로 이어 붙인 것 (base.join_boxes), conf = 박스 확신도 평균 × 100.
       join='lr' 이면 왼→오른 순서로만 붙임 (join_lr, 축 정렬된 긴 알약용).
       saved = (상자 목록, 검출 ms) 가 주어지면 검출을 다시 하지 않고 그 상자를 씀 (Paddle GPU 환경에서 미리 계산한 것)."""
    t0 = time.perf_counter()
    if saved is None:
        boxes = detect(det, img)
        det_ms = (time.perf_counter() - t0) * 1000
    else:
        boxes, det_ms = saved
    t1 = time.perf_counter()
    fallback = not boxes
    if fallback:                                               # 박스가 없으면 사진 전체를 한 줄로
        boxes = [(0, 0, img.shape[1], img.shape[0])]
    items, texts, full = [], [], []
    for x0, y0, x1, y1 in boxes:
        t, c = reader.read(img[y0:y1, x0:x1])
        t = base.norm(t)
        texts.append(t)
        if t:
            items.append((x0, y0, t, c))
            full.append((x0, y0, x1, y1, t, c))                      # join_line 은 상자 아래 y 도 필요
    if join == "lr":
        text, conf = join_lr(items)
    elif join == "line":
        text, conf = join_line(full)
    else:
        text, conf = base.join_boxes(items)
    t2 = time.perf_counter()
    return dict(text=base.norm(text), conf=conf, boxes=boxes, texts=texts, fallback=fallback,
                det_ms=det_ms, rec_ms=(t2 - t1) * 1000)


def pick_best(dirs):
    """12방향 중 확신도가 가장 높은 방향 (글자가 하나도 안 나온 방향은 뒤로). 전부 빈 결과면 0° 의 빈 결과."""
    return max(dirs, key=lambda d: (bool(d["text"]), d["conf"]))


class OrientPicker:
    """--pick orient: 확신도 최고 방향 d 와 반대 방향(d + 180°) 중, 상자 조각의 '똑바름 확률' 평균이 높은 쪽.
       확률 = orient_crop_cns 네 칸 판별기(orient_crop_v2)의 칸 0 (PARSeq 확신도는 위아래를 못 가름, CLAUDE.md 10장 3-2)."""

    def __init__(self, weights):
        import orient_crop_cns as oc4
        oc4.NC = 4
        self.oc4 = oc4
        self.m = oc4.make_model(pretrained=False)
        self.m.load_state_dict(torch.load(weights, map_location=oc4.DEV))
        self.m.eval()

    def score(self, img, d):
        if d["fallback"] or not d["text"]:
            return -1.0
        rot = rotate_color(img, d["ang"])
        crops = [rot[y0:y1, x0:x1] for (x0, y0, x1, y1), t in zip(d["boxes"], d["texts"]) if t and y1 > y0 and x1 > x0]
        if not crops:
            return -1.0
        return float(np.mean(self.oc4.p_up(self.m, crops)))

    def pick(self, img, dirs, best):
        """→ (고른 방향, 바꿨는지, 고른 쪽 점수, 반대쪽 점수)."""
        opp = [d for d in dirs if abs(((d["ang"] - best["ang"]) % 360) - 180) < 1e-6]
        if not opp:
            return best, False, None, None
        s0, s1 = self.score(img, best), self.score(img, opp[0])
        return (opp[0], True, s1, s0) if s1 > s0 else (best, False, s0, s1)


def read_all(img, fn, angs, det, reader, saved, join):
    """사진 하나를 angs 의 모든 방향으로 읽음 → 방향별 결과 목록. saved 에 상자가 있으면 그 상자를 씀 (없으면 직접 검출)."""
    dirs = []
    for ang in angs:
        r = read_direction(rotate_color(img, ang), det, reader,
                           saved.get(fn, ang) if saved.d is not None else None, join=join)
        r["ang"] = ang
        dirs.append(r)
    return dirs


# ---------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------
def edit_distance(a, b):
    """편집거리 (글자 넣기·빼기·바꾸기 횟수)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extra_o0(truth, read):
    """정답보다 O·0 이 더 많이 나왔는지 (둥근 테두리를 'O'·'0' 으로 읽은 경우를 세려는 보조 표시)."""
    return (read.count("O") + read.count("0")) > (truth.count("O") + truth.count("0"))


# ---------------------------------------------------------------------
# dt 그림: 원본 | 고른 방향 + 박스·박스별 글자 | 정답·결과 (OpenCV 글씨는 영문만)
# ---------------------------------------------------------------------
def draw_dt(orig400, best, it, ok, prep_ms, retry=None):
    """retry = None 이면 다시 찾기 옵션 없음, 아니면 (전처리 앞 7글자, 다시 찾았는지) → 오른쪽 칸 맨 아래에 한 줄 더."""
    img = best["img"].copy()
    for (x0, y0, x1, y1), t in zip(best["boxes"], best["texts"]):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255) if best["fallback"] else (0, 200, 0), 2)
        cv2.putText(img, t or "-", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    s = 400 / max(img.shape[:2])                                     # 돌리면 캔버스가 커지므로 400 안에 맞춤
    img = cc.fit_side(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 400)
    panel = np.full((400, 360, 3), 255, np.uint8)
    pm = f"{prep_ms:.0f}" if prep_ms != "" else "-"
    lines = [(f"{it['drug_N']}  cam{it['camera_la']}", (0, 0, 0)),
             (f"truth: {it['truth']}", (0, 0, 0)),
             (f"read : {best['text'] or '(empty)'}", (0, 0, 0)),
             ("O  correct" if ok else "X  wrong", (0, 160, 0) if ok else (0, 0, 220)),
             (f"dir {best['ang']} deg / conf {best['conf']:.1f}", (60, 60, 60)),
             (f"boxes {0 if best['fallback'] else len(best['boxes'])}  fallback {'yes' if best['fallback'] else 'no'}", (60, 60, 60)),
             (f"ms: prep {pm}", (60, 60, 60)),
             (f"    det {best['det_total']:.0f} rec {best['rec_total']:.0f} ({best['n_dirs']} dirs)", (60, 60, 60))]
    if retry is not None:
        lines.append((f"retry box {retry[0]} {'yes' if retry[1] else 'no'}", (200, 100, 0) if retry[1] else (60, 60, 60)))
    for i, (t, col) in enumerate(lines):
        cv2.putText(panel, t, (10, 36 + i * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if i != 3 else 0.9, col, 2 if i == 3 else 1)
    return np.hstack([orig400, img, panel])


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train500", "trainall", "trainval", "exp300test", "hardset", "valdev"] + list(oc.CROP51),
                    help="train500 = 검출기 파인튜닝 라벨용 학습 500장 (make_det_labels_cns.py)")
    ap.add_argument("--det", default="craft", choices=["craft", "paddle_t0"] + SAVED_ONLY + SAVED_FT)
    ap.add_argument("--prep", default="orig", help="orig 또는 prep_v2\\final 안의 폴더 이름")
    ap.add_argument("--limit", type=int, default=0, help="시험 운전: 고르게 N 장만")
    ap.add_argument("--no-dt", action="store_true", help="dt 그림 저장 안 함")
    ap.add_argument("--suffix", default="", help="결과 폴더 이름 뒤에 붙일 말 (재확인 실행이 기존 결과를 덮어쓰지 않게)")
    ap.add_argument("--angles", type=int, default=12, choices=[12, 36], help="돌리는 방향 수 (12 = 30° 간격, 36 = 10° 간격)")
    ap.add_argument("--align", default="none", choices=["none", "long"],
                    help="long = 긴 알약은 축 각도·+180° 두 방향만 (align_long_cns v3), 나머지는 12방향")
    ap.add_argument("--join", default="tb", choices=["tb", "lr", "line"],
                    help="tb = 위→아래·왼→오른 (기존), lr = 왼→오른만, line = 세로로 겹치는 상자끼리 한 줄 "
                         "(lr·line 은 축 정렬된 긴 알약에만, 나머지는 tb 그대로)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="paddle_low·dbpp·pse 의 '낮춘 기준' (0.1 / 0.05). 결과 폴더 '<det>_th<기준>'")
    ap.add_argument("--hard", action="store_true",
                    help="'박스를 잘 못 찾은 사진'만 (ocr_common_cns.hard_names). 축 정렬 + 줄 묶기와 같이 씀, 폴더 '..._alignlong_line_hard'")
    ap.add_argument("--box-prep", default=None,
                    help="상자는 이 전처리 사진에서 찾은 것(저장된 boxes.json)을 쓰고, 읽기는 --prep 사진으로. "
                         "두 전처리 모두 같은 400 정사각형 좌표·같은 돌리기라 상자 위치를 그대로 옮겨 쓸 수 있음. 폴더 끝 '_box<앞 7글자>'")
    ap.add_argument("--retry-box-prep", default=None,
                    help="고른 방향이 박스 0개(fallback)인 사진만 이 전처리의 저장된 상자로 다시 읽음 "
                         "(paddle_t0 + --align long 과만). 폴더 끝 '_retry<앞 7글자>' (--suffix 뒤에 붙음)")
    ap.add_argument("--parseq", default=None, help="파인튜닝 PARSeq 가중치 (parseq_ft_cns.py best.pt). 폴더 이름은 --suffix 로 구분")
    ap.add_argument("--pad", action="store_true",
                    help="가로/세로 < 2 인 조각은 늘리지 않고 양옆을 채워 읽음 (Reader.pad_crop). 폴더 이름은 --suffix 로 구분")
    ap.add_argument("--pick", default="conf", choices=["conf", "orient"],
                    help="conf = 확신도 최고 방향 (기존), orient = 그 방향과 +180° 중 위아래 판별기 점수 높은 쪽 (폴더 끝 '_porient')")
    ap.add_argument("--orient-model", default=str(cc.RESULTS / "orient_crop_v2" / "model" / "best.pt"))
    a = ap.parse_args()
    if a.retry_box_prep and not (a.det == "paddle_t0" and a.align == "long" and not a.box_prep):
        ap.error("--retry-box-prep 은 --det paddle_t0 --align long 과만 같이 씀 (--box-prep 과는 같이 안 씀)")
    if a.box_prep and a.det == "craft":
        ap.error("--box-prep 은 저장된 상자가 있는 검출기(paddle_t0·paddle_low·dbpp·pse·파인튜닝)만")
    if a.join != "tb" and a.align != "long":
        ap.error("--join lr/line 은 --align long 과 같이 씀")
    if a.align == "long" and a.angles != 12:
        ap.error("--align long 은 12방향과만 같이 씀")
    if a.hard and not (a.align == "long" and a.join == "line"):
        ap.error("--hard 는 --align long --join line 과 같이 씀")
    if a.det in SAVED_ONLY and a.thresh is None:
        ap.error(f"--det {a.det} 는 --thresh 가 필요함")
    if a.split in ("test", "exp300test"):
        print("[주의] 평가용(test)입니다. 규칙상 마지막에 한 번만 돌립니다.")

    if a.hard:                                                          # 검출기 비교용: 잘 못 찾은 사진만
        hard = oc.hard_names(a.split, a.prep)
        items = [it for it in oc.load_split(a.split, 0) if it["file_name"] in hard]
        if a.limit:
            items = items[::max(1, len(items) // a.limit)][:a.limit]
    else:
        items = oc.load_split(a.split, a.limit)
    prep = oc.Prep(a.prep, a.split)
    reader = Reader(a.parseq, a.pad)
    picker = OrientPicker(a.orient_model) if a.pick == "orient" else None
    det_name = f"{a.det}_th{a.thresh:g}" if a.det in SAVED_ONLY else a.det
    out = oc.out_dir(a.split, det_name, a.prep, a.limit, a.angles, a.align,   # 시험 운전 '__tryN', 36방향 '_a36', 축 정렬 '_alignlong'
                     "_line_hard" if a.hard else "")
    box_out = out if a.split != "trainval" else oc.out_dir("trainall", det_name, a.prep, 0, a.angles, a.align)   # trainval 은 trainall 상자
    saved = SavedBoxes(box_out / "boxes.json") if a.det != "craft" else SavedBoxes(out / "_none_")
    if a.box_prep:
        # 상자 찾기 전처리를 따로: paddle_t0 는 그 전처리의 468장 전체 상자, 나머지는 '--prep 기준으로 고른 사진'에 계산한 상자
        # (det2x_boxes_cns.py --prep <box_prep> --hard-prep <prep> 가 만든 '..._line_hard_<prep 앞 7글자>')
        # 파인튜닝 검출기도 paddle_boxes_cns.py --prep <box_prep> --align long --name <이름> 이 468장 전체에 저장한 상자
        if a.det == "paddle_t0" or a.det in SAVED_FT:
            box_dir = oc.out_dir(a.split, a.det, a.box_prep, 0, a.angles, a.align)
        else:
            box_dir = oc.out_dir(a.split, det_name, a.box_prep, 0, 12, "long", f"_line_hard_{a.prep[:7]}")
        saved = SavedBoxes(box_dir / "boxes.json")
        if saved.d is None:
            ap.error(f"{box_dir / 'boxes.json'} 가 없음")
        out = out.with_name(out.name + f"_box{a.box_prep[:7]}")
    if a.det in SAVED_ONLY and saved.d is None:
        ap.error(f"{out / 'boxes.json'} 가 없음 — 먼저 Paddle 환경에서 det2x_boxes_cns.py 로 상자를 계산")
    if a.det in SAVED_FT and saved.d is None:                           # 없으면 기본 Paddle 로 돌아가 버리므로 멈춤
        ap.error(f"{out / 'boxes.json'} 가 없음 — 먼저 Paddle 환경에서 paddle_boxes_cns.py --name {a.det} --det-model ... 로 계산")
    if a.suffix:
        out = out.with_name(out.name + a.suffix)
    if a.pick == "orient":
        out = out.with_name(out.name + "_porient")
    retry_saved, retry_tag = None, None
    if a.retry_box_prep:
        # 다시 찾기용 상자: 그 전처리 사진 468장 × 축 정렬 방향으로 미리 계산한 Paddle 상자 (paddle_boxes_cns.py)
        rdir = oc.out_dir(a.split, "paddle_t0", a.retry_box_prep, 0, 12, "long")
        retry_saved = SavedBoxes(rdir / "boxes.json")
        if retry_saved.d is None:
            ap.error(f"{rdir / 'boxes.json'} 가 없음")
        retry_tag = a.retry_box_prep[:7]
        out = out.with_name(out.name + f"_retry{retry_tag}")
    os.makedirs(out / "dt", exist_ok=True)
    print(f"[{a.split}] {len(items)}장, 글자 찾기 {a.det}"
          + (" (저장된 상자 사용)" if saved.d is not None else "") + f", 전처리 {a.prep} → {out}")

    rows = []
    for k, it in enumerate(items, 1):
        bgr = cc.imread(it["image_path"])
        img, prep_ms = prep(it, bgr)
        if a.align == "long":                                           # 긴 알약이면 축 각도·+180° 두 방향만, 아니면 12방향
            angs, axis = al.align_angles(bgr)
            aligned = len(angs) == 2
        else:
            angs, axis, aligned = oc.angles(a.angles), None, False
        join = a.join if aligned else "tb"
        dirs = read_all(img, it["file_name"], angs, a.det, reader, saved, join)
        best = pick_best(dirs)
        flip = None
        if picker is not None:
            best, switched, s_best, s_opp = picker.pick(img, dirs, best)
            flip = (int(switched), "" if s_best is None else round(s_best, 4), "" if s_opp is None else round(s_opp, 4))
        used = dirs                                                     # 시간 합계에 넣을 방향들
        retried = False
        if retry_saved is not None and best["fallback"]:
            # 고른 방향이 박스 0개 → 같은 사진을 다른 전처리에서 찾은 상자로 모든 방향 다시 읽고, 그 결과를 답으로
            # (다시 읽어도 박스 0개일 수 있음 — 그래도 다시 읽은 결과를 씀. 61.1% 계산 때와 같은 규칙)
            dirs2 = read_all(img, it["file_name"], angs, a.det, reader, retry_saved, join)
            best, retried = pick_best(dirs2), True
            used = dirs + dirs2
        best["img"] = rotate_color(img, best["ang"])
        best["det_total"] = sum(d["det_ms"] for d in used)
        best["rec_total"] = sum(d["rec_ms"] for d in used)
        best["n_dirs"] = len(used)
        truth = it["truth"]
        ok = best["text"] == truth
        cer = edit_distance(best["text"], truth) / max(1, len(truth))
        rows.append(dict(file_name=it["file_name"], drug_N=it["drug_N"], form=it["form"], camera_la=it["camera_la"],
                         face=it["face"], truth=truth, read=best["text"], correct=int(ok), cer=round(cer, 4),
                         angle=best["ang"], conf=round(best["conf"], 2),
                         n_boxes=0 if best["fallback"] else len(best["boxes"]), fallback=int(best["fallback"]),
                         extra_o0=int(extra_o0(truth, best["text"])),
                         prep_ms=round(prep_ms, 1) if prep_ms != "" else "",
                         det_ms_12dir=round(best["det_total"], 1), rec_ms_12dir=round(best["rec_total"], 1),
                         all_dirs=" | ".join(f"{d['ang']}:{d['text']}:{d['conf']:.1f}" for d in dirs),
                         # 축 정렬 정보 (--align long 일 때만 값): 긴 알약으로 정렬했는지, 긴/짧은 비율, 감싸는 상자가 사진을 덮은 비율
                         aligned=int(aligned), axis_ratio=round(axis["ratio"], 3) if axis else "",
                         axis_cover=round(axis["cover"], 3) if axis else "",
                         # 고른 방향(돌린 사진 좌표)의 상자와 상자별 글자 [[x0, y0, x1, y1, 글자], ...] — 박스 0개면 [].
                         # make_det_labels_cns.py 가 이 상자를 돌리기 전 사진 좌표로 되돌려 검출기 라벨로 씀
                         best_boxes=json.dumps([] if best["fallback"] else
                                               [[*map(int, b), t] for b, t in zip(best["boxes"], best["texts"])])))
        if flip is not None:                                            # --pick orient: 반대 방향으로 바꿨는지, 두 방향 점수
            rows[-1].update(orient_switch=flip[0], orient_best=flip[1], orient_opp=flip[2])
        if retry_saved is not None:
            rows[-1]["retry"] = int(retried)                            # 1 = 박스 0개라서 다른 전처리 상자로 다시 읽음
        if not a.no_dt:
            cc.imwrite(str(out / "dt" / it["file_name"]),
                       draw_dt(cc.fit_side(bgr, 400), best, it, ok, prep_ms,
                               (retry_tag, retried) if retry_saved is not None else None))
        if k % 20 == 0 or k == len(items):
            acc = np.mean([r["correct"] for r in rows]) * 100
            print(f"  {k}/{len(items)}  정답률 {acc:.1f}%")

    with open(out / "pred.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 요약 출력 (표 정리는 ocr_summary_cns.py)
    acc = np.mean([r["correct"] for r in rows]) * 100
    print(f"\n정답률 {acc:.1f}% ({sum(r['correct'] for r in rows)}/{len(rows)}), CER 평균 {np.mean([r['cer'] for r in rows]):.3f}, "
          f"fallback {sum(r['fallback'] for r in rows)}장, O·0 끼어듦 {sum(r['extra_o0'] for r in rows)}장")
    pm = [r["prep_ms"] for r in rows if r["prep_ms"] != ""]
    print(f"한 장당 평균 ms: 전처리 {np.mean(pm) if pm else float('nan'):.0f} / 글자 찾기 {np.mean([r['det_ms_12dir'] for r in rows]):.0f} "
          f"/ 읽기 {np.mean([r['rec_ms_12dir'] for r in rows]):.0f} ({a.angles}방향 합)")


if __name__ == "__main__":
    main()
