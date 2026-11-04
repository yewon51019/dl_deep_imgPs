# =====================================================================
#  det2x_boxes_cns.py  —  '박스를 잘 못 찾은 사진'에서 다른 글자 검출기의 상자 계산 (Paddle 전용 GPU 환경에서 실행)
# =====================================================================
#
#  [왜]  Paddle 기본 설정 + 축 정렬 + 줄 묶기 결과에서 박스를 못 찾았거나 정답의 일부만 읽은 사진이 남음.
#        검출기를 바꾸거나 기준을 낮추면 박스를 더 잘 찾는지 같은 사진들에서 비교한다. (읽기는 ocr_cns.py --hard 가 PARSeq 로)
#  [검출기]  모두 '최대한 낮춘' 기준 t (0.1, 0.05) — 글자 지도에서 t 이상을 글자 점으로, 상자 평균 점수 t 이상을 상자로
#    paddle_low : PaddleOCR 3.7.0 TextDetection (PP-OCRv6 검출), thresh = box_thresh = t
#                 (0.0 을 넣으면 내부에서 기본값으로 바뀌는 문제 → ocr_common_cns.paddle_boxes 주석)
#    dbpp       : DBNet++ (ResNet50, ICDAR2015 학습 가중치) — PaddleOCR 2.10 소스(third_party\PaddleOCR_2x)로 불러옴
#                 후처리 DBPostProcess(thresh = box_thresh = t, unclip 1.5 = 설정 파일 값)
#    pse        : PSENet (ResNet50_vd, ICDAR2015 학습 가중치) — 같은 소스
#                 원래 후처리는 C(Cython) 코드라 컴파일러가 필요 → 같은 원리를 파이썬으로 다시 짬 (pse_expand)
#                 글자 핵 기준: sigmoid 확률 t 이상 (원래 설정은 로짓 0 = 확률 0.5), 상자 평균 점수 t 이상 (원래 0.85)
#  [입력 크기]  2.x 설정 파일의 평가용 크기를 쓰되, DB++ 의 '고정 1152×2048' 은 정사각형 알약 사진을 옆으로 늘려 버려서
#               PSE 와 같은 '짧은 변 736 이상(32 배수)' 로 맞춤. 정규화는 각 설정 파일 값 그대로 (BGR 순서, 2.x 학습과 같음)
#  [방향]  align_long_cns.align_angles (축 정렬 v3): 긴 알약 = 축 각도·+180°, 나머지 = 12방향. 축은 원본 사진에서 구함
#  [사진]  ocr_common_cns.hard_names(split, prep) — 전처리마다 따로 고름
#  [저장]  exp_cns\results\ocr_v1\<split>\<det>_th<t>\<prep>_alignlong_line_hard\boxes.json  (paddle_boxes_cns 와 같은 형식)
#  [전체]  --all → 잘 못 찾은 사진만이 아니라 split 전체 (예: 개발용 468장). 폴더 '<prep>_alignlong' (paddle_t0 와 같은 이름 규칙)
#          → ocr_cns.py --det dbpp --thresh 0.1 --align long --join line --suffix _line 이 이 상자를 읽음
#  [실행 예]
#    .\.venv_paddle_gpu\Scripts\python.exe exp_cns\det2x_boxes_cns.py --det dbpp --thresh 0.1 --prep orig --limit 3
#    .\.venv_paddle_gpu\Scripts\python.exe exp_cns\det2x_boxes_cns.py --det dbpp --thresh 0.1 --prep orig --all
# =====================================================================

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import align_long_cns as al
import common_cns as cc
import ocr_common_cns as oc

P2X = cc.ROOT / "third_party" / "PaddleOCR_2x"
CFG = {"dbpp": ("configs/det/det_r50_db++_icdar15.yml", "pretrain/det_r50_db++_icdar15_train/best_accuracy"),
       "pse": ("configs/det/det_r50_vd_pse.yml", "pretrain/det_r50_vd_pse_v2.0_train/best_accuracy")}


# ---------------------------------------------------------------------
# PSE 후처리 (파이썬판)
# ---------------------------------------------------------------------
def pse_expand(kernels, min_area=16):
    """PSENet 의 '점점 넓혀 가기(progressive scale expansion)'.
       kernels: (n, H, W) 0/1 — 0번이 가장 큰 글자 영역, 마지막이 가장 작은 글자 핵.
       1) 가장 작은 핵을 4-연결 덩어리로 나눠 번호를 붙이고 (min_area 보다 작은 덩어리는 버림)
       2) 다음으로 큰 핵 안에서, 이미 번호가 붙은 점의 상하좌우 이웃으로 번호를 퍼뜨림 (더 못 퍼질 때까지 반복)
       3) 가장 큰 영역까지 반복 → 번호 지도
       원래 C 코드는 줄(queue) 순서대로 '먼저 닿은 번호' 를 주고, 여기서는 한 번에 한 칸씩 동시에 넓히며
       두 번호가 같은 칸에 동시에 닿으면 큰 번호가 이김 (3×3 십자 최대값 필터). 경계 한두 픽셀만 다를 수 있음."""
    n_lab, label = cv2.connectedComponents(kernels[-1].astype(np.uint8), connectivity=4)
    for i in range(1, n_lab):
        if (label == i).sum() < min_area:
            label[label == i] = 0
    pred = label.astype(np.float32)                                     # cv2.dilate 는 float32 지원 (번호 2^24 까지 정확)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for k in range(kernels.shape[0] - 2, -1, -1):                       # 작은 핵 → 큰 영역 순서
        allow = kernels[k] > 0
        while True:
            grown = cv2.dilate(pred, cross)
            new = (pred == 0) & allow & (grown > 0)
            if not new.any():
                break
            pred[new] = grown[new]
    return pred.astype(np.int32)


def pse_boxes(maps, shape, t, min_area=16):
    """PSENet 출력(로짓, (n, h, w)) → 원본 좌표 다각형 목록. ppocr PSEPostProcess(quad) 와 같은 순서, 기준만 t 로."""
    logit_t = float(np.log(t / (1 - t)))                                # 확률 t ↔ 로짓
    score = 1 / (1 + np.exp(-maps[0]))
    kernels = (maps > logit_t).astype(np.uint8)
    kernels = kernels * kernels[0][None]                                # 핵은 가장 큰 글자 영역 안에서만
    label = pse_expand(kernels, min_area)
    src_h, src_w, ratio_h, ratio_w = shape
    polys = []
    for i in range(1, int(label.max()) + 1):
        ind = label == i
        pts = np.array(np.where(ind)).T[:, ::-1]
        if pts.shape[0] < min_area or score[ind].mean() < t:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
        box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, src_h)
        polys.append(box)
    return polys


# ---------------------------------------------------------------------
# PaddleOCR 2.x 모델 (DB++ / PSE)
# ---------------------------------------------------------------------
class Det2x:
    """PaddleOCR 2.10 소스로 학습 가중치를 불러와 사진 한 장 → 다각형 목록."""

    def __init__(self, name, t):
        sys.path.insert(0, str(P2X))
        import paddle
        import yaml
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.utils.save_load import load_model
        # ppocr.data(사진 준비 모듈)는 학습용 증강 라이브러리(albumentations)까지 불러와서 쓰지 않고,
        # 추론에 필요한 두 단계(DetResizeForTest 'min' 방식, NormalizeImage)만 아래 prepare() 에 같은 계산으로 옮김
        self.paddle, self.name, self.t = paddle, name, t
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.set_device("gpu:0")
        cfg, w = CFG[name]
        c = yaml.safe_load(open(P2X / cfg, encoding="utf-8"))
        c["Architecture"].get("Backbone", {}).pop("pretrained", None)   # 백본 사전학습 파일은 필요 없음 (전체 가중치를 불러옴)
        self.m = build_model(c["Architecture"])
        self.m.eval()
        c["Global"]["pretrained_model"], c["Global"]["checkpoints"] = str(P2X / w), None
        load_model(c, self.m)
        norm = [op["NormalizeImage"] for op in c["Eval"]["dataset"]["transforms"] if "NormalizeImage" in op][0]
        self.mean = np.array(norm["mean"], np.float32)                  # 설정 파일의 정규화 값 그대로 (DB++ 와 PSE 가 다름)
        self.std = np.array(norm["std"], np.float32)
        if name == "dbpp":
            self.post = build_post_process({**c["PostProcess"], "thresh": t, "box_thresh": t})

    def prepare(self, img, limit=736):
        """ppocr DetResizeForTest(limit_type='min') + NormalizeImage(scale 1/255, hwc) + ToCHWImage 와 같은 계산.
           짧은 변이 limit 보다 작으면 limit 으로 키우고, 가로·세로를 32 의 배수로 반올림. 색 순서는 BGR 그대로 (2.x 학습과 같음).
           → (1×3×H×W 입력, [원래 h, 원래 w, 세로 배율, 가로 배율])"""
        h, w = img.shape[:2]
        r = limit / min(h, w) if min(h, w) < limit else 1.0
        rh, rw = max(int(round(h * r / 32) * 32), 32), max(int(round(w * r / 32) * 32), 32)
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32), np.array([h, w, rh / h, rw / w], np.float32)

    def __call__(self, img):
        image, shape = self.prepare(img)
        x = self.paddle.to_tensor(image)
        with self.paddle.no_grad():
            out = self.m(x)
        if self.name == "dbpp":
            return list(self.post(out, shape[None])[0]["points"])
        return pse_boxes(out["maps"].numpy()[0], shape, self.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--det", required=True, choices=["paddle_low", "dbpp", "pse"])
    ap.add_argument("--thresh", type=float, required=True, help="최대한 낮춘 기준 (0.1 또는 0.05)")
    ap.add_argument("--prep", default="orig", help="orig 또는 prep_v2\\final 안의 폴더 이름")
    ap.add_argument("--limit", type=int, default=0, help="시험 운전: 고르게 N 장만")
    ap.add_argument("--hard-prep", default=None,
                    help="'잘 못 찾은 사진'을 어느 전처리 결과로 고를지 (기본 = --prep 과 같음). "
                         "다르면 폴더 끝에 '_<그 전처리 앞 7글자>' — 예: 원본에서 고른 사진의 K0L17R6 상자 (상자 찾기와 읽기 전처리 나누기 시험)")
    ap.add_argument("--all", action="store_true", help="잘 못 찾은 사진만이 아니라 split 전체. 폴더 '<prep>_alignlong'")
    a = ap.parse_args()
    if a.split == "test":
        print("[주의] 평가용(test)입니다. 규칙상 마지막에 한 번만 돌립니다.")
    if a.all and a.hard_prep:
        ap.error("--all 과 --hard-prep 은 같이 안 씀")
    hp = a.hard_prep or a.prep
    if a.all:
        items = oc.load_split(a.split, a.limit)
    else:
        hard = oc.hard_names(a.split, hp)
        items = [it for it in oc.load_split(a.split, 0) if it["file_name"] in hard]
        if a.limit:
            items = items[::max(1, len(items) // a.limit)][:a.limit]
    prep = oc.Prep(a.prep, a.split)
    det_name = f"{a.det}_th{a.thresh:g}"
    tail = "" if a.all else "_line_hard" + ("" if hp == a.prep else f"_{hp[:7]}")
    out = oc.out_dir(a.split, det_name, a.prep, a.limit, 12, "long", tail)
    os.makedirs(out, exist_ok=True)
    if a.det == "paddle_low":
        run = lambda im: oc.paddle_boxes(im, a.thresh, a.thresh)
    else:
        d2 = Det2x(a.det, a.thresh)
        run = lambda im: oc.polys_to_rects(d2(im), *im.shape[:2])
    print(f"[{a.split}] {'전체' if a.all else '잘 못 찾은 사진'} {len(items)}장, {det_name}, 전처리 {a.prep} → {out}")
    res, t_all = {}, time.perf_counter()
    for k, it in enumerate(items, 1):
        src = cc.imread(it["image_path"])
        img, _ = prep(it, src)
        res[it["file_name"]] = {}
        for ang in al.align_angles(src)[0]:
            t0 = time.perf_counter()
            boxes = run(oc.rotate_color(img, ang))
            res[it["file_name"]][str(ang)] = {"boxes": boxes, "det_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if k % 20 == 0 or k == len(items):
            print(f"  {k}/{len(items)}  경과 {time.perf_counter() - t_all:.0f}초")
    json.dump(res, open(out / "boxes.json", "w", encoding="utf-8"))
    nb = [len(d["boxes"]) for f in res.values() for d in f.values()]
    print(f"저장: {out / 'boxes.json'}  /  방향당 상자 평균 {np.mean(nb):.1f}개, 상자 0개인 방향 {np.mean([n == 0 for n in nb]) * 100:.0f}%")


if __name__ == "__main__":
    main()
