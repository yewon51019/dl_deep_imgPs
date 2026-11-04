# =====================================================================
#  parseq_ft_cns.py  —  PARSeq 파인튜닝 (똑바로 세운 글자 조각, 학습 계획 1단계)
# =====================================================================
#
#  [데이터]  parseq_label_v1\crops_up4\train.csv · val.csv  (orient_crop_cns.py --classes 4 로 세운 조각, 약 단위 학습 253종 / 검증 64종)
#     --data sure : 방향을 순서로 확인한 조각(known_*) + 판별기 확신도 p_up ≥ 0.9 인 조각만 (방향이 틀린 조각을 줄이려고)
#     --data all  : 전부
#     에폭 고르기 = 사람이 검수한 검증 조각(human, 72개) 정확도 (norm 한 읽기 == norm 한 라벨)
#        — 자동 조각은 사전학습 PARSeq 가 이미 맞힌 사진에서 나와 학습 전에도 95.9% 라 고르는 데 못 씀 (1차 시도 parseq_ft1_sure_try1)
#        참고로 검증 sure 조각 정확도도 같이 기록. PARSeq 글자표에 없는 글자가 든 라벨은 뺌
#     사람 조각(어려운 사진)은 학습에서 HUMAN_REP 번 반복
#  [학습]  사전학습 PARSeq(baudm/parseq) 에서 시작, PARSeq 의 training_step(순서 바꾸기 언어 모델 손실) 그대로
#          AdamW lr 5e-5 (처음 10% 데움 + 코사인), 배치 48, 20 에폭. 늘리기: 회전 ±5°, 밝기·대비 ±20%, 가끔 흐림
#          입력 변환은 읽을 때와 같음 (32×128 bicubic, 정규화 0.5)
#  [저장]  exp_cns\results\parseq_ft1_<data>\  model\best.pt (state_dict), train_log.csv, train_curve.png
#  [실행]  python exp_cns\parseq_ft_cns.py --data sure
# =====================================================================

import argparse
import csv
import math
import os
import random

import torch
import cv2
import numpy as np
from PIL import Image

import common_cns as cc
import ocr_cns

CROPS = cc.RESULTS / "parseq_label_v1" / "crops_up4"
EPOCHS = 20
BS = 48
LR = 5e-5
HUMAN_REP = 4


def load(part, data, charset):
    """→ (사람 조각을 뺀 sure(또는 전부) 조각 목록, 사람 조각 목록). 조각 = (사진, 글자)."""
    rows = list(csv.DictReader(open(CROPS / f"{part}.csv", encoding="utf-8-sig")))
    sure = lambda r: r["orient"] != "cnn" or float(r["p_up"]) >= 0.9
    rows = [r for r in rows if all(c in charset for c in r["label"])]
    human = [(cc.imread(str(CROPS / r["path"])), r["label"]) for r in rows if r["source"] == "human"]
    rows = [r for r in rows if r["source"] != "human" and (sure(r) or (part == "train" and data == "all"))]
    return [(cc.imread(str(CROPS / r["path"])), r["label"]) for r in rows], human


def augment(im, rng):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1)
    im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    im = np.clip(im.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2) * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return im


def tensor(ps, im):
    return ps.tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))


@torch.no_grad()
def accuracy(ps, data, bs=128):
    ps.m.eval()
    norm = ocr_cns.base.norm
    ok = 0
    for i in range(0, len(data), bs):
        x = torch.stack([tensor(ps, im) for im, _ in data[i:i + bs]]).to(ps.dev)
        labels, _ = ps.m.tokenizer.decode(ps.m(x).softmax(-1))
        ok += sum(norm(p) == norm(t) for p, (_, t) in zip(labels, data[i:i + bs]))
    return 100 * ok / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sure", choices=["sure", "all"])
    a = ap.parse_args()
    out = cc.RESULTS / f"parseq_ft1_{a.data}"
    os.makedirs(out / "model", exist_ok=True)
    torch.manual_seed(42)
    rng = random.Random(42)
    ps = ocr_cns.Reader().ps                                         # run_exp300.PARSeq (모델 ps.m, 입력 변환 ps.tf)
    charset = set(ps.m.hparams.charset_train)
    (tr, tr_h), (va, va_h) = load("train", a.data, charset), load("val", a.data, charset)
    tr = tr + tr_h * HUMAN_REP                                       # 사람 조각은 확신도와 상관없이 전부, 반복해서
    print(f"[{a.data}] 학습 조각 {len(tr)} (사람 {len(tr_h)} × {HUMAN_REP}) / 검증 sure {len(va)} · 사람 {len(va_h)}")
    acc0, h0 = accuracy(ps, va), accuracy(ps, va_h)
    print(f"  학습 전 검증 sure {acc0:.1f} / 사람 {h0:.1f}")
    opt = torch.optim.AdamW(ps.m.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(tr) / BS)
    warm = int(0.1 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    log, best = [dict(epoch=0, loss="", val_acc=round(acc0, 2), val_human=round(h0, 2))], h0
    torch.save(ps.m.state_dict(), out / "model" / "best.pt")
    best_ep = 0
    for ep in range(1, EPOCHS + 1):
        ps.m.train()
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        tot, n = 0.0, 0
        for i in range(0, len(idx), BS):
            b = [tr[j] for j in idx[i:i + BS]]
            x = torch.stack([tensor(ps, augment(im, rng)) for im, _ in b]).to(ps.dev)
            loss = ps.m.training_step((x, [t for _, t in b]), i)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps.m.parameters(), 20)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        acc, h = accuracy(ps, va), accuracy(ps, va_h)
        log.append(dict(epoch=ep, loss=round(tot / n, 4), val_acc=round(acc, 2), val_human=round(h, 2)))
        print(f"  에폭 {ep:2d}  loss {tot / n:.4f}  검증 sure {acc:.1f} / 사람 {h:.1f}", flush=True)
        if h > best:
            best, best_ep = h, ep
            torch.save(ps.m.state_dict(), out / "model" / "best.pt")
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([l["epoch"] for l in log[1:]], [l["loss"] for l in log[1:]])
    ax[0].set(title="train loss", xlabel="epoch")
    ax[1].plot([l["epoch"] for l in log], [l["val_acc"] for l in log], marker=".", label="val sure crops")
    ax[1].plot([l["epoch"] for l in log], [l["val_human"] for l in log], marker=".", label="val human crops")
    ax[1].set(title=f"val ({a.data}) - best human {best:.1f} @ {best_ep}", xlabel="epoch")
    ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "train_curve.png", dpi=110)
    print(f"최고 검증 사람 조각 정확도 {best:.1f} (에폭 {best_ep}, 학습 전 {h0:.1f}) → {out / 'model' / 'best.pt'}")


if __name__ == "__main__":
    main()
