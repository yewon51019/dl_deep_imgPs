# =====================================================================
#  orient_crop_cns.py  —  글자 조각 위아래(0° / 180°) 판별기 (사람 라벨 없이, 상자 순서로 방향을 아는 조각으로 학습)
# =====================================================================
#
#  [왜]  PARSeq 는 거꾸로 된 조각도 정답으로 읽고 확신도도 '받은 방향'이 높아서 0°/180° 를 못 가름 (CLAUDE.md 10장 3-2)
#  [방향을 아는 조각]  parseq_label_v1\crops\ 중 상자가 2개 이상인 사진 (ocr_v1\trainall pred.csv 의 best_boxes)
#     up   = 맞힌 사진(auto_ok)이고 상자 글자가 서로 다름 → 이어 붙인 순서가 정답 = 똑바로 선 조각
#     flip = 순서만 다름(auto_order)이고 상자 순서를 거꾸로 붙이면 정답 → 사진이 뒤집혀 읽힌 것 = 180° 조각 (돌려서 똑바로)
#  [학습]  똑바른 조각 → 라벨 0, 180° 돌린 것 → 라벨 1 (에폭마다 조각마다 둘 중 하나를 무작위로)
#          ResNet18(ImageNet) + 출력 2칸, 입력 64×192 (그냥 늘림, PARSeq 처럼), 밝기·대비 ±15%, 회전 ±8°, 좌우 뒤집기는 안 함(거울 글자)
#          나누기 = crops 의 train / val (약 단위, attr_cnn_v1 과 같은 약). val = 방향을 아는 val 조각을 그대로·돌려서 둘 다 채점
#  [--step fix]  모든 조각을 똑바로 세움 → parseq_label_v1\crops_up\<train|val>\ + train.csv · val.csv (crops\ 는 그대로)
#          방향을 아는 조각은 그 방향, 나머지(상자 1개·사람 검수)는 P(똑바름 | 그대로) vs P(똑바름 | 180° 돌림) 큰 쪽
#          csv 칸 추가: orient (known_up / known_flip / cnn), turned (180° 돌렸으면 1), p_up (그대로 둔 쪽의 똑바름 확률)
#          확인 그림 orient_crop_v1\check_fixed24.png (cnn 으로 정한 조각 무작위 24개, 시드 42)
#  [저장]  exp_cns\results\orient_crop_v1\  model\best.pt, train_log.csv, train_curve.png, eval_val.csv
#  [--classes 4 (v2)]  0/90/180/270° 네 칸. 세로로 선 조각(높이 > 폭, 2글자 이상, 약 10%)까지 세우려고.
#          입력 = 128×128 정사각형 (긴 변을 128 로 줄이고 가장자리 색으로 채움 — 세로 조각이 찌그러지지 않게)
#          학습 = 방향을 아는 조각 중 넓은 것만 (높이 > 폭 이고 2글자 이상이면 이미 세로일 수 있어 뺌), 네 방향으로 돌려 라벨 0~3
#          세우기 = 넓은 조각(2글자+) 0/180 중, 높은 조각(2글자+) 90/270 중, 1글자는 네 방향 중 P(똑바름) 최고
#                   (방향을 아는 조각은 먼저 그 방향으로 세운 뒤 같은 규칙)
#          저장 = results\orient_crop_v2\, parseq_label_v1\crops_up4\  (v1 결과는 그대로)
#  [실행]  python exp_cns\orient_crop_cns.py --step train   [--classes 4]
#          python exp_cns\orient_crop_cns.py --step fix     [--classes 4]
# =====================================================================

import argparse
import collections
import csv
import json
import os
import random

import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np

import common_cns as cc

OUT = cc.RESULTS / "orient_crop_v1"
CROPS = cc.RESULTS / "parseq_label_v1" / "crops"
CROPS_UP = cc.RESULTS / "parseq_label_v1" / "crops_up"
PRED = cc.RESULTS / "ocr_v1" / "trainall" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"
H, W = 64, 192
EPOCHS = 12
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NC = 2                                                               # 칸 수 (main 에서 --classes 로 바꿈)
ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rot(im, k):
    """k × 90° 시계 방향."""
    return im if k % 4 == 0 else cv2.rotate(im, ROT[k % 4])


def tall(im, label):
    return im.shape[0] > im.shape[1] and len(label) >= 2


def load_crops(part):
    """crops\<part>.csv 의 조각마다 known = 'up' / 'flip' / '' (모름)."""
    pred = {r["file_name"]: r for r in csv.DictReader(open(PRED, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(CROPS / f"{part}.csv", encoding="utf-8-sig")))
    for x in rows:
        r = pred[x["file_name"]]
        pcs = [b[4] for b in json.loads(r["best_boxes"] or "[]") if b[4]]
        x["known"] = ""
        if x["source"] == "auto_ok" and len(pcs) >= 2 and len(set(pcs)) == len(pcs):
            x["known"] = "up"
        elif x["source"] == "auto_order" and len(pcs) >= 2 and "".join(pcs[::-1]) == r["truth"]:
            x["known"] = "flip"
    return rows


def upright(x):
    im = cc.imread(str(CROPS / x["path"]))
    return cv2.rotate(im, cv2.ROTATE_180) if x["known"] == "flip" else im


def to_tensor(im):
    if NC == 4:                                                      # 정사각형 128: 긴 변 맞추고 가장자리 색으로 채움
        h, w = im.shape[:2]
        s = 128 / max(h, w)
        im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        im = cv2.copyMakeBorder(im, (128 - h) // 2, 128 - h - (128 - h) // 2, (128 - w) // 2, 128 - w - (128 - w) // 2,
                                cv2.BORDER_REPLICATE)
    else:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(im.transpose(2, 0, 1))


def augment(im, rng):
    a = rng.uniform(-8, 8)
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
    im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    c, b = rng.uniform(0.85, 1.15), rng.uniform(-0.15, 0.15) * 255
    return np.clip(im.astype(np.float32) * c + b, 0, 255).astype(np.uint8)


def make_model(pretrained=True):
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NC)
    return m.to(DEV)


@torch.no_grad()
def p_up(model, ims, bs=128):
    """조각 목록 → P(똑바름) (라벨 0 의 확률)."""
    model.eval()
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[i:i + bs]]).to(DEV)
        out += model(x).softmax(-1)[:, 0].tolist()
    return out


def evaluate4(model, ims):
    """(네 칸) 똑바른 조각을 네 방향으로 돌려 → (네 칸 정확도 %, 0/180 고르기 정확도 %, 90/270 고르기 정확도 %, 칸별 확률)."""
    P = [probs(model, [rot(im, k) for im in ims]) for k in range(4)]   # P[k][i] = 조각 i 를 k 돌렸을 때 네 칸 확률
    acc = sum(int(np.argmax(P[k][i])) == k for k in range(4) for i in range(len(ims))) / (4 * len(ims))
    # 받은 조각이 k 돌아가 있을 때, 되돌릴 후보 r 마다 P(똑바름 | 되돌린 것) = P[(k + r) % 4][i][0]
    pick02 = sum(P[0][i][0] > P[2][i][0] for i in range(len(ims))) / len(ims)
    pick13 = sum(P[1][i][1] > P[1][i][3] for i in range(len(ims))) / len(ims)   # 90° 돌아간 조각: 칸 1 이 칸 3 보다 높으면 맞음
    return 100 * acc, 100 * pick02, 100 * pick13, P


@torch.no_grad()
def probs(model, ims, bs=128):
    model.eval()
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[i:i + bs]]).to(DEV)
        out += model(x).softmax(-1).cpu().numpy().tolist()
    return out


def evaluate(model, ims):
    """방향을 아는(똑바로 세운) 조각을 그대로·180° 돌려서 둘 다 → (정확도 %, 짝 정확도 %).
       짝 정확도 = P(그대로) > P(돌림) 인 비율 (fix 단계에서 실제로 쓰는 규칙)."""
    a = p_up(model, ims)
    b = p_up(model, [cv2.rotate(im, cv2.ROTATE_180) for im in ims])
    acc = (sum(p >= 0.5 for p in a) + sum(p < 0.5 for p in b)) / (2 * len(ims))
    pair = sum(p > q for p, q in zip(a, b)) / len(ims)
    return 100 * acc, 100 * pair, a, b


def step_train():
    torch.manual_seed(42)
    rng = random.Random(42)
    tr = [x for x in load_crops("train") if x["known"]]
    va = [x for x in load_crops("val") if x["known"]]
    tr_im = [upright(x) for x in tr]
    va_im = [upright(x) for x in va]
    print(f"방향을 아는 조각: 학습 {len(tr)} ({dict(collections.Counter(x['known'] for x in tr))}) / "
          f"검증 {len(va)} ({dict(collections.Counter(x['known'] for x in va))})")
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 64
    steps = EPOCHS * ((len(tr_im) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(OUT / "model", exist_ok=True)
    log, best = [], -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = list(range(len(tr_im)))
        rng.shuffle(idx)
        tot, n = 0.0, 0
        for i in range(0, len(idx), bs):
            xs, ys = [], []
            for j in idx[i:i + bs]:
                y = rng.random() < 0.5
                im = augment(tr_im[j], rng)
                xs.append(to_tensor(cv2.rotate(im, cv2.ROTATE_180) if y else im))
                ys.append(int(y))
            x, y = torch.stack(xs).to(DEV), torch.tensor(ys).to(DEV)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ys)
            n += len(ys)
        acc, pair, _, _ = evaluate(model, va_im)
        log.append(dict(epoch=ep, loss=round(tot / n, 4), val_acc=round(acc, 2), val_pair=round(pair, 2)))
        print(f"  에폭 {ep:2d}  loss {tot / n:.4f}  검증 정확도 {acc:.1f}  짝 정확도 {pair:.1f}")
        if pair > best:
            best = pair
            torch.save(model.state_dict(), OUT / "model" / "best.pt")
    with open(OUT / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([l["epoch"] for l in log], [l["loss"] for l in log])
    ax[0].set(title="train loss", xlabel="epoch")
    ax[1].plot([l["epoch"] for l in log], [l["val_acc"] for l in log], label="val acc (both turns)")
    ax[1].plot([l["epoch"] for l in log], [l["val_pair"] for l in log], label="val pair (as-is > turned)")
    ax[1].set(title="val (known-orientation crops)", xlabel="epoch")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "train_curve.png", dpi=110)
    # 최고 모델로 검증 조각마다 결과 (글자 길이별로 볼 수 있게)
    model.load_state_dict(torch.load(OUT / "model" / "best.pt", map_location=DEV))
    acc, pair, a, b = evaluate(model, va_im)
    with open(OUT / "eval_val.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "known", "p_up_asis", "p_up_turned", "ok"])
        for x, p, q in zip(va, a, b):
            w.writerow([x["path"], x["label"], x["known"], round(p, 4), round(q, 4), int(p > q)])
    by = collections.defaultdict(list)
    for x, p, q in zip(va, a, b):
        by[min(len(x["label"]), 4)].append(p > q)
    print(f"최고 모델: 검증 정확도 {acc:.1f} / 짝 정확도 {pair:.1f}  글자 수별 짝 정확도 "
          + ", ".join(f"{k}{'+' if k == 4 else ''}글자 {100 * sum(v) / len(v):.1f} ({len(v)})" for k, v in sorted(by.items())))


def step_fix():
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(OUT / "model" / "best.pt", map_location=DEV))
    allrec = []
    for part in ("train", "val"):
        rows = load_crops(part)
        ims = [cc.imread(str(CROPS / x["path"])) for x in rows]
        a = p_up(model, ims)
        b = p_up(model, [cv2.rotate(im, cv2.ROTATE_180) for im in ims])
        os.makedirs(CROPS_UP / part, exist_ok=True)
        out = []
        for x, im, p, q in zip(rows, ims, a, b):
            if x["known"]:
                turn, orient = x["known"] == "flip", "known_" + x["known"]
            else:
                turn, orient = q > p, "cnn"
            cc.imwrite(str(CROPS_UP / x["path"]), cv2.rotate(im, cv2.ROTATE_180) if turn else im)
            rec = {k: x[k] for k in ("path", "label", "source", "file_name", "drug_N")}
            rec.update(orient=orient, turned=int(turn), p_up=round(q if turn else p, 4))
            out.append(rec)
        with open(CROPS_UP / f"{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        c = [r for r in out if r["orient"] == "cnn"]
        print(f"{part}: {len(out)}개 저장 / cnn 으로 정한 {len(c)}개 중 180° 돌림 {sum(r['turned'] for r in c)} "
              f"(사람 {sum(r['turned'] for r in c if r['source'] == 'human')}/{sum(r['source'] == 'human' for r in c)}), "
              f"확신 낮음(p_up < 0.7) {sum(r['p_up'] < 0.7 for r in c)}")
        allrec += out
    # 확인 그림: cnn 으로 정한 조각 무작위 24개 (세운 뒤 모습 + 글자 + 돌렸는지·확률)
    rng = random.Random(42)
    pick = rng.sample([r for r in allrec if r["orient"] == "cnn"], 24)
    tiles = []
    for r in pick:
        im = cc.imread(str(CROPS_UP / r["path"]))
        im = cv2.resize(im, (min(260, max(24, int(im.shape[1] * 80 / im.shape[0]))), 80))
        t = np.full((130, 270, 3), 255, np.uint8)
        t[:80, :im.shape[1]] = im
        cv2.putText(t, r["label"][:14], (2, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        cv2.putText(t, f"{'turned' if r['turned'] else 'as-is'} p={r['p_up']:.2f} {r['source']}", (2, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)])
    cc.imwrite(str(OUT / "check_fixed24.png"), sheet)
    print("저장:", OUT / "check_fixed24.png")


def step_train4():
    """네 칸판 학습 (step_train 과 같은 절차, 라벨 = 돌린 횟수 0~3)."""
    torch.manual_seed(42)
    rng = random.Random(42)
    tr = [x for x in load_crops("train") if x["known"]]
    va = [x for x in load_crops("val") if x["known"]]
    tr_im = [upright(x) for x in tr]
    va_im = [upright(x) for x in va]
    keep = lambda xs, ims: [im for x, im in zip(xs, ims) if not tall(im, x["label"])]
    tr_im, va_im = keep(tr, tr_im), keep(va, va_im)
    print(f"방향을 아는 넓은 조각: 학습 {len(tr_im)} / 검증 {len(va_im)} (높은 조각 뺌)")
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 64
    steps = EPOCHS * ((len(tr_im) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(OUT / "model", exist_ok=True)
    log, best = [], -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = list(range(len(tr_im)))
        rng.shuffle(idx)
        tot, n = 0.0, 0
        for i in range(0, len(idx), bs):
            xs, ys = [], []
            for j in idx[i:i + bs]:
                k = rng.randrange(4)
                xs.append(to_tensor(rot(augment(tr_im[j], rng), k)))
                ys.append(k)
            x, y = torch.stack(xs).to(DEV), torch.tensor(ys).to(DEV)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ys)
            n += len(ys)
        acc, p02, p13, _ = evaluate4(model, va_im)
        log.append(dict(epoch=ep, loss=round(tot / n, 4), val_acc4=round(acc, 2), val_pick02=round(p02, 2), val_pick13=round(p13, 2)))
        print(f"  에폭 {ep:2d}  loss {tot / n:.4f}  검증 네 칸 {acc:.1f}  0/180 고르기 {p02:.1f}  90/270 고르기 {p13:.1f}")
        if p02 + p13 > best:
            best = p02 + p13
            torch.save(model.state_dict(), OUT / "model" / "best.pt")
    with open(OUT / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    model.load_state_dict(torch.load(OUT / "model" / "best.pt", map_location=DEV))
    acc, p02, p13, _ = evaluate4(model, va_im)
    print(f"최고 모델: 검증 네 칸 {acc:.1f} / 0/180 고르기 {p02:.1f} / 90/270 고르기 {p13:.1f}")


def step_fix4():
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(OUT / "model" / "best.pt", map_location=DEV))
    allrec = []
    for part in ("train", "val"):
        rows = load_crops(part)
        ims = [upright(x) if x["known"] else cc.imread(str(CROPS / x["path"])) for x in rows]   # 아는 방향은 먼저 세움
        P = [probs(model, [rot(im, k) for im in ims]) for k in range(4)]
        os.makedirs(CROPS_UP / part, exist_ok=True)
        out = []
        for i, (x, im) in enumerate(zip(rows, ims)):
            if len(x["label"]) < 2:
                cand = [0, 1, 2, 3]
            elif tall(im, x["label"]):
                cand = [1, 3]
            else:
                cand = [0] if x["known"] else [0, 2]                 # 순서로 확인한 0/180 은 CNN 보다 믿음
            k = max(cand, key=lambda r: P[r][i][0])                  # 되돌릴 회전 r: P(똑바름 | r 돌린 조각) 최고
            cc.imwrite(str(CROPS_UP / x["path"]), rot(im, k))
            rec = {key: x[key] for key in ("path", "label", "source", "file_name", "drug_N")}
            rec.update(orient=("known_" + x["known"]) if x["known"] else "cnn", turn=90 * k, p_up=round(P[k][i][0], 4))
            out.append(rec)
        with open(CROPS_UP / f"{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"{part}: {len(out)}개 / 추가로 돌린 각도 {dict(collections.Counter(r['turn'] for r in out))}  "
              f"(방향 아는 조각 중 돌림 {sum(r['turn'] != 0 for r in out if r['orient'] != 'cnn')}/{sum(r['orient'] != 'cnn' for r in out)}), "
              f"확신 낮음(p_up < 0.7) {sum(r['p_up'] < 0.7 for r in out)}")
        allrec += out
    # 확인 그림: 방향을 몰랐던 조각 24개 (시드 42) + 높은 조각이었던 것 6개 섞음
    rng = random.Random(42)
    unk = [r for r in allrec if r["orient"] == "cnn" and r["turn"] in (0, 180)]
    tl = [r for r in allrec if r["turn"] in (90, 270)]
    pick = rng.sample(unk, 18) + rng.sample(tl, min(6, len(tl)))
    tiles = []
    for r in pick:
        im = cc.imread(str(CROPS_UP / r["path"]))
        im = cv2.resize(im, (min(260, max(24, int(im.shape[1] * 80 / im.shape[0]))), 80))
        t = np.full((130, 270, 3), 255, np.uint8)
        t[:80, :im.shape[1]] = im
        cv2.putText(t, r["label"][:14], (2, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        cv2.putText(t, f"turn {r['turn']} p={r['p_up']:.2f} {r['source']}", (2, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)])
    cc.imwrite(str(OUT / "check_fixed24.png"), sheet)
    print("저장:", OUT / "check_fixed24.png", "(위 3줄 = 0/180 고른 조각, 마지막 줄 = 세로였던 조각)")


def main():
    global NC, OUT, CROPS_UP
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["train", "fix"])
    ap.add_argument("--classes", type=int, default=2, choices=[2, 4])
    a = ap.parse_args()
    if a.classes == 4:
        NC, OUT, CROPS_UP = 4, cc.RESULTS / "orient_crop_v2", cc.RESULTS / "parseq_label_v1" / "crops_up4"
        {"train": step_train4, "fix": step_fix4}[a.step]()
    else:
        {"train": step_train, "fix": step_fix}[a.step]()


if __name__ == "__main__":
    main()
