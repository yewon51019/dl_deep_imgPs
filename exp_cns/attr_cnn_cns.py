# =====================================================================
#  attr_cnn_cns.py  —  알약 모양·제형·색 CNN (학습 계획 5번, 약 판별 DB 매칭에 쓸 확률)
# =====================================================================
#
#  [왜]  각인을 못 읽거나 비슷한 각인이 여럿일 때 모양·제형·색으로 약 후보를 가르기 위해.
#        exp300 에서는 색을 규칙(Lab 대표값)으로 정해 동점 깨기만으로 개발용 76.5 → 86.3% 였지만, 흰 알약을 파랑으로 보는 등 틀림이 있었음.
#  [라벨]  AI Hub 라벨 JSON 에서 바로 (사람 손 없음). 사진에서 보이는 것만 남기도록 묶음:
#    - 모양 4칸 : 원형 / 장방형 / 타원형 / 기타 (삼각형·마름모형·팔각형·기타 = 학습용 7종뿐이라 하나로)
#    - 제형 3칸 : 정제 / 경질캡슐 / 연질캡슐  ← form_code_name 칸에서 ('연질캡슐' 포함 → 연질, '캡슐' 포함 → 경질, 나머지 정제)
#                 dl_custom_shape(서방형정제·장용정제…)는 몸속에서 녹는 방식이라 사진으로 구분 불가 + 빈칸 47종 → 안 씀
#    - 색 12칸 (해당하는 색 모두 1, 멀티라벨) : color_class1 + color_class2 를 ',' 로 나눠 꾸밈말(투명·진한·옅은)을 떼고
#                 남색→파랑, 청록→초록, 자주→보라 로 묶음 (개발용 '남색' 은 학습용 1번 색에 없음)
#  [입력]  AI Hub 사진은 이미 알약에 맞게 잘린 크롭 → 비율 유지로 긴 변 224 정사각형 (cc.fit_side, 여백은 모서리 색)
#  [나누기]  학습용(trainall: 개발용 36종·Λ 5종 제외) 323종을 약 단위로 80% 학습 / 20% 검증(에폭 고르기용, 시드 42).
#            개발용 468장은 마지막 측정만. 평가용은 안 씀
#  [단계]
#    --step data  : labels.csv (사진마다 split·모양·제형·색) + 칸별 수 출력 + check_crops.png (24장, 입력 사진과 라벨 확인)
#    --step train : ResNet18 (ImageNet 사전학습, torchvision) + 출력 3개 (모양 4 · 제형 3 · 색 12 멀티라벨).
#                   손실 = CE(모양) + CE(제형) + BCE(색).  늘리기 = 아무 각도 회전(테두리 늘이기) · 좌우 뒤집기 · 밝기·대비 ±10%
#                   (색을 배워야 하므로 색조·채도는 안 바꿈).  에폭마다 검증(val) → 세 점수 평균이 최고인 가중치 저장
#                   → model\best.pt, model\last.pt, train_log.csv, train_curve.png
#    --step eval  : best.pt 로 val·dev 측정 → eval_<part>.csv (사진마다 정답·예측·확률) + 칸별 정확도·혼동행렬 출력
#    --step probs : best.pt 로 val·dev 확률 전체 → probs_<part>.csv (match_cns.py 용). --parts test 면 평가용 (라벨 안 씀)
#  [저장]  exp_cns\results\attr_cnn_v1\   (그림에는 원본 사진이 들어가므로 .gitignore 로 안 올림)
#  [실행]  python exp_cns\attr_cnn_cns.py --step data
#          python exp_cns\attr_cnn_cns.py --step train --epochs 15
#          python exp_cns\attr_cnn_cns.py --step eval
# =====================================================================

import argparse
import collections
import csv
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

import common_cns as cc
import ocr_common_cns as oc

OUT = cc.RESULTS / "attr_cnn_v1"
SIDE = 224
SHAPES = ["원형", "장방형", "타원형", "기타"]
FORMS = ["정제", "경질캡슐", "연질캡슐"]
COLORS = ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "파랑", "보라", "회색", "검정"]
COLOR_MERGE = {"남색": "파랑", "청록": "초록", "자주": "보라"}
COLOR_DROP = {"투명", "진한", "옅은", ""}
COLOR_EN = {"하양": "white", "노랑": "yellow", "주황": "orange", "분홍": "pink", "빨강": "red", "갈색": "brown",
            "연두": "lime", "초록": "green", "파랑": "blue", "보라": "purple", "회색": "gray", "검정": "black"}


def attrs(im, strict=True):
    """라벨 한 줄(JSON images[0] 또는 cns_drugs_summary.xlsx 한 행, 같은 칸 이름) → (모양 4칸, 제형 3칸, 색 목록).
       strict=False 면 모르는 색은 빼고 경고만 (match_cns 의 DB 용)."""
    shape = im["drug_shape"] if im["drug_shape"] in SHAPES[:3] else "기타"
    fc = im["form_code_name"] or ""
    form = "연질캡슐" if "연질캡슐" in fc else "경질캡슐" if "캡슐" in fc else "정제"
    cols = []
    for t in f"{im['color_class1'] or ''},{im['color_class2'] or ''}".split(","):
        t = COLOR_MERGE.get(t.strip(), t.strip())
        if t in COLOR_DROP:
            continue
        if t not in COLORS:
            if strict:
                raise ValueError(f"모르는 색 {t!r} ({im['drug_N']})")
            print(f"  [경고] 모르는 색 {t!r} 뺌 ({im['drug_N']})")
            continue
        if t not in cols:
            cols.append(t)
    return shape, form, cols


def drug_info(root):
    """라벨 JSON → {약 코드: (모양, 제형, 색 목록, 원래 색 글자)}. 같은 약이면 사진마다 같은 값이라 약마다 첫 JSON 만."""
    out = {}
    for p in sorted(glob.glob(str(Path(root) / "labels" / "*.json"))):
        im = json.load(open(p, encoding="utf-8"))["images"][0]
        if im["drug_N"] in out:
            continue
        out[im["drug_N"]] = (*attrs(im), f"{im['color_class1']}|{im['color_class2']}")
    return out


def step_data():
    info = drug_info(cc.TRAIN_ROOT)
    tr = oc.load_split("trainall", 0)
    dv = oc.load_split("dev", 0)
    drugs = sorted({it["drug_N"] for it in tr})
    rng = random.Random(42)
    val = set(rng.sample(drugs, round(len(drugs) * 0.2)))
    rows = []
    for split, its in (("train", tr), ("dev", dv)):
        for it in its:
            s, f, c, raw = info[it["drug_N"]]
            part = ("val" if it["drug_N"] in val else "train") if split == "train" else "dev"
            rows.append(dict(file_name=it["file_name"], image_path=it["image_path"], drug_N=it["drug_N"], part=part,
                             shape=s, form=f, colors="|".join(c), color_raw=raw, camera_la=it["camera_la"],
                             back_color=it["back_color"]))
    os.makedirs(OUT, exist_ok=True)
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for part in ("train", "val", "dev"):
        rs = [r for r in rows if r["part"] == part]
        dd = {r["drug_N"]: r for r in rs}.values()                   # 약 단위 수
        print(f"\n== {part}: 사진 {len(rs)} / 약 {len(dd)}")
        print("  모양:", dict(collections.Counter(r["shape"] for r in dd)))
        print("  제형:", dict(collections.Counter(r["form"] for r in dd)))
        print("  색  :", dict(collections.Counter(c for r in dd for c in r["colors"].split("|") if c)),
              f"/ 두 색 이상 {sum('|' in r['colors'] for r in dd)}종 / 색 없음 {sum(not r['colors'] for r in dd)}종")
        print("  배경:", dict(collections.Counter(r["back_color"] for r in rs)))
    # 확인 그림: 학습용 24장 (약마다 1장, 시드 42) — 실제 입력(224 정사각형) + 라벨
    rng = random.Random(42)
    tr_rows = [r for r in rows if r["part"] != "dev"]
    by = collections.defaultdict(list)
    for r in tr_rows:
        by[r["drug_N"]].append(r)
    pick = [rng.choice(by[d]) for d in rng.sample(sorted(by), 24)]
    cells = []
    sh_en = {"원형": "round", "장방형": "oblong", "타원형": "oval", "기타": "other"}
    fo_en = {"정제": "tablet", "경질캡슐": "hard cap", "연질캡슐": "soft cap"}
    for i, r in enumerate(pick, 1):
        img = cc.fit_side(cc.imread(r["image_path"]), SIDE)
        bar = np.full((44, SIDE, 3), 255, np.uint8)
        cv2.putText(bar, f"#{i} {r['drug_N']} {r['part']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cols = "+".join(COLOR_EN[c] for c in r["colors"].split("|") if c) or "-"
        cv2.putText(bar, f"{sh_en[r['shape']]} / {fo_en[r['form']]} / {cols}", (4, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 0, 0), 1)
        cells.append(np.pad(np.vstack([bar, img]), ((3, 3), (3, 3), (0, 0)), constant_values=128))
    grid = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, 24, 6)])
    title = np.full((34, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "attr CNN input (224, fit_side) + labels: shape / form / colors  (train, seed 42)", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cc.imwrite(str(OUT / "check_crops.png"), np.vstack([title, grid]))
    print(f"\n저장: {OUT / 'labels.csv'} ({len(rows)}줄), {OUT / 'check_crops.png'}")


# ---------------------------------------------------------------------
# 학습 · 측정
# ---------------------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], np.float32)                 # ImageNet 정규화 (사전학습 가중치와 같게)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rows(parts):
    return [r for r in csv.DictReader(open(OUT / "labels.csv", encoding="utf-8-sig")) if r["part"] in parts]


def load_images(rows):
    """사진을 미리 224 정사각형(RGB)으로 읽어 메모리에 (약 4,800장 × 150KB ≈ 0.7GB)."""
    return np.stack([cv2.cvtColor(cc.fit_side(cc.imread(r["image_path"]), SIDE), cv2.COLOR_BGR2RGB) for r in rows])


def targets(rows):
    sh = np.array([SHAPES.index(r["shape"]) for r in rows])
    fo = np.array([FORMS.index(r["form"]) for r in rows])
    co = np.zeros((len(rows), len(COLORS)), np.float32)
    for i, r in enumerate(rows):
        for c in r["colors"].split("|"):
            if c:
                co[i, COLORS.index(c)] = 1
    return sh, fo, co


def augment(img, rng):
    """아무 각도 회전(빈 곳은 테두리 늘이기) + 좌우 뒤집기 + 밝기·대비 ±10%."""
    m = cv2.getRotationMatrix2D((SIDE / 2, SIDE / 2), rng.uniform(0, 360), 1.0)
    img = cv2.warpAffine(img, m, (SIDE, SIDE), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    a, b = rng.uniform(0.9, 1.1), rng.uniform(-0.1, 0.1) * 255
    return np.clip(img.astype(np.float32) * a + b, 0, 255)


def to_tensor(batch):
    import torch
    x = (np.asarray(batch, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def build_model():
    import torch.nn as nn
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(SHAPES) + len(FORMS) + len(COLORS))   # 출력 하나로 이어 붙이고 나눠 씀
    return net


def split_out(o):
    a, b = len(SHAPES), len(SHAPES) + len(FORMS)
    return o[:, :a], o[:, a:b], o[:, b:]


def predict(net, imgs, bs=64):
    import torch
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            outs.append(net(to_tensor(imgs[i:i + bs]).cuda()).float().cpu())
    s, f, c = split_out(torch.cat(outs))
    return s.softmax(1).numpy(), f.softmax(1).numpy(), c.sigmoid().numpy()


def scores(ps, pf, pc, sh, fo, co):
    """모양 정확도, 제형 정확도, 색 '모든 칸 맞음' 비율 (0.5 기준)."""
    return (float((ps.argmax(1) == sh).mean()), float((pf.argmax(1) == fo).mean()),
            float(((pc > 0.5) == (co > 0.5)).all(1).mean()))


def step_train(a):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    tr, va = load_rows({"train"}), load_rows({"val"})
    print(f"사진 읽는 중: 학습 {len(tr)} / 검증 {len(va)}")
    xtr, xva = load_images(tr), load_images(va)
    ytr, yva = targets(tr), targets(va)
    net = build_model().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    rng = random.Random(42)
    os.makedirs(OUT / "model", exist_ok=True)
    log, best = [], -1
    for ep in range(1, a.epochs + 1):
        net.train()
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        tot = 0.0
        for k in range(nb):
            b = idx[k * a.bs:(k + 1) * a.bs]
            s, f, c = split_out(net(to_tensor([augment(xtr[i], rng) for i in b]).cuda()))
            loss = (F.cross_entropy(s, torch.from_numpy(ytr[0][b]).cuda())
                    + F.cross_entropy(f, torch.from_numpy(ytr[1][b]).cuda())
                    + F.binary_cross_entropy_with_logits(c, torch.from_numpy(ytr[2][b]).cuda()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        sc = scores(*predict(net, xva), *yva)
        mean = sum(sc) / 3
        log.append([ep, round(tot / nb, 4), *[round(v, 4) for v in sc], round(mean, 4)])
        print(f"  에폭 {ep:2d}  loss {log[-1][1]:.3f}  검증 모양 {sc[0]:.3f} 제형 {sc[1]:.3f} 색 {sc[2]:.3f}  평균 {mean:.3f}", flush=True)
        if mean > best:
            best = mean
            torch.save({"state": net.state_dict(), "epoch": ep, "val": sc}, OUT / "model" / "best.pt")
    torch.save({"state": net.state_dict(), "epoch": a.epochs}, OUT / "model" / "last.pt")
    with open(OUT / "train_log.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss", "val_shape", "val_form", "val_color_all", "val_mean"])
        w.writerows(log)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [r[0] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(e, [r[1] for r in log], color="#3b6fb6")
    ax[0].set(title="train loss", xlabel="epoch")
    for i, (n, col) in enumerate((("shape", "#3b6fb6"), ("form", "#c9a227"), ("color (all correct)", "#2e8b57"), ("mean", "#c0392b"))):
        ax[1].plot(e, [r[2 + i] for r in log], label=n, color=col, lw=2 if n == "mean" else 1)
    bi = max(log, key=lambda r: r[5])
    ax[1].axvline(bi[0], ls="--", color="#c0392b", lw=0.8)
    ax[1].set(title=f"val (65 drugs) - best mean {bi[5]:.3f} @ epoch {bi[0]}", xlabel="epoch")
    ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "train_curve.png", dpi=110)
    print(f"저장: {OUT / 'model' / 'best.pt'} (에폭 {bi[0]}), {OUT / 'train_curve.png'}")


def step_eval(a):
    import torch
    net = build_model().cuda()
    ck = torch.load(OUT / "model" / "best.pt", map_location="cuda")
    net.load_state_dict(ck["state"])
    print(f"가중치: best.pt (에폭 {ck['epoch']})")
    for part in ("val", "dev"):
        rows = load_rows({part})
        ps, pf, pc = predict(net, load_images(rows))
        sh, fo, co = targets(rows)
        sc = scores(ps, pf, pc, sh, fo, co)
        print(f"\n== {part} ({len(rows)}장): 모양 {sc[0]:.3f} / 제형 {sc[1]:.3f} / 색 모든 칸 맞음 {sc[2]:.3f}")
        for name, cls, y, p in (("모양", SHAPES, sh, ps), ("제형", FORMS, fo, pf)):
            cm = np.zeros((len(cls), len(cls)), int)
            for t, q in zip(y, p.argmax(1)):
                cm[t, q] += 1
            print(f"  {name} 혼동행렬 (줄 = 정답, 칸 = 예측) {cls}")
            for i, c in enumerate(cls):
                print(f"    {c:6s} {cm[i].tolist()}")
        print("  색 칸별 (정답 장수 / 정밀도 / 재현율):")
        for j, c in enumerate(COLORS):
            t, q = co[:, j] > 0.5, pc[:, j] > 0.5
            if t.sum() or q.sum():
                print(f"    {c}: {int(t.sum())} / {(t & q).sum() / max(q.sum(), 1):.2f} / {(t & q).sum() / max(t.sum(), 1):.2f}")
        with open(OUT / f"eval_{part}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "drug_N", "shape", "shape_pred", "shape_p", "form", "form_pred", "form_p",
                        "colors", "colors_pred"] + [f"p_{c}" for c in COLORS])
            for i, r in enumerate(rows):
                w.writerow([r["file_name"], r["drug_N"], r["shape"], SHAPES[ps[i].argmax()], round(float(ps[i].max()), 3),
                            r["form"], FORMS[pf[i].argmax()], round(float(pf[i].max()), 3), r["colors"],
                            "|".join(c for j, c in enumerate(COLORS) if pc[i, j] > 0.5)] + [round(float(v), 3) for v in pc[i]])
        print(f"  저장: {OUT / f'eval_{part}.csv'}")


def step_probs(a):
    """best.pt 로 val·dev 사진마다 모양 4칸·제형 3칸·색 12칸 확률 전체 → probs_<part>.csv (match_cns 의 겉모습 점수용)."""
    import torch
    net = build_model().cuda()
    net.load_state_dict(torch.load(OUT / "model" / "best.pt", map_location="cuda")["state"])
    for part in a.parts.split(","):
        if part in ("test", "exp300test") or part.startswith("exp300dev51"):   # 평가용·사후 비교용: labels.csv 에 없음 → 사진 목록만 (라벨은 안 씀)
            import ocr_common_cns as oc
            rows = [dict(file_name=it["file_name"], drug_N=it["drug_N"], image_path=it["image_path"])
                    for it in oc.load_split(part, 0)]
        else:
            rows = load_rows({part})
        ps, pf, pc = predict(net, load_images(rows))
        with open(OUT / f"probs_{part}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "drug_N"] + [f"s_{c}" for c in SHAPES] + [f"f_{c}" for c in FORMS] + [f"c_{c}" for c in COLORS])
            for i, r in enumerate(rows):
                w.writerow([r["file_name"], r["drug_N"]] + [round(float(v), 5) for v in (*ps[i], *pf[i], *pc[i])])
        print(f"저장: {OUT / f'probs_{part}.csv'} ({len(rows)}장)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["data", "train", "eval", "probs"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--parts", default="val,dev", help="--step probs 의 대상 (test = 평가용, 마지막에 한 번)")
    a = ap.parse_args()
    if a.step == "data":
        step_data()
    else:
        {"train": step_train, "eval": step_eval, "probs": step_probs}[a.step](a)


if __name__ == "__main__":
    main()
