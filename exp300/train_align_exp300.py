# =====================================================================
#  train_align_exp300.py  —  ⑤ 각도 분류기 / ⑥ TPS 펴기 학습 (가짜 알약 사진으로만)
# =====================================================================
#
#  [⑤ cnn]  ResNet18(ImageNet 사전학습)이 둥근 알약 사진을 보고 '글자를 세우려면 몇 도 돌려야 하나'를 36칸(10°씩) 중에서 고름
#           학습 사진: synth_pills.make_pill(phi = 0~360° 무작위) → 정답 = (360 - phi) % 360
#  [⑥ tps]  TPSRectifier(작은 CNN + TPS)가 알약 사진에서 글자 띠를 펴서 32×128 로 꺼내고, 얼려 둔 PARSeq 가 읽음
#           PARSeq 는 학습하지 않고(사전학습 그대로) 펴는 네트워크만 'PARSeq 가 정답을 읽도록' 학습
#           손실 = teacher forcing (정답 글자를 앞 글자로 넣고 다음 글자의 로그확률, 대·소문자 합침) — tf_loss 참고
#                + 기준점 위치 손실 (가짜 사진은 글자 띠의 위·아래 테두리를 알므로 기준점 20개의 정답 위치와의 거리)
#           ※ 읽기 손실만으로는 펴기망이 사진마다 다르게 펴는 법을 거의 못 배웠음 (정답률 50% 그대로) → 기준점 손실 추가
#           학습 사진: 한 줄 / 가장자리를 따라 휜 글자, 기울기 ±20° (실제로는 12방향(30°씩) 중 가장 가까운 방향에서 펴므로)
#
#  [저장]  exp300\results\models\angle_cnn.pt, tps_stn.pt  (+ 학습 기록 *_log.csv)
#  [규칙]  AI Hub 사진은 학습에 쓰지 않음 → 평가용(test)과 겹칠 일이 없음. 고르기는 개발용(dev) 51장으로.
#
#  [실행 예]
#     python exp300\train_align_exp300.py --task cnn
#     python exp300\train_align_exp300.py --task tps
#     python exp300\train_align_exp300.py --task tps_eval     ← 합성 검증표 (곧은/휜 글자 × 학습 전/후)
# =====================================================================

import argparse
import csv
import os
import time
from multiprocessing import Pool

import torch                                         # Paddle 보다 먼저

import cv2
import numpy as np

import align_exp300 as al
import run_exp300 as base
from synth_pills import make_pill


# ----------------------------- 가짜 사진 만들기 (여러 프로세스로) -----------------------------
def _gen_cnn(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        phi = float(rng.uniform(0, 360))
        g, _, _ = make_pill(rng, 192, phi=phi)
        xs.append(cv2.resize(g, (al.CNN_SIZE, al.CNN_SIZE), interpolation=cv2.INTER_AREA))
        ys.append(int(round(((360 - phi) % 360) / (360 / al.CNN_BINS))) % al.CNN_BINS)
    return np.stack(xs), np.array(ys)


def _gen_tps(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    for _ in range(n):
        lay = "arc" if rng.random() < 0.35 else "line"
        g, t, _, fid = make_pill(rng, al.TPS_IN, phi=float(rng.uniform(-20, 20)), layout=lay, return_fid=True)
        xs.append(g)
        ts.append((t, fid.astype(np.float32)))
    return np.stack(xs), ts


def generate(fn, n, seed, workers=8, chunk=500):
    jobs = [(seed * 100000 + i, min(chunk, n - i * chunk)) for i in range((n + chunk - 1) // chunk)]
    with Pool(workers) as p:
        parts = p.map(fn, jobs)
    xs = np.concatenate([a for a, _ in parts])
    ys = [y for _, b in parts for y in b]
    return xs, ys


# ----------------------------- ⑤ 학습 -----------------------------
def train_cnn(a, dev):
    t0 = time.time()
    xtr, ytr = generate(_gen_cnn, a.n_train, 1)
    xva, yva = generate(_gen_cnn, a.n_val, 2)
    ytr, yva = np.array(ytr), np.array(yva)
    print(f"가짜 사진: 학습 {len(xtr)} / 검증 {len(xva)} ({time.time() - t0:.0f}s)")
    m = al.build_angle_cnn(pretrained=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xtr) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    log = []
    for ep in range(a.epochs):
        m.train()
        perm = np.random.default_rng(ep).permutation(len(xtr))
        tl = 0.0
        for i in range(len(xtr) // a.bs):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            x = al.cnn_input(list(xtr[idx]), dev)
            x = x * torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(0.8, 1.2) \
                + torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(-0.3, 0.3)          # 밝기·대비 흔들기
            loss = lossf(m(x), torch.from_numpy(ytr[idx]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tl += loss.item()
        acc10, acc20, acc180 = eval_cnn(m, xva, yva, dev)
        log.append({"epoch": ep + 1, "loss": round(tl / (len(xtr) // a.bs), 4), "val_within10": acc10,
                    "val_within20": acc20, "val_within20_mod180": acc180})
        print(f"  epoch {ep + 1}/{a.epochs} loss {log[-1]['loss']:.3f} | 검증: ±10° {acc10:.3f}  ±20° {acc20:.3f}  "
              f"(180° 뒤집힘 허용 ±20° {acc180:.3f})  {time.time() - t0:.0f}s")
    return m, log


def eval_cnn(m, x, y, dev):
    m.eval()
    pr = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            pr.append(m(al.cnn_input(list(x[i:i + 256]), dev)).argmax(-1).cpu().numpy())
    d = np.abs(np.concatenate(pr) - y) % al.CNN_BINS
    d = np.minimum(d, al.CNN_BINS - d) * (360 / al.CNN_BINS)
    d180 = np.minimum(d, 180 - d)
    return float(np.mean(d <= 10)), float(np.mean(d <= 20)), float(np.mean(d180 <= 20))


# ----------------------------- ⑥ 학습 -----------------------------
def case_alt(tok, dev):
    """토큰 번호 → 대·소문자 짝 번호 (없으면 자기 자신)"""
    alt = torch.arange(len(tok), device=dev)
    for ch, i in tok._stoi.items():
        if len(ch) == 1 and ch.isalpha():
            o = ch.lower() if ch.isupper() else ch.upper()
            if o in tok._stoi:
                alt[i] = tok._stoi[o]
    return alt


def tf_loss(parseq, images, labels, alt):
    """teacher forcing 손실: 정답 글자를 앞 글자로 넣고 다음 글자 확률을 봄 (대·소문자는 같은 글자로 합침).
       PARSeq 기본 forward_logits_loss 는 평가 모드에서 '자기가 읽은 글자'를 앞 글자로 넣어서,
       앞에서 틀리면 뒤가 엉망이 되고 펴기망에 쓸 만한 신호가 안 감 → 1차 학습이 거의 안 됨(정답률 50.1→50.4%)"""
    m, tok = parseq.m, parseq.m.tokenizer
    tgt = tok.encode(labels, images.device)
    tin, tout = tgt[:, :-1], tgt[:, 1:]
    L, B = tin.shape[1], tin.shape[0]
    mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=images.device), 1)
    pad_mask = (tin == tok.pad_id) | (tin == tok.eos_id)
    mem = m.model.encode(images)
    out = m.model.decode(tin, mem, mask, pad_mask, tgt_query=m.model.pos_queries[:, :L].expand(B, -1, -1),
                         tgt_query_mask=mask)
    lp = m.model.head(out).log_softmax(-1)
    valid = tout != tok.pad_id
    tout = torch.where(valid, tout, torch.full_like(tout, tok.eos_id))
    a = lp.gather(-1, tout[..., None])[..., 0]
    alt_id = alt[tout]
    b = lp.gather(-1, alt_id[..., None])[..., 0]
    tok_lp = torch.where(alt_id != tout, torch.logaddexp(a, b), a)
    return -(tok_lp * valid).sum() / valid.sum()


def train_tps(a, dev):
    t0 = time.time()
    xtr, ttr = generate(_gen_tps, a.n_train, 3)
    xva, tva = generate(_gen_tps, a.n_val, 4)
    ftr = torch.from_numpy(np.stack([f for _, f in ttr])).to(dev)           # 기준점 정답 (N,20,2)
    ttr, tva = [t for t, _ in ttr], [t for t, _ in tva]
    print(f"가짜 사진: 학습 {len(xtr)} / 검증 {len(xva)} ({time.time() - t0:.0f}s)")
    parseq = base.PARSeq()
    for p in parseq.m.parameters():
        p.requires_grad_(False)
    parseq.m.eval()
    m = al.TPSRectifier().to(dev)
    print(f"  처음(가운데 띠 그대로) 검증 정답률: {eval_tps(m, parseq, xva, tva, dev):.3f}")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    alt = case_alt(parseq.m.tokenizer, dev)
    log, tl = [], 0.0
    for s in range(1, a.steps + 1):
        m.train()
        idx = rng.integers(0, len(xtr), a.bs)
        x = torch.from_numpy(xtr[idx].astype(np.float32) / 255)[:, None].to(dev)
        rect, Cp = m(x)
        loss = tf_loss(parseq, al.to_parseq(rect), [ttr[i] for i in idx], alt) \
            + a.fid_w * (Cp - ftr[torch.from_numpy(idx).to(dev)]).abs().mean()   # 기준점 위치를 직접 가르침
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        sch.step()
        tl += loss.item()
        if s % a.eval_every == 0 or s == a.steps:
            acc = eval_tps(m, parseq, xva, tva, dev)
            log.append({"step": s, "loss": round(tl / a.eval_every, 4), "val_acc": acc})
            print(f"  step {s}/{a.steps} loss {tl / a.eval_every:.3f} | 검증 정답률 {acc:.3f}  {time.time() - t0:.0f}s")
            tl = 0.0
    return m, log


def eval_tps(m, parseq, x, t, dev):
    m.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(x), 128):
            xb = torch.from_numpy(x[i:i + 128].astype(np.float32) / 255)[:, None].to(dev)
            rect, _ = m(xb)
            labels, _ = parseq.m.tokenizer.decode(parseq.m(al.to_parseq(rect)).softmax(-1))
            ok += sum(base.norm(l) == tt for l, tt in zip(labels, t[i:i + 128]))
    return ok / len(x)


def eval_tps_layouts(a, dev):
    """합성 검증표: 학습 전(가운데 띠 그대로) vs 학습 후, 곧은 글자 / 휜 글자로 나눠서 (학습과 겹치지 않는 시드 9)"""
    rng = np.random.default_rng(9)
    xs, ts, lays = [], [], []
    for i in range(a.n_val):
        lay = "arc" if i % 2 else "line"                                       # 반반
        g, t, _ = make_pill(rng, al.TPS_IN, phi=float(rng.uniform(-20, 20)), layout=lay)
        xs.append(g), ts.append(t), lays.append(lay)
    x = np.stack(xs)
    parseq = base.PARSeq()
    before = al.TPSRectifier().to(dev)
    after = al.TPSRectifier().to(dev)
    after.load_state_dict(torch.load(os.path.join(al.MODEL_DIR, "tps_stn.pt"), map_location=dev))
    rows = []
    for name, m in (("학습 전 (가운데 띠 그대로)", before), ("학습 후", after)):
        m.eval()
        ok = []
        with torch.no_grad():
            for i in range(0, len(x), 128):
                xb = torch.from_numpy(x[i:i + 128].astype(np.float32) / 255)[:, None].to(dev)
                rect, _ = m(xb)
                labels, _ = parseq.m.tokenizer.decode(parseq.m(al.to_parseq(rect)).softmax(-1))
                ok += [base.norm(l) == t for l, t in zip(labels, ts[i:i + 128])]
        ok = np.array(ok)
        lay = np.array(lays)
        rows.append({"model": name, "곧은 글자": round(ok[lay == "line"].mean() * 100, 1),
                     "휜 글자": round(ok[lay == "arc"].mean() * 100, 1), "전체": round(ok.mean() * 100, 1),
                     "n_line": int((lay == "line").sum()), "n_arc": int((lay == "arc").sum())})
    print(f"\n[⑥ 합성 검증표] 가짜 사진 {len(x)}장 (곧은 {rows[0]['n_line']} / 휜 {rows[0]['n_arc']}), 기울기 ±20°")
    print(f"  {'':<26} {'곧은 글자':>8} {'휜 글자':>8} {'전체':>8}")
    for r in rows:
        print(f"  {r['model']:<26} {r['곧은 글자']:7.1f}% {r['휜 글자']:7.1f}% {r['전체']:7.1f}%")
    with open(os.path.join(al.MODEL_DIR, "tps_synth_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cnn", "tps", "tps_eval"])
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8, help="⑤")
    ap.add_argument("--steps", type=int, default=4000, help="⑥")
    ap.add_argument("--eval-every", type=int, default=500, help="⑥")
    ap.add_argument("--fid-w", type=float, default=5.0, help="⑥ 기준점 위치 손실의 무게 (0 이면 읽기 손실만)")
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(al.MODEL_DIR, exist_ok=True)
    torch.manual_seed(0)
    if a.task == "tps_eval":
        eval_tps_layouts(a, dev)
        return
    if a.task == "cnn":
        a.n_train, a.bs, a.lr = a.n_train or 30000, a.bs or 128, a.lr or 1e-3
        m, log = train_cnn(a, dev)
        name = "angle_cnn"
    else:
        a.n_train, a.bs, a.lr = a.n_train or 40000, a.bs or 48, a.lr or 5e-4
        m, log = train_tps(a, dev)
        name = "tps_stn"
    torch.save(m.state_dict(), os.path.join(al.MODEL_DIR, f"{name}.pt"))
    with open(os.path.join(al.MODEL_DIR, f"{name}_log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"저장 완료: {os.path.join(al.MODEL_DIR, name + '.pt')}")


if __name__ == "__main__":
    main()
