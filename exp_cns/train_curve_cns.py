# =====================================================================
#  train_curve_cns.py  —  검출기 파인튜닝 로그 → 에폭별 학습 loss·검증 hmean 그림 (CLAUDE.md 10장 2-5)
# =====================================================================
#
#  [읽는 것]  exp_cns\results\<run>\train_stdout.log  (PaddleOCR tools\train.py 출력)
#             - 'epoch: [e/50] ... loss: x'  줄 → 에폭마다 평균 loss
#             - 'cur metric, precision: p, recall: r, hmean: h' 줄 → 검증 결과 (바로 앞 학습 줄의 에폭으로 붙임)
#  [저장]    exp_cns\results\<run>\train_curve.png  (숫자만 있는 그림)
#            exp_cns\results\<run>\train_curve.csv  (epoch, loss, precision, recall, hmean)
#  [실행]    python exp_cns\train_curve_cns.py --run det_ft2
# =====================================================================

import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_cns as cc

RE_STEP = re.compile(r"epoch: \[(\d+)/\d+\].*?\bloss: ([\d.]+)")
RE_EVAL = re.compile(r"cur metric, precision: ([\d.]+), recall: ([\d.]+), hmean: ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="det_ft2", help="results 아래 학습 폴더 이름")
    a = ap.parse_args()
    d = cc.RESULTS / a.run
    loss, ev, ep = {}, {}, 0
    for line in open(d / "train_stdout.log", encoding="utf-8", errors="ignore"):
        m = RE_STEP.search(line)
        if m:
            ep = int(m.group(1))
            loss.setdefault(ep, []).append(float(m.group(2)))
            continue
        m = RE_EVAL.search(line)
        if m and ep:
            ev[ep] = tuple(float(x) for x in m.groups())          # 검증은 에폭 끝마다 한 번 (eval_batch_step = 한 에폭 걸음 수)
    eps = sorted(loss)
    with open(d / "train_curve.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "precision", "recall", "hmean"])
        for e in eps:
            p, r, h = ev.get(e, ("", "", ""))
            w.writerow([e, round(sum(loss[e]) / len(loss[e]), 4), p, r, h])
    best = max(ev, key=lambda e: ev[e][2])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(eps, [sum(loss[e]) / len(loss[e]) for e in eps], color="#3b6fb6")
    ax[0].set(title="train loss (epoch mean)", xlabel="epoch", ylabel="loss")
    ee = sorted(ev)
    for i, (name, c) in enumerate((("precision", "#999999"), ("recall", "#c9a227"), ("hmean", "#c0392b"))):
        ax[1].plot(ee, [ev[e][i] for e in ee], label=name, color=c, lw=2 if name == "hmean" else 1)
    ax[1].axvline(best, ls="--", color="#c0392b", lw=0.8)
    ax[1].set(title=f"val ({a.run}) - best hmean {ev[best][2]:.3f} @ epoch {best}", xlabel="epoch", ylim=(0.4, 1.0))
    ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "train_curve.png", dpi=110)
    print(f"저장: {d / 'train_curve.png'}  / 최고 hmean {ev[best][2]:.4f} (에폭 {best}), 마지막 {ev[ee[-1]][2]:.4f} (에폭 {ee[-1]})")


if __name__ == "__main__":
    main()
