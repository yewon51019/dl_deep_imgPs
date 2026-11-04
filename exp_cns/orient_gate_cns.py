# =====================================================================
#  orient_gate_cns.py  —  '판별기 점수 차이가 δ 보다 클 때만 반대 방향으로 바꾸기' (2단계 수정판)
# =====================================================================
#
#  [왜]  --pick orient(무조건 판별기 따르기)가 개발용 읽기를 68.2 → 64.1 로 떨어뜨림: 판별기가 새 약(특히 긴 알약)에서
#        조각 77~88% 정도라, 확신도 방식이 이미 맞힌 사진도 뒤집음. 그래서 판별기가 '확실히' 반대쪽이라고 할 때만 바꿈
#  [방법]  같은 사진의 두 결과(확신도 방식 pred.csv / --pick orient pred.csv)를 섞음:
#          orient 결과가 바꿨고(orient_switch=1) 점수 차이(orient_best − orient_opp) > δ 이면 orient 행, 아니면 확신도 행
#          δ 는 검증 부분(trainval 828장)의 읽기 정답률로 고름 (같으면 큰 δ = 덜 바꾸는 쪽), 개발용은 측정만
#  [저장]  ocr_v1\<dev|trainval>\paddle_ft1_u2.0\orig_alignlong<tail>_pgate\pred.csv  + 같은 폴더 gate.csv (δ 별 검증 정답률)
#  [실행]  python exp_cns\orient_gate_cns.py --tail _line        (사전학습 PARSeq)
#          python exp_cns\orient_gate_cns.py --tail _line_pft1   (파인튜닝 PARSeq)
# =====================================================================

import argparse
import csv
import os

import common_cns as cc

D = cc.RESULTS / "ocr_v1" / "{split}" / "paddle_ft1_u2.0"
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]   # 1.01 = 한 번도 안 바꿈 (= 확신도 방식)


def load(split, tail):
    d = cc.RESULTS / "ocr_v1" / split / "paddle_ft1_u2.0"
    rd = lambda n: {r["file_name"]: r for r in csv.DictReader(open(d / n / "pred.csv", encoding="utf-8-sig"))}
    return d, rd(f"orig_alignlong{tail}"), rd(f"orig_alignlong{tail}_porient")


def mix(conf, orient, delta):
    out = []
    for fn, r in conf.items():
        o = orient[fn]
        sw = o["orient_switch"] == "1" and float(o["orient_best"]) - float(o["orient_opp"]) > delta
        row = dict(o if sw else r)
        row.update(orient_switch=int(sw), orient_best=o["orient_best"], orient_opp=o["orient_opp"], gate=delta)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", required=True, choices=["_line", "_line_pft1"])
    a = ap.parse_args()
    _, vc, vo = load("trainval", a.tail)
    accs = {dl: 100 * sum(r["correct"] == "1" for r in mix(vc, vo, dl)) / len(vc) for dl in DELTAS}
    best = max(DELTAS, key=lambda dl: (round(accs[dl], 6), dl))
    print(f"[{a.tail}] 검증 읽기 정답률:", {f"{k:g}": round(v, 1) for k, v in accs.items()}, f"→ δ = {best:g}")
    for split in ("trainval", "dev"):
        d, c, o = load(split, a.tail)
        rows = mix(c, o, best)
        od = d / f"orig_alignlong{a.tail}_pgate"
        os.makedirs(od, exist_ok=True)
        keys = list(rows[0])
        with open(od / "pred.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        n_sw = sum(r["orient_switch"] for r in rows)
        acc = 100 * sum(r["correct"] == "1" for r in rows) / len(rows)
        base = 100 * sum(r["correct"] == "1" for r in c.values()) / len(c)
        better = sum(r["correct"] == "1" and c[r["file_name"]]["correct"] == "0" for r in rows)
        worse = sum(r["correct"] == "0" and c[r["file_name"]]["correct"] == "1" for r in rows)
        print(f"  {split:8s} 바꾼 사진 {n_sw}  읽기 {base:.1f} → {acc:.1f}  (+{better} / −{worse})")
        if split == "trainval":
            with open(od / "gate.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["delta", "val_acc"])
                w.writerows([[k, round(v, 2)] for k, v in accs.items()])


if __name__ == "__main__":
    main()
