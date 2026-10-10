# =====================================================================
#  evaluate_exp300.py  —  predictions.csv 채점하기
# =====================================================================
#
#  [무엇을 계산하나요?]
#   (가) 약 판별  — "약제 ID를 맞혔나?"  ← 메인 지표
#        정확도(accuracy) = 맞힌 사진 수 / 전체 사진 수
#        정밀도·재현율·F1 = 약 종류마다 계산한 뒤 평균(macro)
#          · 정밀도: "A약이라고 판정한 것 중 진짜 A약인 비율"
#          · 재현율: "진짜 A약 사진 중 A약이라고 맞힌 비율"
#          · F1    : 정밀도와 재현율의 조화평균
#        '판정 없음'은 틀린 것으로 셈 (재현율이 깎임)
#   (나) 각인 글자 — 보조 지표
#        정확일치율 = 읽은 각인이 정답과 완전히 같은 비율
#        글자 정확도 = 1 - (틀린 글자 수 / 정답 글자 수) 의 평균
#        후보 정답률 = 돌려 읽은 여러 결과 중에 정답이 하나라도 있었던 비율
#                      (높은데 정확일치가 낮으면 '고르는 방법'을 고치면 좋아질 여지가 큼)
#
#  [매칭 방식 3가지]  run_exp300.py 가 셋 다 저장해 둠
#    exact = 완전일치만, edit = 편집거리(모델 비교 기준), attr = 모양·색으로 좁힌 뒤 편집거리
#
#  [실행 예]
#    1단계 순위 + 상위 4개:  python evaluate_exp300.py --pred pred_dev.csv --top 4
#    2단계 최종 + 혼동행렬:  python evaluate_exp300.py --pred pred_test.csv --cm all
# =====================================================================

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

MATCH = {"exact": "pred_exact", "edit": "pred_edit", "attr": "pred_attr"}
NONE = "(판정없음)"


def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def drug_metrics(y_true, y_pred):
    """정확도 + macro 정밀도/재현율/F1 (약 종류 = 정답에 나온 약들)"""
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    P, R, F = [], [], []
    for c in sorted(set(y_true)):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(P)), float(np.mean(R)), float(np.mean(F))


def imprint_metrics(rows):
    exact = np.mean([r["pred_imprint"] == r["truth_imprint"] for r in rows])
    cacc = np.mean([max(0.0, 1 - levenshtein(r["pred_imprint"], r["truth_imprint"]) / max(len(r["truth_imprint"]), 1))
                    for r in rows])
    oracle = np.mean([r["truth_imprint"] in json.loads(r["candidates"]) for r in rows])
    empty = np.mean([r["pred_imprint"] == "" for r in rows])
    return float(exact), float(cacc), float(oracle), float(empty)


def pct(x):
    return f"{x * 100:5.1f}"


def save_confusion(rows, key, path_png, path_csv, title):
    """혼동행렬: 세로 = 진짜 약, 가로 = 판정한 약. 대각선이 진할수록 잘 맞힌 것"""
    y_t = [r["truth_drug"] for r in rows]
    y_p = [r[key] or NONE for r in rows]
    labels = sorted(set(y_t)) + sorted(set(y_p) - set(y_t) - {NONE}) + [NONE]
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(set(y_t)), len(labels)), int)
    rows_lab = sorted(set(y_t))
    for t, p in zip(y_t, y_p):
        M[rows_lab.index(t), idx[p]] += 1
    with open(path_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["진짜\\판정"] + labels)
        for i, t in enumerate(rows_lab):
            w.writerow([t] + list(M[i]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.38), max(5, len(rows_lab) * 0.36)))
        ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["NONE" if l == NONE else l for l in labels], rotation=90, fontsize=7)
        ax.set_yticks(range(len(rows_lab)))
        ax.set_yticklabels(rows_lab, fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_xlabel("predicted (drug_N)")
        ax.set_ylabel("true (drug_N)")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [안내] 혼동행렬 그림은 못 그렸어요({e}). CSV는 저장됨: {path_csv}")


def breakdown(rows, key, field):
    """조건별(제형, 앞뒷면, 카메라각도, 회전각) 정확도"""
    g = defaultdict(list)
    for r in rows:
        g[r[field]].append(r["truth_drug"] == r[key])
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}


def main():
    ap = argparse.ArgumentParser(description="predictions.csv 채점")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--rank-by", default="edit", choices=list(MATCH), help="순위를 매길 매칭 방식 (기본 edit)")
    ap.add_argument("--top", type=int, default=4, help="상위 몇 개 조합을 고를지")
    ap.add_argument("--cm", default="best", help="혼동행렬을 그릴 조합: best / all / easyocr:P2,...")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.pred, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[r["combo"]].append(r)

    # ---- 1) 조합별 점수표 ----
    table = []
    for combo, rs in by_combo.items():
        y_t = [r["truth_drug"] for r in rs]
        rec = {"combo": combo, "model": rs[0]["model"], "prep": rs[0]["prep"], "n": len(rs)}
        for m, col in MATCH.items():
            acc, p, r_, f1 = drug_metrics(y_t, [r[col] for r in rs])
            rec.update({f"{m}_acc": acc, f"{m}_P": p, f"{m}_R": r_, f"{m}_F1": f1})
        ex, ca, orc, emp = imprint_metrics(rs)
        rec.update({"imp_exact": ex, "imp_char": ca, "imp_oracle": orc, "empty": emp,
                    "ms": float(np.mean([int(r["ms"]) for r in rs]))})
        table.append(rec)
    rk = a.rank_by
    table.sort(key=lambda t: (t[f"{rk}_F1"], t[f"{rk}_acc"]), reverse=True)

    cols = list(table[0].keys()) if table else []
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})

    print(f"\n[조합별 점수]  (순위 기준: {rk} 매칭 F1, 단위 %)  사진 수 n")
    print(f"{'순위':>3} {'조합':<14}{'n':>4} | {'정확도':>6}{'정밀도':>6}{'재현율':>6}{'F1':>6} |"
          f" {'attr정확':>7}{'attrF1':>6} | {'각인일치':>7}{'글자정확':>7}{'후보정답':>7}{'빈결과':>6} | {'ms':>6}")
    for i, t in enumerate(table, 1):
        print(f"{i:>3} {t['combo']:<14}{t['n']:>4} | {pct(t[rk+'_acc'])} {pct(t[rk+'_P'])} {pct(t[rk+'_R'])} {pct(t[rk+'_F1'])} |"
              f"  {pct(t['attr_acc'])} {pct(t['attr_F1'])} |  {pct(t['imp_exact'])}  {pct(t['imp_char'])}  "
              f"{pct(t['imp_oracle'])} {pct(t['empty'])} | {t['ms']:6.0f}")

    top = [t["combo"] for t in table[: a.top]]
    with open(os.path.join(a.out_dir, "top_combos.txt"), "w") as f:
        f.write(",".join(top))
    print(f"\n상위 {a.top}개 조합: {','.join(top)}   (eval/top_combos.txt 에 저장 → 2단계 --combos 에 그대로 사용)")

    # ---- 2) 혼동행렬 + 조건별 분석 + 자주 틀린 각인 ----
    if a.cm == "best":
        targets = top[:1]
    elif a.cm == "all":
        targets = [t["combo"] for t in table]
    else:
        targets = [c.strip() for c in a.cm.split(",") if c.strip() in by_combo]
    for combo in targets:
        rs = by_combo[combo]
        safe = combo.replace(":", "_")
        key = MATCH[rk]
        save_confusion(rs, key, os.path.join(a.out_dir, f"cm_{safe}.png"), os.path.join(a.out_dir, f"cm_{safe}.csv"),
                       f"{combo} ({rk}) acc={np.mean([r['truth_drug'] == r[key] for r in rs]) * 100:.1f}%")
        print(f"\n[{combo}] 조건별 정확도 ({rk})")
        lines = []
        for field, name in [("form", "제형"), ("drug_dir", "앞/뒷면"), ("camera_la", "카메라각도"),
                            ("rot_deg", "알약회전"), ("det_shape", "판정모양")]:
            b = breakdown(rs, key, field)
            txt = ", ".join(f"{k}={v[0] * 100:.0f}%({v[1]})" for k, v in b.items())
            print(f"  - {name}: {txt}")
            lines.append([name, txt])
        wrong = Counter((r["truth_imprint"], r["pred_imprint"] or "(빈칸)") for r in rs
                        if r["pred_imprint"] != r["truth_imprint"])
        print("  - 자주 틀린 각인 (정답 → 읽은 것):",
              ", ".join(f"{t}→{p}×{n}" for (t, p), n in wrong.most_common(8)) or "없음")
        with open(os.path.join(a.out_dir, f"breakdown_{safe}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerows(lines)
            w.writerow([])
            w.writerow(["정답 각인", "읽은 각인", "횟수"])
            for (t, p), n in wrong.most_common():
                w.writerow([t, p, n])
    print(f"\n저장 폴더: {a.out_dir}/  (summary.csv, top_combos.txt, cm_*.png, breakdown_*.csv)")


if __name__ == "__main__":
    main()
