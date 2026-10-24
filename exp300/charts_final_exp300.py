# =====================================================================
#  charts_final_exp300.py  —  최종 보고용 비교 그래프 (PNG, 엑셀에도 붙임)
# =====================================================================
#   1) 정확도: 개발용 vs 평가용 (최종 4가지)        2) F1: 개발용 vs 평가용 (최종 4가지)  ← 두 지표는 그래프를 나눔 (축 하나)
#   3) 개발용 모든 방식 정확도 (정렬 13가지 + 색·합친 방식)   4) 평가용 남은 오류 원인 (방식별 쌓은 막대, 숫자 직접 표시)
#   색: 기본 팔레트 범주색 1(파랑)=개발용, 2(주황)=평가용 — validate_palette.js 통과. 오류 원인 6색은 대비 경고가 있어 숫자를 막대에 적음
#  [실행 예]  python exp300\charts_final_exp300.py --final exp300\results\test_final --dev-all exp300\results\combined_cv410\combined_metrics.csv ^
#             --out exp300\results\viz_final_report\charts
# =====================================================================

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
DEV, TEST = "#2a78d6", "#eb6834"
CAUSE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LABELS = {"V3+색": "보정 V3 + 색 (주 방식)", "C4": "C4: V3→색→dbscore", "A": "길 A: ②+박스 순서", "vote": "12방향 다수결 (지금)"}
ORDER = ["V3+색", "C4", "A", "vote"]
ERROR_ORDER = ["각인 같은 약", "마크뿐", "라벨 앞뒤 뒤바뀜", "각인 일부만 보임", "박스 못 찾음", "잘못 읽음"]


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2)


def dev_vs_test(side, metric, path, title):
    rows = {r["method"]: r for r in side if r["group"] == "전체"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    h = 0.36
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        for dy, key, col, name in ((h / 2 + 0.02, f"{metric}_개발", DEV, "개발용 51장"), (-h / 2 - 0.02, f"{metric}_평가", TEST, "평가용 237장")):
            v = float(rows[m][key])
            ax.barh(y + dy, v, height=h, color=col, edgecolor=SURFACE, linewidth=2, label=name if i == 0 else None)
            ax.text(v + 0.8, y + dy, f"{v:.1f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[m] for m in reversed(ORDER)], color=INK)
    ax.set_xlim(0, 105)
    style(ax, f"{metric} (%)")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def dev_all(rows, path):
    names = {"vote": "12방향 다수결", "conf": "② 확신도", "box": "① CRAFT+③", "pbox": "① Paddle+③", "pbox_cls": "① Paddle+분류기",
             "stroke": "③ 획 방향", "polar": "④ 극좌표", "polar_conf": "④+②", "cnn": "⑤ 각도 CNN", "tps": "⑥ TPS",
             "conf_perm": "길 A", "two_conf": "두 길(확신도)", "two_db": "두 길(dbscore)", "A+색": "길 A + 색", "V3": "보정 V3",
             "V3+색": "보정 V3 + 색 (주 방식)", "db": "dbscore", "db+색 δ=0.05": "dbscore + 색", "C1 δ=0.05": "C1", "C2 δ=0.05": "C2",
             "C3": "C3", "C4": "C4"}
    sel = [r for r in rows if r["group"] == "전체" and r["method"] in names]
    sel = list({r["method"]: r for r in sel}.values())
    fig, ax = plt.subplots(figsize=(8.5, 7.2), facecolor=SURFACE)
    for i, r in enumerate(sel):
        y = len(sel) - 1 - i
        v = float(r["정확도"])
        ax.barh(y, v, height=0.62, color=DEV, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.8, y, f"{v:.1f}", va="center", color=INK, fontsize=9.5,
                fontweight="bold" if r["method"] == "V3+색" else "normal")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([names[r["method"]] for r in reversed(sel)], color=INK, fontsize=10)
    for t in ax.get_yticklabels():
        if "주 방식" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlim(0, 100)
    style(ax, "정확도 (%)")
    ax.set_title("개발용 51장: 모든 방식의 약 정확도 (같은 환경)", loc="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def errors(etab, path):
    rows = {r["method"]: r for r in etab if r["split"] == "평가용"}
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor=SURFACE)
    for i, m in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        left = 0
        for c, col in zip(ERROR_ORDER, CAUSE_COLORS):
            v = int(rows[m][c])
            if v:
                ax.barh(y, v, left=left, height=0.6, color=col, edgecolor=SURFACE, linewidth=2,
                        label=c if c not in ax.get_legend_handles_labels()[1] else None)
                if v >= 3:
                    ax.text(left + v / 2, y, str(v), va="center", ha="center", color=INK, fontsize=9.5)
                left += v
        ax.text(left + 0.8, y, f"합계 {left}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[m] for m in reversed(ORDER)], color=INK)
    style(ax, "틀린 사진 수 (평가용 237장 중)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, labelcolor=INK, fontsize=10)
    ax.set_title("평가용 남은 오류 원인", loc="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="test_final 폴더 (dev_vs_test.csv, error_table.csv)")
    ap.add_argument("--dev-all", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    side = load(os.path.join(a.final, "dev_vs_test.csv"))
    dev_vs_test(side, "정확도", os.path.join(a.out, "1_accuracy_dev_vs_test.png"), "약 정확도: 개발용 vs 평가용 (전체)")
    dev_vs_test(side, "F1", os.path.join(a.out, "2_f1_dev_vs_test.png"), "F1: 개발용 vs 평가용 (전체)")
    dev_all(load(a.dev_all), os.path.join(a.out, "3_dev_all_methods_accuracy.png"))
    errors(load(os.path.join(a.final, "error_table.csv")), os.path.join(a.out, "4_test_error_causes.png"))
    print("저장 완료:", a.out)


if __name__ == "__main__":
    main()
