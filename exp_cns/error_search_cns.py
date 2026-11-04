# =====================================================================
#  error_search_cns.py  —  약 판별 오답 그림: 답이 어떤 읽기·DB 각인에서 나왔는지 사진마다 보기 (pft1, OCR 안 돌림)
# =====================================================================
#
#  [대상]  기준 방식 pft1 (λ 0.9, w 0.02) 에서 약을 틀린 사진 (틀림 + 판정 없음). 검증 828장, 개발 468장. 평가용 사진·결과는 읽지 않음
#          저장된 답과 사진마다 같은지 확인 (display_cns.score)
#  [칸]    (CLAUDE.md 3-17 표와 같음, 앞 칸부터)
#     틀림_짧은각인    : 답 약에서 일치한 DB 각인이 2자 이하
#     틀림_그밖
#     없음_정답포함동점 / 없음_정답빠진동점 / 없음_0.3넘는약없음
#  [근거]  답 약·정답 약마다 T 를 만든 (점수, 방향, 읽기 후보, DB 각인). 방향 = 고른 / 다른 (다른 방향은 × 0.9 후 점수)
#  [저장]  exp_cns\results\match_v7\errors_pft1\
#            error_list.csv, dev_sheet_NN.png, val_sheet_NN.png (한 장에 6장, 칸·약 순서. PNG 는 .gitignore)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\error_search_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv
import json
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_cns as cc
import compare_pft1_cns as cp
import display_cns as dc
import match2_cns as m2
import match_cns as m1
import ocr_common_cns as oc
from correct_exp300 import MIN_SIM, sim_v

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = cc.RESULTS / "match_v7" / "errors_pft1"
LAM, PER = 0.9, 6
ORDER = ["틀림_짧은각인", "틀림_그밖", "없음_정답포함동점", "없음_정답빠진동점", "없음_0.3넘는약없음"]
SPLIT = {"dev": "dev", "val": "trainval"}
PART_NAME = {"dev": "개발용", "val": "검증"}
EPS = 1e-9


def evidence(cm, co, imprints):
    """이 약의 T 를 만든 (점수, 방향, 읽기, 각인). 0.3 미만이면 None."""
    best = None
    for cands, lam, way in ((cm, 1.0, "고른"), (co, LAM, "다른")):
        for c in cands:
            for imp in imprints:
                s = sim_v(c, imp, *m1.OPT)
                if s >= MIN_SIM and lam * s >= MIN_SIM and (best is None or lam * s > best[0] + EPS):
                    best = (lam * s, way, c, imp)
    return best


def ev_text(ev):
    return f"{ev[0]:.2f} ({ev[1]} 방향 '{ev[2]}' ↔ 각인 '{ev[3]}')" if ev else "각인 점수 0"


def classify(r, pred, ranked, ev_pred, any_T):
    if pred:
        return "틀림_짧은각인" if ev_pred and len(ev_pred[3]) <= 2 else "틀림_그밖"
    if not any_T:
        return "없음_0.3넘는약없음"
    tied = [d for d, v in ranked if v >= ranked[0][1] - EPS]
    return "없음_정답포함동점" if r["drug_N"] in tied else "없음_정답빠진동점"


def collect(part, db):
    byid = {d["drug_N"]: d for d in db}
    out = []
    for r, pred, ranked in dc.score("pft1", part, db):
        if pred == r["drug_N"]:
            continue
        cm, co = m2.split_candidates(r, cp.MAX_LEN)
        ev_pred = evidence(cm, co, byid[pred]["imprints"]) if pred else None
        ev_top = evidence(cm, co, byid[ranked[0][0]]["imprints"])
        ev_me = evidence(cm, co, byid[r["drug_N"]]["imprints"])
        any_T = bool(m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM))
        s1 = ranked[0][1]
        tied = [d for d, v in ranked if v >= s1 - EPS]
        me_score = next(v for d, v in ranked if d == r["drug_N"])
        worst = sum(v >= me_score - EPS for _, v in ranked)
        show = lambda d: f"{byid[d]['name'][:16]} [{'/'.join(byid[d]['imprints']) or '-'}]"
        out.append({"part": part, "cause": classify(r, pred, ranked, ev_pred, any_T), "file_name": r["file_name"],
                    "drug_N": r["drug_N"], "name": byid[r["drug_N"]]["name"], "truth": r["truth"],
                    "faces": "/".join(byid[r["drug_N"]]["imprints"]), "read": r["read"], "conf": r["conf"],
                    "angle": r["angle"], "n_boxes": r["n_boxes"], "fallback": r["fallback"], "camera": r["camera_la"], "aligned": r["aligned"],
                    "pred": pred, "pred_name": show(pred) if pred else "", "pred_evidence": ev_text(ev_pred) if pred else "",
                    "top_evidence": ev_text(ev_top), "truth_evidence": ev_text(ev_me), "truth_rank": worst, "n_tied": len(tied),
                    "top3": " | ".join(f"{show(d)} {v:.3f}" for d, v in ranked[:3]),
                    "from_other": int(bool(ev_pred and ev_pred[1] == "다른")),
                    "all_dirs": r["all_dirs"], "best_boxes": r["best_boxes"]})
    out.sort(key=lambda x: (ORDER.index(x["cause"]), x["drug_N"], x["file_name"]))
    return out


def picture(x, path):
    """왼쪽 원본 | 오른쪽 고른 방향 + 상자·상자별 글자 (dt 그림과 같은 모양)."""
    img = cc.fit_side(cc.imread(path), 400)
    rot = oc.rotate_color(img, float(x["angle"])).copy()
    for b in json.loads(x["best_boxes"] or "[]"):
        cv2.rectangle(rot, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 2)
        cv2.putText(rot, b[4] or "-", (b[0], max(14, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    s = 400 / max(rot.shape[:2])
    rot = cc.fit_side(cv2.resize(rot, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 400)
    return cv2.cvtColor(np.hstack([img, rot]), cv2.COLOR_BGR2RGB)


def sheets(part, rows, paths):
    n = (len(rows) + PER - 1) // PER
    for s in range(n):
        cur = rows[s * PER:(s + 1) * PER]
        fig, ax = plt.subplots(PER, 2, figsize=(15, 2.75 * PER), gridspec_kw={"width_ratios": [1, 1.05]})
        for i in range(PER):
            ax[i][0].axis("off")
            ax[i][1].axis("off")
            if i >= len(cur):
                continue
            x = cur[i]
            ax[i][0].imshow(picture(x, paths[x["file_name"]]))
            dirs = " ".join(f"{d.split(':')[0]}:{d.split(':')[1] or '-'}" for d in x["all_dirs"].split(" | "))
            answer = f"답: {x['pred_name']}\n  근거 {x['pred_evidence']}" if x["pred"] else f"답: 판정 없음 (1등 동점 {x['n_tied']}개)\n  1등 근거 {x['top_evidence']}"
            txt = (f"#{s * PER + i + 1} [{x['cause']}]  {x['file_name'][:28]}\n"
                   f"정답 약: {x['name'][:20]}  (사진 면 {x['truth']} / 두 면 {x['faces']})\n"
                   f"읽음: {x['read'] or '(없음)'}  (확신도 {x['conf']}, 상자 {x['n_boxes']}개, 카메라 {x['camera']})\n"
                   f"{answer}\n"
                   f"정답 약 점수: {x['truth_evidence']}, 등수 {x['truth_rank']}\n"
                   + "\n".join(textwrap.wrap("방향별 읽기: " + dirs, 92)))
            ax[i][1].text(0, 0.98, txt, fontsize=9.5, va="top", ha="left", transform=ax[i][1].transAxes)
        fig.suptitle(f"약 판별 오답 (pft1) {PART_NAME[part]} {s + 1}/{n}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"{part}_sheet_{s + 1:02d}.png", dpi=85)
        plt.close(fig)
    print(f"  {part} 그림 {n}장", flush=True)


def main():
    if (OUT / "error_list.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    allrows = []
    for part in ("dev", "val"):
        rows = collect(part, db)
        c = collections.Counter(x["cause"] for x in rows)
        print(part, len(rows), {k: c[k] for k in ORDER}, "틀림 중 다른 방향 근거", sum(x["from_other"] for x in rows), flush=True)
        for k in ORDER:
            d = collections.Counter(x["drug_N"] for x in rows if x["cause"] == k)
            print("  ", k, d.most_common())
        allrows += rows
        paths = {it["file_name"]: it["image_path"] for it in oc.load_split(SPLIT[part], 0)}
        sheets(part, rows, paths)
    with open(OUT / "error_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["no"] + list(allrows[0]))
        w.writeheader()
        for part in ("dev", "val"):
            for i, x in enumerate([y for y in allrows if y["part"] == part], 1):
                w.writerow({"no": i, **x})
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
