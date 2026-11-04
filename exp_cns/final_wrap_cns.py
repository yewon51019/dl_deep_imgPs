# =====================================================================
#  final_wrap_cns.py  —  보고서용 마무리: 사진 2장 지표 · 결과 표시 흐름 · 최종 방식 dt 저장 (OCR 안 돌림, 방식을 고르지 않음)
# =====================================================================
#
#  [최종 방식, 사용자 결정 2026-11-02, 사후]  pft1 (λ 0.9, w 0.02, TF3·SIM·SUP 없음). 공식 평가 결과는 사전 기록한 pft1 + TF3 그대로
#  [--step pairs]  match_v6c\pred_pairs.csv → pair_metrics.csv: 2장 / 1장씩 정확도, macro 정밀도·재현율·F1, 답 낸 사진 중 맞은 비율
#  [--step flow]   표시 흐름 (실행 전 고정): 확실 = 약 1개 / 불확실·동점이고 후보 5개 이하 = 후보 표시 / 5개 초과 = 다른 면 사진·다시 찍기 요청
#                  → display_v3\flow.csv (검증·개발·최종 평가)
#  [--step dt]     저장된 pft1 dt 그림 + 오른쪽 칸(약 답, 표시 후보, 1~3등, 공식 주 방식 답)
#                  → final_method_dt\test\, dev\ (AI Hub 사진이 들어 있어 .gitignore, 공개하지 않음. 그림 글자는 영문만)
#  [저장]  각 결과가 이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\final_wrap_cns.py --step pairs|flow|dt      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import argparse
import collections
import csv

import cv2
import numpy as np

import common_cns as cc
import display_cns as dc
import match7_cns as m7
import match_cns as m1
import ocr_common_cns as oc
import rerank_tf_cns as rt
import test_final2_cns as tf2
from test_final_cns import macro

RES = cc.RESULTS
CAP = 5
READ_DT = oc.OCR_OUT / "{}" / rt.DET / "orig_alignlong_line_pft1" / "dt"


def pair_line(group, rows):
    """같은 쌍에서 2장 판별과 1장씩 판별의 지표 한 줄."""
    drugs = sorted({r["drug_N"] for r in rows})
    n = len(rows)
    pairs = [(r["drug_N"], r["pair_pred"]) for r in rows]
    single = [(r["drug_N"], r[k]) for r in rows for k in ("single_front_pred", "single_back_pred")]
    out = {"group": group, "drugs": len(drugs), "pairs": n}
    for tag, xs in (("two", pairs), ("one", single)):
        hit = sum(t == p for t, p in xs)
        answered = sum(bool(p) for _, p in xs)
        P, R, F = macro(xs, drugs)
        out.update({f"{tag}_n": len(xs), f"{tag}_acc": round(100 * hit / len(xs), 1), f"{tag}_hit": hit, f"{tag}_none": len(xs) - answered,
                    f"{tag}_answered_right": round(100 * hit / max(answered, 1), 1), f"{tag}_macro_P": P, f"{tag}_macro_R": R, f"{tag}_macro_F1": F})
    return out


def step_pairs():
    path = RES / "match_v6c" / "pair_metrics.csv"
    if path.exists():
        raise SystemExit(f"이미 있음: {path}")
    rows = list(csv.DictReader(open(RES / "match_v6c" / "pred_pairs.csv", encoding="utf-8-sig")))
    val = [r for r in rows if r["part"] == "val"]
    dev = [r for r in rows if r["part"] == "dev"]
    clean = [r for r in dev if r["drug_N"] != "K-000147"]
    table = [pair_line("검증", val), pair_line("개발", dev), pair_line("개발 (K-000147 제외)", clean),
             pair_line("검증 + 개발", rows), pair_line("검증 + 개발 (K-000147 제외)", val + clean)]
    m7.write_csv(path, table)
    for t in table:
        print(t)


def flow_line(part, group, items):
    n = len(items)
    pct = lambda v, m=n: round(100 * v / max(m, 1), 1)
    cats = collections.defaultdict(list)
    for r, pred, ranked in items:
        kind, sh = dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)
        cat = "확정" if kind == "sure" else ("후보" if len(sh) <= CAP else "요청")
        cats[cat].append(r["drug_N"] in sh)
    shown_ok = sum(cats["확정"]) + sum(cats["후보"])
    return {"part": part, "group": group, "n": n, "sure": len(cats["확정"]), "sure_share": pct(len(cats["확정"])), "sure_right": sum(cats["확정"]), "sure_wrong": len(cats["확정"]) - sum(cats["확정"]),
            "cands": len(cats["후보"]), "cands_share": pct(len(cats["후보"])), "cands_include": pct(sum(cats["후보"]), len(cats["후보"])),
            "request": len(cats["요청"]), "request_share": pct(len(cats["요청"])), "shown_include_all": pct(shown_ok)}


def test_items(db):
    return tf2.base_scores("pft1", db, tf2.load_probs())


def step_flow():
    out = RES / "display_v3"
    if (out / "flow.csv").exists():
        raise SystemExit(f"이미 있음: {out}")
    out.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    table = []
    for part in ("val", "dev"):
        items = dc.score("pft1", part, db)
        table.append(flow_line(part, "전체", items))
        if part == "dev":
            table.append(flow_line(part, "K-000147 제외", [x for x in items if x[0]["drug_N"] != "K-000147"]))
    table.append(flow_line("test", "전체", test_items(db)))
    m7.write_csv(out / "flow.csv", table)
    for t in table:
        print(t)


def ascii_only(s):
    return "".join(c if ord(c) < 128 else "?" for c in s)


def panel(lines, h, width=430):
    p = np.full((h, width, 3), 255, np.uint8)
    for i, (t, col) in enumerate(lines):
        cv2.putText(p, ascii_only(t)[:44], (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return p


def step_dt():
    db = m1.load_db()
    byid = {d["drug_N"]: d for d in db}
    label = lambda d: f"{d} [{'/'.join(byid[d]['imprints']) or '-'}]"
    official = {r["file_name"]: r["main"] for r in csv.DictReader(open(RES / "test_final_pillset" / "pred_test.csv", encoding="utf-8-sig"))}
    for part, split, items in (("test", "test", test_items(db)), ("dev", "dev", dc.score("pft1", "dev", db))):
        out = RES / "final_method_dt" / part
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"이미 있음: {out}")
        out.mkdir(parents=True, exist_ok=True)
        src = str(READ_DT).format(split)
        ok = 0
        for r, pred, ranked in items:
            fn = r["file_name"]
            im = cc.imread(f"{src}\\{fn}")
            kind, sh = dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)
            right = pred == r["drug_N"]
            ok += right
            top = {"sure": "one answer", "unsure": "show candidates", "tie": "tie"}[kind]
            lines = [(f"FINAL METHOD (pft1)", (0, 0, 0)), (f"truth drug: {label(r['drug_N'])}", (0, 0, 0)),
                     (f"answer: {label(pred) if pred else 'NO DECISION'}", (0, 160, 0) if right else (0, 0, 220)),
                     (f"display: {top}, {len(sh)} drugs" + ("  (ask other face)" if kind != "sure" and len(sh) > CAP else ""), (60, 60, 60))]
            lines += [(f"  {label(d)}", (0, 160, 0) if d == r["drug_N"] else (60, 60, 60)) for d in sh[:4]]
            lines += [(f"{i}. {d} {v:.3f}", (90, 90, 90)) for i, (d, v) in enumerate(ranked[:3], 1)]
            if part == "test":
                lines.append((f"official main (+TF3): {official[fn] or 'NO DECISION'}", (150, 90, 0)))
            cc.imwrite(str(out / fn), np.hstack([im, panel(lines, im.shape[0])]))
        print(f"{part}: {len(items)}장 저장, 맞음 {ok}, {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["pairs", "flow", "dt"])
    a = ap.parse_args()
    {"pairs": step_pairs, "flow": step_flow, "dt": step_dt}[a.step]()


if __name__ == "__main__":
    main()
