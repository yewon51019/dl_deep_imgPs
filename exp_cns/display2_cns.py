# =====================================================================
#  display2_cns.py  —  결과 표시 규칙에 TF3 붙이기: 사진마다 약 후보 몇 개를 보여 주고, 그 안에 정답이 있는가 (저장 점수로 계산, GPU·OCR 안 씀)
# =====================================================================
#
#  [공통, display_cns 와 같음]  확실 (답 있음, 1등 ≥ 0.95, 차이 ≥ 0.1) = 약 1개 / 불확실 = 1등 + 2등 + 2등과 동점인 약 / 동점 = 동점인 약 전부
#  [TF3 답을 낸 사진만 다름]  (기준이 판정 없음이고 TF 각인 1·2등 차이 ≥ 0.3)
#     D1 = 동점인 약 전부 + TF3 답 (맨 앞에 TF3 답)
#     D2 = TF3 답 + TF 2등 약 + 그 약과 같은 점수의 약
#     TF3 답은 '확실' 로 세지 않음 (후보와 함께 보여 줌)
#  [고르는 규칙, 실행 전 고정]  검증에서 D2 의 평균 표시 개수가 D1 보다 적고 정답 포함률 저하가 1%p 이하면 D2, 아니면 D1
#  [대상]  pft1, 검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\display_v2\  summary.csv, sure_wrong.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\display2_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import numpy as np

import display_cns as dc
import match7_cns as m7
import match_cns as m1
import rerank_tf_cns as rt

OUT = m1.cc.RESULTS / "display_v2"
GAP = 0.3
EPS = 1e-6


def items(part, db):
    """사진마다 dict(r, kind, base_shown, tf, d2_shown). tf = TF3 답 ('' = TF3 가 답을 내지 않음)."""
    z = np.load(rt.OUT / f"tf_scores_{part}.npz", allow_pickle=False)
    idx = {t: i for i, t in enumerate(z["texts"])}
    scores = dict(zip(z["names"], z["scores"]))
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out = []
    for r, pred, ranked in dc.score("pft1", part, db):
        kind, sh = dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)
        tf, d2 = "", None
        if not pred:
            row = scores[r["file_name"]]
            top2 = np.sort(row)[-2:]
            if float(top2[1] - top2[0]) >= GAP:
                tf, s = rt.tf_answer(row, idx, probs[r["file_name"]], db)
                if tf:
                    second = max(v for d, v in s.items() if d != tf)
                    d2 = [tf] + [d for d, v in s.items() if d != tf and v >= second - EPS]
        out.append({"r": r, "kind": kind, "shown": sh, "tf": tf, "d2": d2, "pred": pred})
    return out


def lists(x, rule):
    if not x["tf"]:
        return x["shown"]
    if rule == "base":
        return x["shown"]
    if rule == "D1":
        return [x["tf"]] + [d for d in x["shown"] if d != x["tf"]]
    return x["d2"]


def measure(its, rule):
    n = len(its)
    pct = lambda v, m=n: round(100 * v / max(m, 1), 1)
    has = lambda x: x["r"]["drug_N"] in lists(x, rule)
    sizes = [len(lists(x, rule)) for x in its]
    first = lambda x: x["tf"] if (x["tf"] and rule != "base") else x["pred"]
    sure = [x for x in its if x["kind"] == "sure"]
    tfp = [x for x in its if x["tf"]]
    return {"n": n, "include_truth": pct(sum(has(x) for x in its)), "top1": pct(sum(first(x) == x["r"]["drug_N"] for x in its)),
            "mean_shown": round(sum(sizes) / n, 2), "max_shown": max(sizes),
            "sure": len(sure), "sure_wrong": sum(not has(x) for x in sure),
            "tf3": len(tfp) if rule != "base" else 0, "tf3_include": pct(sum(has(x) for x in tfp), len(tfp)) if rule != "base" else "",
            "tf3_mean_shown": round(sum(len(lists(x, rule)) for x in tfp) / max(len(tfp), 1), 2) if rule != "base" else ""}


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    table, wrong = [], []
    for part in ("val", "dev"):
        its = items(part, db)
        groups = [("전체", its)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in its if x["r"]["drug_N"] != "K-000147"]))
        for gname, cur in groups:
            for rule in ("base", "D1", "D2"):
                table.append({"part": part, "group": gname, "rule": rule, **measure(cur, rule)})
                print(table[-1])
        for x in its:
            if x["kind"] == "sure" and x["r"]["drug_N"] not in x["shown"]:
                wrong.append({"part": part, "file_name": x["r"]["file_name"], "drug_N": x["r"]["drug_N"], "truth": x["r"]["truth"], "read": x["r"]["read"], "shown": x["pred"]})
    m7.write_csv(OUT / "summary.csv", table)
    if wrong:
        m7.write_csv(OUT / "sure_wrong.csv", wrong)
    get = lambda r: next(t for t in table if t["part"] == "val" and t["group"] == "전체" and t["rule"] == r)
    d1, d2 = get("D1"), get("D2")
    pick = "D2" if d2["mean_shown"] < d1["mean_shown"] and d1["include_truth"] - d2["include_truth"] <= 1.0 else "D1"
    print("검증으로 고른 표시 규칙:", pick)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
