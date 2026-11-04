# =====================================================================
#  vote_char_cns.py  —  글자 투표 규칙 V2 시험: 방향마다 많이 나온 글자에 비중 주기 (pft1 저장 읽기로 계산, OCR 안 돌림)
# =====================================================================
#
#  [기준]  pft1 (λ 0.9, w 0.02), 저장된 답과 사진마다 같은지 확인 (display_cns.score)
#  [V2, 실행 전 고정]
#     대상 = 12방향 사진(축 정렬 아님) 중 불확실한 사진 (display_cns 규칙: 1등 < 0.95 또는 1·2등 차이 < 0.1, 동점 포함)
#     글자 비중 = 그 글자가 나온 방향 수 ÷ 글자가 나온 읽기 수 (헷갈리는 글자 통일 fold 후, 한 읽기에서 한 번만 셈)
#     C(약) = 각인 글자 비중의 평균 (앞·뒷면 중 높은 쪽)
#     후보 = 각인 점수 T ≥ 0.3 인 약. 답 = T + c·C 1등. 같은 점수면 겉모습 점수 A, 그래도 같으면 판정 없음
#     후보가 없으면 기존 답 유지. c 는 0.1 / 0.2 / 0.5 중 검증 맞은 수로 고름 (같으면 작은 값). 개발용으로 고르지 않음
#  [채택 규칙, 실행 전 고정]  기준 대비 (좋아짐 − 나빠짐) 이 검증 +5장 이상, 개발 0장 이상
#  [대상]  검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\vote_v2\  summary.csv, grid_val.csv, changed.csv, pred_val.csv, pred_dev.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\vote_char_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import compare_pft1_cns as cp
import display_cns as dc
import match2_cns as m2
import match7_cns as m7
import match_cns as m1
from correct_exp300 import fold

OUT = m1.cc.RESULTS / "vote_v2"
LAM = 0.9
C_GRID = [0.1, 0.2, 0.5]
BETTER_MIN, DEV_MIN = 5, 0
EPS = 1e-9


def char_weights(r):
    """글자 → 그 글자가 나온 방향 비율."""
    reads = [fold(t) for t in (s.split(":")[1] for s in r["all_dirs"].split(" | ")) if t]
    n = collections.Counter(c for t in reads for c in set(t))
    return {c: v / len(reads) for c, v in n.items()}


def char_score(weights, imprints):
    return max((sum(weights.get(c, 0.0) for c in fold(x)) / len(x) for x in imprints if x), default=0.0)


def apply_rule(T, weights, prob, byid, c):
    s = {d: v + c * char_score(weights, byid[d]["imprints"]) for d, v in T.items()}
    top = max(s.values())
    same = [d for d, v in s.items() if v >= top - EPS]
    return same[0] if len(same) == 1 else m1.decide({d: m1.attr_score(prob, byid[d]) for d in same})[0]


def predict(part, db, c_values):
    byid = {d["drug_N"]: d for d in db}
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out, used = [], 0
    for r, pred, ranked in dc.score("pft1", part, db):
        res = {"base": (pred, ranked[:3]), **{f"V2_{c:g}": (pred, ranked[:3]) for c in c_values}}
        if r["aligned"] != "1" and dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)[0] != "sure":
            cm, co = m2.split_candidates(r, cp.MAX_LEN)
            T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM)
            if T:
                used += 1
                w = char_weights(r)
                for c in c_values:
                    res[f"V2_{c:g}"] = (apply_rule(T, w, probs[r["file_name"]], byid, c), ranked[:3])
        out.append((r, res))
    print(f"  {part} 규칙 적용 {used}장", flush=True)
    return out


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    val = predict("val", db, C_GRID)
    grid = [{"c": c, **m7.measure(val, f"V2_{c:g}")} for c in C_GRID]
    m7.write_csv(OUT / "grid_val.csv", grid)
    for g in grid:
        print(g)
    c = max(C_GRID, key=lambda x: (next(g["hit"] for g in grid if g["c"] == x), -x))
    print(f"검증으로 고른 c = {c:g}", flush=True)
    dev = predict("dev", db, [c])
    name = f"V2_{c:g}"
    table, changed = [], []
    for part, out in (("val", val), ("dev", dev)):
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for cfg in ("base", name):
                table.append({"part": part, "group": gname, "config": cfg, **m7.measure(its, cfg)})
        for r, p in out:
            k = m7.kind(r["drug_N"], p["base"][0], p[name][0])
            if k:
                changed.append({"part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                "all_dirs": r["all_dirs"], "change": k, "pred_base": p["base"][0], "pred_new": p[name][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "base_pred", "V2_pred"])
            for r, p in out:
                w.writerow([r["file_name"], r["drug_N"], r["truth"], r["read"], p["base"][0], p[name][0]])
    m7.write_csv(OUT / "summary.csv", table)
    if changed:
        m7.write_csv(OUT / "changed.csv", changed)
    for t in table:
        print(t)
    for x in sorted(changed, key=lambda x: (x["part"], x["change"], x["drug_N"])):
        print(x["part"], x["change"], x["drug_N"], x["truth"], "| 읽기", x["read"], "|", x["pred_base"] or "(없음)", "→", x["pred_new"] or "(없음)")
    get = lambda part: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == name)
    v, d = get("val"), get("dev")
    vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
    print(f"{name}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}) → {'통과' if vd >= BETTER_MIN and dd >= DEV_MIN else '미달'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
