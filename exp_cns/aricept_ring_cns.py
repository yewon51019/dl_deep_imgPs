# =====================================================================
#  aricept_ring_cns.py  —  고리 비교(M2)를 ARICEPT 24장에만 적용해 보기 (사후 확인, 규칙 채택 근거 아님, OCR 안 돌림)
# =====================================================================
#
#  [대상]  최종 평가 K-023336·K-023337 ARICEPT 면 24장, 저장된 pft1 읽기. DB 1,211종 전체
#  [비교]  pft1 / SIM (match8) / RING = SIM + 고리 비교: 5자 이상 각인을 시작 위치를 돌려 가며 비교, 돌려서 맞으면 × 0.95
#          점수 방식은 모두 기준과 같음 (고른 방향 1.0배, 다른 방향 0.9배, + 0.02·A)
#  [칸]    맞음 / 다른 용량 (ARICEPT 인데 5mg↔10mg) / 판정 없음 (ARICEPT 끼리 동점 포함 여부 표시) / 다른 약
#  [주의]  평가 세트 사진이고 다른 약에 주는 손해는 안 잼
#  [저장]  exp_cns\results\test_final_pillset\aricept_ring\  per_photo.csv, summary.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\aricept_ring_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import match7_cns as m7
import match8_cns as m8
import match_cns as m1
import ocr_common_cns as oc
import rerank_tf_cns as rt
from correct_exp300 import sim_v

OUT = m1.cc.RESULTS / "test_final_pillset" / "aricept_ring"
READ = oc.OCR_OUT / "test" / rt.DET / "orig_alignlong_line_pft1" / "pred.csv"
DRUGS = {"K-023336", "K-023337"}
RING_MIN, RING_W = 5, 0.95
W = 0.02
EPS = 1e-9


def load_pairs():
    """match_v8\\confusable.csv 의 5번 이상 글자 쌍."""
    return {frozenset(r["pair"]) for r in csv.DictReader(open(m8.OUT / "confusable.csv", encoding="utf-8-sig")) if int(r["count"]) >= m8.PAIR_MIN}


def sim_ring(a, b, pairs):
    s = m8.sim_soft(a, b, pairs)
    if len(b) >= RING_MIN:
        s = max(s, RING_W * max(m8.sim_soft(a, b[i:] + b[:i], pairs) for i in range(1, len(b))))
    return s


def verdict(pred, ranked, truth):
    if pred == truth:
        return "맞음"
    if pred in DRUGS:
        return "다른 용량"
    if not pred:
        top = {d for d, v in ranked if v >= ranked[0][1] - EPS}
        return "판정 없음 (ARICEPT 동점)" if top & DRUGS else "판정 없음 (그 밖)"
    return "다른 약"


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    names = {d["drug_N"]: d["name"] for d in db}
    pairs = load_pairs()
    sims = {"pft1": m8.Sims(db, lambda a, b: sim_v(a, b, *m1.OPT)),
            "SIM": m8.Sims(db, lambda a, b: m8.sim_soft(a, b, pairs)),
            "RING": m8.Sims(db, lambda a, b: sim_ring(a, b, pairs))}
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / "probs_test.csv", encoding="utf-8-sig"))}
    rows = [r for r in csv.DictReader(open(READ, encoding="utf-8-sig")) if r["drug_N"] in DRUGS and r["truth"] == "ARICEPT"]
    assert len(rows) == 24, len(rows)
    per, count = [], {k: collections.Counter() for k in sims}
    for r in rows:
        A = {d["drug_N"]: m1.attr_score(probs[r["file_name"]], d) for d in db}
        main_c, other, _ = m8.candidates(r)
        row = {"file_name": r["file_name"], "drug_N": r["drug_N"], "camera": r["camera_la"], "read": r["read"]}
        for name, s in sims.items():
            T = m8.t_lambda(main_c, other, s)
            sc = {d: T.get(d, 0.0) + W * A[d] for d in A}
            pred, _ = m1.decide(sc)
            ranked = m1.rank(sc)
            v = verdict(pred, ranked, r["drug_N"])
            count[name][v] += 1
            me = sc[r["drug_N"]]
            row.update({f"{name}_answer": names.get(pred, "(판정 없음)")[:20], f"{name}_verdict": v,
                        f"{name}_truth_rank": sum(x > me + EPS for x in sc.values()) + 1, f"{name}_top_score": round(ranked[0][1], 3), f"{name}_truth_score": round(me, 3)})
        per.append(row)
    m7.write_csv(OUT / "per_photo.csv", per)
    kinds = ["맞음", "다른 용량", "판정 없음 (ARICEPT 동점)", "판정 없음 (그 밖)", "다른 약"]
    table = []
    for name in sims:
        ranks = [p[f"{name}_truth_rank"] for p in per]
        table.append({"method": name, **{k: count[name][k] for k in kinds}, "truth_top1": sum(x == 1 for x in ranks), "truth_top3": sum(x <= 3 for x in ranks), "median_rank": sorted(ranks)[len(ranks) // 2]})
        print(table[-1])
    m7.write_csv(OUT / "summary.csv", table)
    for p in per:
        print(p["file_name"][:26], "| 읽음", p["read"], "|", p["pft1_verdict"], "→", p["RING_verdict"], "|", p["RING_answer"], p["RING_top_score"], "정답 점수", p["RING_truth_score"], "등수", p["RING_truth_rank"])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
