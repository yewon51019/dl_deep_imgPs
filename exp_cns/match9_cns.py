# =====================================================================
#  match9_cns.py  —  지지 방향 수(SUP) 시험: 한 번 맞은 읽기보다 여러 방향이 지지하는 약을 우선 (저장 읽기로 계산, OCR 안 돌림)
# =====================================================================
#
#  [기준]  pft1, T = max(T_고른 방향, 0.9·T_다른 방향), 점수 = T + 0.02·A  (저장된 답과 사진마다 같은지 확인)
#  [SUP]   12방향 사진만: 점수 = T + β × (그 약을 지지하는 방향 수 ÷ 글자가 나온 방향 수) + 0.02·A   (긴 알약은 그대로)
#          지지 = 그 방향 읽기와 약의 각인(앞·뒷면 중 높은 쪽)의 고리 비교 유사도 ≥ 0.6
#          고리 비교 = 5자 이상 각인은 시작 위치를 돌려 비교, 돌려서 맞으면 × 0.95. 한 글자 읽기는 부분일치가 안 되므로 같은 각인만 지지
#          β 는 0.1 / 0.3 / 0.5 중 pft1+SUP 의 검증 맞은 수로 고름 (같으면 작은 값), SIM+SUP 도 같은 β. 개발용으로 고르지 않음
#  [비교]  기준 pft1 / pft1+SUP / SIM+SUP (SIM = match8 의 비슷한 글자 부분 점수)
#  [채택 규칙, 실행 전 고정]  모두 만족
#     ① (좋아짐 − 나빠짐) 검증 ≥ +5, 개발 ≥ 0   ② 틀린 답 증가 ≤ 맞은 수 증가 (검증·개발 각각)
#     여럿이면 검증 맞은 수 최고, 같으면 규칙이 적은 쪽
#  [대상]  검증 828장 + 개발 468장으로만 고름
#  [사후 참고, 채택과 무관]  최종 평가 ARICEPT 24장의 정답 1등(동점 포함)·3등 안 장수 (같은 β)
#  [저장]  exp_cns\results\match_v9\  summary.csv, grid_val.csv, changed.csv, pred_val.csv, pred_dev.csv, aricept.csv  (이미 있으면 멈춤)
#  [실행]  python exp_cns\match9_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import aricept_ring_cns as ar
import display_cns as dc
import match7_cns as m7
import match8_cns as m8
import match_cns as m1
from correct_exp300 import sim_v

OUT = m1.cc.RESULTS / "match_v9"
W = 0.02
SUP_SIM = 0.6
BETAS = [0.1, 0.3, 0.5]
RING_MIN, RING_W = ar.RING_MIN, ar.RING_W
BETTER_MIN, DEV_MIN = 5, 0
EPS = 1e-9


def ring_v3(a, b):
    """V3 유사도의 고리 비교판."""
    s = sim_v(a, b, *m1.OPT)
    if len(b) >= RING_MIN:
        s = max(s, RING_W * max(sim_v(a, b[i:] + b[:i], *m1.OPT) for i in range(1, len(b))))
    return s


def support(r, ring):
    """(약별 지지 방향 수, 글자가 나온 방향 수)."""
    reads = [s.split(":")[1] for s in r["all_dirs"].split(" | ")]
    reads = [t for t in reads if t]
    cnt = {}
    for t in reads:
        for d, v in ring.get(t)[0].items():
            if v >= SUP_SIM:
                cnt[d] = cnt.get(d, 0) + 1
    return cnt, len(reads)


class Env:
    def __init__(self, db, pairs):
        self.db = db
        self.v3 = m8.Sims(db, lambda a, b: sim_v(a, b, *m1.OPT))
        self.soft = m8.Sims(db, lambda a, b: m8.sim_soft(a, b, pairs))
        self.ring_v3 = m8.Sims(db, ring_v3)
        self.ring_soft = m8.Sims(db, lambda a, b: ar.sim_ring(a, b, pairs))

    def scores(self, r, prob, cfg, beta):
        """cfg = base / pft1SUP / SIMSUP → 약별 최종 점수."""
        main_c, other, _ = m8.candidates(r)
        T = m8.t_lambda(main_c, other, self.soft if cfg == "SIMSUP" else self.v3)
        A = {d["drug_N"]: m1.attr_score(prob, d) for d in self.db}
        s = {d: T.get(d, 0.0) + W * A[d] for d in A}
        if cfg != "base" and r["aligned"] != "1":
            cnt, n = support(r, self.ring_soft if cfg == "SIMSUP" else self.ring_v3)
            for d, c in cnt.items():
                s[d] += beta * c / max(n, 1)
        return s


def predict(part, env, cfgs):
    """cfgs = [(이름, 방식, β)]."""
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out = []
    for r, pred, ranked in dc.score("pft1", part, env.db):
        res = {"base": (pred, ranked[:3])}
        for name, cfg, beta in cfgs:
            res[name] = m1.decide(env.scores(r, probs[r["file_name"]], cfg, beta))
        out.append((r, res))
        if len(out) % 200 == 0:
            print(f"  {part} {len(out)}장", flush=True)
    return out


def aricept(env, beta):
    """사후 참고: 최종 평가 ARICEPT 24장 (채택과 무관)."""
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / "probs_test.csv", encoding="utf-8-sig"))}
    rows = [r for r in csv.DictReader(open(ar.READ, encoding="utf-8-sig")) if r["drug_N"] in ar.DRUGS and r["truth"] == "ARICEPT"]
    assert len(rows) == 24
    table = []
    for cfg in ("base", "pft1SUP", "SIMSUP"):
        top1, top3, none, other = 0, 0, 0, 0
        for r in rows:
            s = env.scores(r, probs[r["file_name"]], cfg, beta)
            pred = m1.decide(s)[0]
            me = s[r["drug_N"]]
            rank = sum(v > me + EPS for v in s.values()) + 1
            top1 += rank == 1
            top3 += rank <= 3
            none += pred == ""
            other += bool(pred) and pred not in ar.DRUGS
        table.append({"config": cfg, "photos": 24, "truth_top1_tied": top1, "truth_top3": top3, "none": none, "other_drug": other})
        print("ARICEPT", table[-1], flush=True)
    m7.write_csv(OUT / "aricept.csv", table)


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    env = Env(db, ar.load_pairs())
    grid_cfg = [(f"pft1SUP_{b:g}", "pft1SUP", b) for b in BETAS] + [(f"SIMSUP_{b:g}", "SIMSUP", b) for b in BETAS]
    val = predict("val", env, grid_cfg)
    grid = [{"config": c[0], **m7.measure(val, c[0])} for c in grid_cfg]
    m7.write_csv(OUT / "grid_val.csv", grid)
    for g in grid:
        print(g, flush=True)
    hits = {g["config"]: g["hit"] for g in grid}
    beta = min(BETAS, key=lambda b: (-hits[f"pft1SUP_{b:g}"], b))
    print(f"검증으로 고른 β = {beta:g}", flush=True)
    finals = {"pft1SUP": f"pft1SUP_{beta:g}", "SIMSUP": f"SIMSUP_{beta:g}"}
    dev = predict("dev", env, [(n, f, beta) for f, n in finals.items()])
    table, changed = [], []
    for part, out in (("val", val), ("dev", dev)):
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for fam, name in [("base", "base")] + list(finals.items()):
                table.append({"part": part, "group": gname, "config": fam, **m7.measure(its, name)})
        for r, res in out:
            for fam, name in finals.items():
                k = m7.kind(r["drug_N"], res["base"][0], res[name][0])
                if k:
                    changed.append({"config": fam, "part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                    "change": k, "pred_base": res["base"][0], "pred_new": res[name][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "base"] + list(finals))
            for r, res in out:
                w.writerow([r["file_name"], r["drug_N"], r["truth"], r["read"], res["base"][0]] + [res[n][0] for n in finals.values()])
    m7.write_csv(OUT / "summary.csv", table)
    if changed:
        m7.write_csv(OUT / "changed.csv", changed)
    for t in table:
        print(t, flush=True)
    get = lambda part, cfg: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == cfg)
    passed = []
    for fam in finals:
        v, d, v0, d0 = get("val", fam), get("dev", fam), get("val", "base"), get("dev", "base")
        vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
        ok1 = vd >= BETTER_MIN and dd >= DEV_MIN
        ok2 = v["wrong"] - v0["wrong"] <= v["hit"] - v0["hit"] and d["wrong"] - d0["wrong"] <= d["hit"] - d0["hit"]
        print(f"{fam}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}, 틀림 {v0['wrong']}→{v['wrong']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}, 틀림 {d0['wrong']}→{d['wrong']}) → ① {ok1} ② {ok2}")
        if ok1 and ok2:
            passed.append((v["hit"], fam == "pft1SUP", fam))
    print("채택:", max(passed)[2] if passed else "없음 (기준 유지)")
    aricept(env, beta)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
