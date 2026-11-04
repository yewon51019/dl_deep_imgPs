# =====================================================================
#  match8_cns.py  —  약 판별 규칙 시험: 방향별 확신도 · 긴 일치 우선 · 비슷한 글자 부분 점수 (pft1 저장 읽기로 계산, OCR 안 돌림)
# =====================================================================
#
#  [기준]  pft1, T = max(T_고른 방향, 0.9·T_다른 방향), 점수 = T + 0.02·A  (저장된 답과 사진마다 같은지 확인)
#  [C1]    고른/다른 방향 구분 없이 T(약) = 모든 읽기 후보의 max( 유사도 × (확신도/100)^p ). 고른 방향 상자 순서 바꾸기 후보는 그 방향 확신도
#          p 는 0.5 / 1 / 2 중 검증 맞은 수로 고름 (같으면 작은 값), L1·L1+SIM 도 같은 p
#  [L1]    C1 + 3자 이상 읽기가 3자 이상 DB 각인과 유사도 0.7 이상이면 그 사진에서 2자 이하 읽기 후보를 뺌
#  [SIM]   기준 점수 방식에서 유사도만 바꿈: 글자 통일(fold) 대신 편집거리에서 비슷한 글자 치환 비용 0.25
#          비슷한 글자 = 기존 fold 묶음 + 학습용 읽기 오류에서 5번 이상 나온 글자 쌍 (confusable.csv)
#          부분일치: 읽기가 각인 안에 원래 글자로 들어 있으면 V3 공식, fold 후에만 들어 있으면 × 0.9
#  [L1+SIM] L1 에 SIM 유사도
#  [채택 규칙, 실행 전 고정]  모두 만족
#     ① (좋아짐 − 나빠짐) 검증 ≥ +5, 개발 ≥ 0   ② 틀린 답 증가 ≤ 맞은 수 증가 (검증·개발 각각)
#     여럿이면 검증 맞은 수 최고, 같으면 규칙이 적은 쪽. 개발용으로 p 를 고르지 않음
#  [대상]  검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음 (글자 쌍은 학습용 중 검증 약을 뺀 사진으로만)
#  [저장]  exp_cns\results\match_v8\  summary.csv, grid_val.csv, confusable.csv, changed.csv, pred_val.csv, pred_dev.csv  (이미 있으면 멈춤)
#  [실행]  python exp_cns\match8_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import compare_pft1_cns as cp
import display_cns as dc
import match2_cns as m2
import match7_cns as m7
import match_cns as m1
from correct_exp300 import MIN_SIM, fold, sim_v

RES = m1.cc.RESULTS
OUT = RES / "match_v8"
LAM, W = 0.9, 0.02
P_GRID = [0.5, 1.0, 2.0]
LONG_MIN, LONG_SIM = 3, 0.7
SUB_COST, PAIR_MIN = 0.25, 5
BETTER_MIN, DEV_MIN = 5, 0
FAMILIES = ("C1", "L1", "SIM", "L1SIM")


def learn_pairs():
    """학습용(검증 약 제외) 사전학습 PARSeq 읽기에서 길이가 같고 한 글자만 다른 쌍을 셈. fold 로 이미 같은 쌍은 표시만."""
    val = {r["file_name"] for r in csv.DictReader(open(RES / "attr_cnn_v1" / "labels.csv", encoding="utf-8-sig")) if r["part"] == "val"}
    cnt = collections.Counter()
    path = m1.oc.OCR_OUT / "trainall" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["file_name"] in val or len(r["read"]) != len(r["truth"]):
            continue
        diff = [(x, y) for x, y in zip(r["read"], r["truth"]) if x != y]
        if len(diff) == 1:
            cnt[frozenset(diff[0])] += 1
    rows = [{"pair": "".join(sorted(k)), "count": v, "already_fold": int(fold(min(k)) == fold(max(k)))} for k, v in cnt.most_common()]
    m7.write_csv(OUT / "confusable.csv", rows)
    pairs = {k for k, v in cnt.items() if v >= PAIR_MIN}
    print("비슷한 글자 쌍 (5번 이상):", [(r["pair"], r["count"]) for r in rows if r["count"] >= PAIR_MIN], flush=True)
    return pairs


def soft_lev(a, b, pairs):
    """편집거리, 비슷한 글자 치환 비용 SUB_COST."""
    prev = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                sub = 0.0
            elif fold(ca) == fold(cb) or frozenset((ca, cb)) in pairs:
                sub = SUB_COST
            else:
                sub = 1.0
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def sim_soft(a, b, pairs):
    L = max(len(a), len(b))
    s = 0.0
    if abs(len(a) - len(b)) <= 0.54 * L:                             # 길이 차만으로 0.3 미만이면 편집거리를 안 셈
        ed = soft_lev(a, b, pairs)
        s = (L - ed) / (L + ed)
    if 2 <= len(a) < len(b):
        if a in b:
            s = max(s, 0.5 + 0.5 * len(a) / len(b))
        elif fold(a) in fold(b):
            s = max(s, 0.9 * (0.5 + 0.5 * len(a) / len(b)))
    return s


class Sims:
    """읽기 → (약별 유사도, 긴 일치 여부). 같은 읽기는 한 번만 계산."""

    def __init__(self, db, f):
        self.db, self.f, self.memo = db, f, {}

    def get(self, read):
        if read not in self.memo:
            scores, long_hit = {}, False
            for d in self.db:
                best = 0.0
                for imp in d["imprints"]:
                    s = self.f(read, imp)
                    best = max(best, s)
                    if len(read) >= LONG_MIN and len(imp) >= LONG_MIN and s >= LONG_SIM:
                        long_hit = True
                if best >= MIN_SIM:
                    scores[d["drug_N"]] = best
            self.memo[read] = (scores, long_hit)
        return self.memo[read]


def candidates(r):
    """(고른 방향 후보, 다른 방향 후보, {후보: 확신도}). 같은 글자가 여러 방향이면 높은 확신도."""
    main, other = m2.split_candidates(r, cp.MAX_LEN)
    conf = {}
    for s in r["all_dirs"].split(" | "):
        _, t, c = s.split(":")
        conf[t] = max(conf.get(t, 0.0), float(c))
    for c in main:
        conf[c] = max(conf.get(c, 0.0), float(r["conf"]))
    return main, other, conf


def t_lambda(main, other, sims):
    T = {}
    for c in main:
        for d, s in sims.get(c)[0].items():
            T[d] = max(T.get(d, 0.0), s)
    for c in other:
        for d, s in sims.get(c)[0].items():
            if LAM * s >= MIN_SIM:
                T[d] = max(T.get(d, 0.0), LAM * s)
    return T


def t_conf(cands, conf, sims, p, long_first):
    use = cands
    if long_first and any(sims.get(c)[1] for c in cands):
        use = [c for c in cands if len(c) > 2]
    T = {}
    for c in use:
        w = (conf[c] / 100) ** p
        for d, s in sims.get(c)[0].items():
            if s * w >= MIN_SIM:
                T[d] = max(T.get(d, 0.0), s * w)
    return T


def predict(part, db, v3, soft, names):
    """names = 계산할 방식 이름. 사진마다 (OCR 줄, {이름: (답, 1~3등)})."""
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out = []
    for r, pred, ranked in dc.score("pft1", part, db):
        A = {d["drug_N"]: m1.attr_score(probs[r["file_name"]], d) for d in db}
        main, other, conf = candidates(r)
        cands = list(dict.fromkeys(main + other))
        res = {}
        for name in names:
            if name == "base":
                T = t_lambda(main, other, v3)
            elif name == "SIM":
                T = t_lambda(main, other, soft)
            else:
                fam, p = name.split("_")
                T = t_conf(cands, conf, soft if fam == "L1SIM" else v3, float(p), fam != "C1")
            s = {d: T.get(d, 0.0) + W * A[d] for d in A}
            res[name] = m1.decide(s)
        assert res["base"][0] == pred, (part, r["file_name"], res["base"][0], pred)
        out.append((r, res))
        if len(out) % 200 == 0:
            print(f"  {part} {len(out)}장", flush=True)
    print(f"  {part} {len(out)}장 기준 재현 확인", flush=True)
    return out


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    pairs = learn_pairs()
    v3 = Sims(db, lambda a, b: sim_v(a, b, *m1.OPT))
    soft = Sims(db, lambda a, b: sim_soft(a, b, pairs))
    grid_names = ["base", "SIM"] + [f"{f}_{p:g}" for f in ("C1", "L1", "L1SIM") for p in P_GRID]
    val = predict("val", db, v3, soft, grid_names)
    grid = [{"config": n, **m7.measure(val, n)} for n in grid_names]
    m7.write_csv(OUT / "grid_val.csv", grid)
    for g in grid:
        print(g)
    hits = {g["config"]: g["hit"] for g in grid}
    p = min(P_GRID, key=lambda x: (-hits[f"C1_{x:g}"], x))
    print(f"검증으로 고른 p = {p:g}", flush=True)
    finals = {"C1": f"C1_{p:g}", "L1": f"L1_{p:g}", "SIM": "SIM", "L1SIM": f"L1SIM_{p:g}"}
    dev = predict("dev", db, v3, soft, ["base"] + list(finals.values()))
    table, changed = [], []
    for part, out in (("val", val), ("dev", dev)):
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for fam, name in [("base", "base")] + list(finals.items()):
                table.append({"part": part, "group": gname, "config": fam, "detail": name, **m7.measure(its, name)})
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
        print(t)
    get = lambda part, cfg: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == cfg)
    passed = []
    for fam in FAMILIES:
        v, d, v0, d0 = get("val", fam), get("dev", fam), get("val", "base"), get("dev", "base")
        vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
        ok1 = vd >= BETTER_MIN and dd >= DEV_MIN
        ok2 = v["wrong"] - v0["wrong"] <= v["hit"] - v0["hit"] and d["wrong"] - d0["wrong"] <= d["hit"] - d0["hit"]
        print(f"{fam}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}, 틀림 {v0['wrong']}→{v['wrong']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}, 틀림 {d0['wrong']}→{d['wrong']}) → ① {ok1} ② {ok2}")
        if ok1 and ok2:
            passed.append((v["hit"], -FAMILIES.index(fam), fam))
    print("채택:", max(passed)[2] if passed else "없음 (기준 유지)")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
