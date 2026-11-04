# =====================================================================
#  match7_cns.py  —  약 판별 검색 규칙 R1·R2 시험 (pft1 저장 읽기로 계산, OCR 안 돌림)
# =====================================================================
#
#  [기준]  pft1, T = max(T_고른 방향, 0.9·T_다른 방향), 길이 ≤ 30, 점수 = T + 0.02·A  (compare_pft1_cns 와 같음, 저장된 답과 사진마다 같은지 확인)
#  [R1]    다른 방향에서 읽은 2자 이하 후보에만 더 낮은 λ (0.5 / 0.7 / 0.8 중 검증 맞은 수로 고름, 같으면 큰 쪽)
#  [R2]    위아래 대칭 글자(H I N O S X Z 0 8)로만 된 2자 이상 읽기는 뒤집은 순서도 다른 방향 후보에 넣음
#          (180° 돌리면 HS ↔ SH 가 되는 각인, K-017707)
#  [R1+R2] 고른 방향 읽기를 뒤집은 후보는 λ 0.9, 다른 방향 읽기를 뒤집은 2자 이하 후보만 R1 의 λ
#  [채택 규칙, 실행 전 고정]  기준 대비 (좋아짐 − 나빠짐) 이 검증 +5장 이상, 개발 0장 이상
#          여럿이 통과하면 검증 맞은 수 최고, 같으면 규칙이 적은 쪽. 개발용으로 λ 를 고르지 않음
#  [대상]  검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\match_v7\  summary.csv, grid_val.csv, changed.csv, pred_val.csv, pred_dev.csv
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\match7_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import compare_pft1_cns as cp
import match2_cns as m2
import match_cns as m1

OUT = m1.cc.RESULTS / "match_v7"
LAM, W, SHORT = 0.9, 0.02, 2
SHORT_GRID = [0.8, 0.7, 0.5]
SYM = set("HINOSXZ08")
BETTER_MIN, DEV_MIN = 5, 0


def flipped(cands):
    """위아래 대칭 글자로만 된 2자 이상 후보의 뒤집은 순서."""
    return {c[::-1] for c in cands if len(c) >= 2 and set(c) <= SYM}


def scored_sets(r, db):
    """사진 하나의 후보 묶음별 각인 점수 (고른 방향 / 다른 방향 긴 것·짧은 것 / 뒤집은 후보). 규칙마다 다시 계산하지 않으려고 미리 구함."""
    main, other = map(set, m2.split_candidates(r, cp.MAX_LEN))
    fm = flipped(main) - main - other
    fo = flipped(other) - main - other - fm
    is_short = lambda c: len(c) <= SHORT
    sets = {"main": main, "o_long": {c for c in other if not is_short(c)}, "o_short": {c for c in other if is_short(c)},
            "f_main": fm, "f_long": {c for c in fo if not is_short(c)}, "f_short": {c for c in fo if is_short(c)}}
    return {k: m2.text_scores(sorted(v), db, False) for k, v in sets.items()}


def total_T(s, lam_short, use_flip):
    T = dict(s["main"])
    for key, lam in [("o_long", LAM), ("o_short", lam_short)] + ([("f_main", LAM), ("f_long", LAM), ("f_short", lam_short)] if use_flip else []):
        T = m2.combine(T, s[key], lam)
    return T


def predict(part, db, configs):
    """사진마다 (OCR 줄, {방식: (답, 1~3등)}). 기준은 저장된 답과 같은지 확인."""
    ocr_path, saved_path, col = cp.CFG["pft1"][1][part]
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    saved = {r["file_name"]: r[col] for r in csv.DictReader(open(saved_path, encoding="utf-8-sig"))}
    out = []
    for r in csv.DictReader(open(ocr_path, encoding="utf-8-sig")):
        fn = r["file_name"]
        if fn not in probs:
            continue
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        s = scored_sets(r, db)
        res = {}
        for name, lam_short, use_flip in configs:
            T = total_T(s, lam_short, use_flip)
            res[name] = m1.decide({d: T.get(d, 0.0) + W * A[d] for d in A})
        assert res["base"][0] == saved[fn], (part, fn, res["base"][0], saved[fn])
        out.append((r, res))
        if len(out) % 200 == 0:
            print(f"  {part} {len(out)}/{len(probs)}", flush=True)
    assert len(out) == len(probs), (part, len(out), len(probs))
    print(f"  {part} {len(out)}장 기준 재현 확인", flush=True)
    return out


def kind(truth, old, new):
    if old == new:
        return ""
    if new == truth:
        return "좋아짐"
    if old == truth:
        return "나빠짐"
    if new == "":
        return "틀림→판정없음"
    return "판정없음→틀림" if old == "" else "틀림→다른 틀림"


def measure(out, name):
    pick = lambda r, res: res[name][0]
    kinds = [kind(r["drug_N"], res["base"][0], pick(r, res)) for r, res in out]
    n = len(out)
    hit = sum(pick(r, res) == r["drug_N"] for r, res in out)
    group = lambda f: [pick(r, res) == r["drug_N"] for r, res in out if f(r)]
    pc = lambda v: round(100 * sum(v) / max(len(v), 1), 1)
    return {"n": n, "hit": hit, "acc": round(100 * hit / n, 1), "none": sum(pick(r, res) == "" for r, res in out),
            "wrong": sum(pick(r, res) not in ("", r["drug_N"]) for r, res in out),
            "better": kinds.count("좋아짐"), "worse": kinds.count("나빠짐"),
            "wrong_to_none": kinds.count("틀림→판정없음"), "none_to_wrong": kinds.count("판정없음→틀림"), "wrong_to_wrong": kinds.count("틀림→다른 틀림"),
            "long": pc(group(lambda r: r["aligned"] == "1")), "other": pc(group(lambda r: r["aligned"] != "1"))}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    grid = [("base", LAM, False), ("R2", LAM, True)]
    grid += [(f"R1_{x:g}", x, False) for x in SHORT_GRID] + [(f"R1R2_{x:g}", x, True) for x in SHORT_GRID]
    val = predict("val", db, grid)
    grid_rows = [{"config": c[0], "lam_short": c[1], "flip": int(c[2]), **measure(val, c[0])} for c in grid]
    write_csv(OUT / "grid_val.csv", grid_rows)
    hits = {g["config"]: g["hit"] for g in grid_rows}
    best_short = max(SHORT_GRID, key=lambda x: (hits[f"R1_{x:g}"], x))   # R1 의 λ 는 검증으로만 고름, R1+R2 도 같은 값
    print(f"검증으로 고른 R1 λ = {best_short:g}", flush=True)
    finals = [("base", LAM, False), ("R1", best_short, False), ("R2", LAM, True), ("R1R2", best_short, True)]
    alias = {"base": "base", "R1": f"R1_{best_short:g}", "R2": "R2", "R1R2": f"R1R2_{best_short:g}"}
    val = [(r, {**res, **{k: res[v] for k, v in alias.items()}}) for r, res in val]
    dev = predict("dev", db, finals)

    table, changed = [], []
    for part, out in (("val", val), ("dev", dev)):
        groups = [("전체", out)]
        if part == "dev":
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for name, lam_short, use_flip in finals:
                table.append({"part": part, "group": gname, "config": name, "lam_short": lam_short, "flip": int(use_flip), **measure(its, name)})
        for r, res in out:
            for name, _, _ in finals[1:]:
                k = kind(r["drug_N"], res["base"][0], res[name][0])
                if k:
                    changed.append({"config": name, "part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                    "change": k, "pred_base": res["base"][0], "pred_new": res[name][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read"] + [f"{c[0]}_{x}" for c in finals for x in ("pred", "top3")])
            for r, res in out:
                row = [r["file_name"], r["drug_N"], r["truth"], r["read"]]
                for c in finals:
                    p, t3 = res[c[0]]
                    row += [p, " ".join(f"{d}:{v:.3f}" for d, v in t3)]
                w.writerow(row)
    write_csv(OUT / "summary.csv", table)
    if changed:
        write_csv(OUT / "changed.csv", changed)

    for t in table:
        print(t)
    get = lambda part, name: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == name)
    passed = []
    for name, _, _ in finals[1:]:
        v, d = get("val", name), get("dev", name)
        vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
        ok = vd >= BETTER_MIN and dd >= DEV_MIN
        print(f"{name}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}) → {'통과' if ok else '미달'}")
        if ok:
            passed.append((v["hit"], -name.count("R"), name))
    print("채택:", max(passed)[2] if passed else "없음 (기준 유지)")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
