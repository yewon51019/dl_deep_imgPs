# =====================================================================
#  display_cns.py  —  결과 표시 규칙 시험: 사진마다 약 후보 몇 개를 보여 주고, 그 안에 정답이 있는가 (OCR 안 돌림)
# =====================================================================
#
#  [규칙, 실행 전 고정]
#     확실 (답 있음, 1등 점수 ≥ 0.95, 1등 − 2등 ≥ 0.1)  → 약 1개
#     1등 동점 (판정 없음)                              → 1등과 동점인 약 전부
#     그 외 불확실                                      → 1등 + 2등 + 2등과 동점인 약 (DB 순서로 자르지 않음)
#  [재는 것]  정답 포함률, 평균·최대 표시 개수, 약 1개만 보여 준 비율과 그중 틀린 사진 (사람이 확인하지 않을 수 있어 가장 위험)
#  [대상]  pft1 (이후 작업의 기본 읽기) + 비교용 base. 검증 828장, 개발 468장. 평가용 사진·결과는 읽지 않음
#          격자표(grid.csv)는 기준값에 따른 변화를 보여 주는 참고표. 여기서 기준값을 고르지 않음
#  [저장]  exp_cns\results\display_v1\  summary.csv, grid.csv, sure_wrong.csv (약 1개를 보여 줬는데 틀린 사진)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\display_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import compare_pft1_cns as cp
import match2_cns as m2
import match_cns as m1

OUT = m1.cc.RESULTS / "display_v1"
SURE_TOP, SURE_GAP = 0.95, 0.1
GRID_TOP, GRID_GAP = [0.85, 0.9, 0.95, 0.99], [0.05, 0.1, 0.2, 0.4]
EPS = 1e-9


def score(tag, part, db):
    """사진마다 (OCR 줄, 점수순 [(약, 점수)]). 저장된 답과 같은지 확인."""
    lam, paths = cp.CFG[tag]
    ocr_path, saved_path, col = paths[part]
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    saved = {r["file_name"]: r[col] for r in csv.DictReader(open(saved_path, encoding="utf-8-sig"))}
    out = []
    for r in csv.DictReader(open(ocr_path, encoding="utf-8-sig")):
        fn = r["file_name"]
        if fn not in probs:
            continue
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        cm, co = m2.split_candidates(r, cp.MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), lam)
        s = {d: T.get(d, 0.0) + cp.W * A[d] for d in A}
        pred, _ = m1.decide(s)
        assert pred == saved[fn], (tag, part, fn)
        out.append((r, pred, m1.rank(s)))
    assert len(out) == len(probs)
    print(f"  {tag} {part} {len(out)}장 재현 확인", flush=True)
    return out


def shown(pred, ranked, top_min, gap_min):
    """(표시 종류, 보여 줄 약 목록)."""
    s1, s2 = ranked[0][1], ranked[1][1]
    if not pred:
        return "tie", [d for d, v in ranked if v >= s1 - EPS]
    if s1 >= top_min and s1 - s2 >= gap_min:
        return "sure", [pred]
    return "unsure", [d for d, v in ranked if v >= s2 - EPS]


def measure(items, top_min, gap_min):
    res = [(r, pred, *shown(pred, ranked, top_min, gap_min)) for r, pred, ranked in items]
    n = len(res)
    pct = lambda v, m=n: round(100 * v / max(m, 1), 1)
    kinds = {k: [x for x in res if x[2] == k] for k in ("sure", "unsure", "tie")}
    has = lambda x: x[0]["drug_N"] in x[3]
    sizes = [len(x[3]) for x in res]
    return {"n": n, "include_truth": pct(sum(has(x) for x in res)), "top1": pct(sum(x[1] == x[0]["drug_N"] for x in res)),
            "mean_shown": round(sum(sizes) / n, 2), "max_shown": max(sizes),
            "sure": len(kinds["sure"]), "sure_share": pct(len(kinds["sure"])), "sure_wrong": sum(not has(x) for x in kinds["sure"]),
            "sure_acc": pct(sum(has(x) for x in kinds["sure"]), len(kinds["sure"])),
            "unsure": len(kinds["unsure"]), "unsure_include": pct(sum(has(x) for x in kinds["unsure"]), len(kinds["unsure"])),
            "unsure_mean_shown": round(sum(len(x[3]) for x in kinds["unsure"]) / max(len(kinds["unsure"]), 1), 2),
            "tie": len(kinds["tie"]), "tie_include": pct(sum(has(x) for x in kinds["tie"]), len(kinds["tie"])),
            "tie_mean_shown": round(sum(len(x[3]) for x in kinds["tie"]) / max(len(kinds["tie"]), 1), 2)}, res


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
    names = {d["drug_N"]: d["name"] for d in db}
    table, grid, wrong = [], [], []
    for tag in ("pft1", "base"):
        for part in ("val", "dev"):
            items = score(tag, part, db)
            groups = [("전체", items)]
            if part == "dev":
                groups.append(("K-000147 제외", [x for x in items if x[0]["drug_N"] != "K-000147"]))
            for name, its in groups:
                m, res = measure(its, SURE_TOP, SURE_GAP)
                table.append({"tag": tag, "part": part, "group": name, **m})
                print(table[-1])
                if name == "전체":
                    for x in res:
                        if x[2] == "sure" and x[0]["drug_N"] not in x[3]:
                            wrong.append({"tag": tag, "part": part, "file_name": x[0]["file_name"], "drug_N": x[0]["drug_N"], "name": names[x[0]["drug_N"]],
                                          "truth": x[0]["truth"], "read": x[0]["read"], "shown": x[1], "shown_name": names[x[1]]})
            for t in GRID_TOP:
                for g in GRID_GAP:
                    m, _ = measure(items, t, g)
                    grid.append({"tag": tag, "part": part, "top_min": t, "gap_min": g, **m})
    write_csv(OUT / "summary.csv", table)
    write_csv(OUT / "grid.csv", grid)
    write_csv(OUT / "sure_wrong.csv", wrong)
    print("--- pft1 격자 (기준값 → 포함률 / 평균 표시 / 1개 표시 비율 / 1개 표시인데 틀림)")
    for g in grid:
        if g["tag"] == "pft1":
            print(g["part"], g["top_min"], g["gap_min"], g["include_truth"], g["mean_shown"], g["sure_share"], g["sure_wrong"])
    print("--- pft1 약 1개를 보여 줬는데 틀린 사진")
    for w in wrong:
        if w["tag"] == "pft1":
            print(w["part"], w["file_name"], w["truth"], "| 읽기", w["read"], "| 답", w["shown"], w["shown_name"])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
