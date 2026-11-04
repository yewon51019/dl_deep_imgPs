# =====================================================================
#  compare_pft1_cns.py  —  사전학습 PARSeq(base) vs 파인튜닝 PARSeq(pft1) 약 판별을 사진 단위로 비교 (OCR 안 돌림)
# =====================================================================
#
#  [읽기]  저장된 pred.csv (상자는 둘 다 ft1_u2.0). 설정은 match3 에서 검증 부분으로 고른 값: base λ 0.95, pft1 λ 0.9, 둘 다 w 0.02
#  [Top-k] topk_cns 와 같음: 정답 약과 동점이면 가장 나쁜 등수. k = 1 은 저장된 답과 사진마다 같은지 확인
#  [대상]  검증 부분 828장 + 개발용 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\pft1_compare\  summary.csv, per_drug.csv, diff_dev.csv, diff_val.csv (둘의 맞고 틀림이 다른 사진)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\compare_pft1_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import match2_cns as m2
import match_cns as m1

OCR = m1.oc.OCR_OUT
RES = m1.cc.RESULTS
OUT = RES / "pft1_compare"
W, MAX_LEN = 0.02, 30
CFG = {   # 방식 → (λ, 부분 → (읽기 파일, 저장된 답 파일, 저장된 답 열))
    "base": (0.95, {"dev": (OCR / "dev" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv", RES / "match_v3" / "base" / "pred_dev.csv", "best_wA_0.02_pred"),
                    "val": (OCR / "trainall" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv", RES / "match_v3" / "base" / "pred_val.csv", "L0.95_wA_0.02_pred")}),
    "pft1": (0.9, {"dev": (OCR / "dev" / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv", RES / "match_v3" / "pft1" / "pred_dev.csv", "best_wA_0.02_pred"),
                   "val": (OCR / "trainval" / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv", RES / "match_v3" / "pft1" / "pred_val.csv", "L0.9_wA_0.02_pred")}),
}


def score(tag, part, db):
    """사진 이름 → 읽기·답·정답 등수."""
    lam, paths = CFG[tag]
    ocr_path, saved_path, col = paths[part]
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    saved = {r["file_name"]: r[col] for r in csv.DictReader(open(saved_path, encoding="utf-8-sig"))}
    out = {}
    for r in csv.DictReader(open(ocr_path, encoding="utf-8-sig")):
        fn = r["file_name"]
        if fn not in probs:
            continue
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), lam)
        s = {d: T.get(d, 0.0) + W * A[d] for d in A}
        pred, _ = m1.decide(s)
        assert pred == saved[fn], (tag, part, fn, pred, saved[fn])    # 저장된 결과와 사진마다 같은지
        worst = sum(v >= s[r["drug_N"]] - 1e-9 for v in s.values())
        assert (worst == 1) == (pred == r["drug_N"]), fn
        out[fn] = {"drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"], "exact": r["correct"] == "1", "pred": pred,
                   "rank": worst, "aligned": r["aligned"]}
    assert len(out) == len(probs), (tag, part, len(out), len(probs))
    print(f"  {tag} {part} {len(out)}장 재현 확인", flush=True)
    return out


def line(tag, part, name, rs, better, worse):
    n = len(rs)
    pct = lambda v: round(100 * v / max(n, 1), 1)
    s = {"tag": tag, "part": part, "group": name, "n": n, "read_exact": pct(sum(r["exact"] for r in rs))}
    for k in (1, 2, 3, 4, 5):
        s[f"top{k}"] = pct(sum(r["rank"] <= k for r in rs))
    s["none"] = sum(r["pred"] == "" for r in rs)
    s["better_than_base"], s["worse_than_base"] = better, worse
    return s


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
    table, per = [], []
    for part in ("val", "dev"):
        base, pft = score("base", part, db), score("pft1", part, db)
        assert set(base) == set(pft)
        ok = lambda res, f: res[f]["rank"] == 1
        diff = []
        for f in sorted(base):
            if ok(base, f) != ok(pft, f) or base[f]["exact"] != pft[f]["exact"]:
                diff.append({"file_name": f, "drug_N": base[f]["drug_N"], "name": names[base[f]["drug_N"]], "truth": base[f]["truth"],
                             "base_read": base[f]["read"], "pft1_read": pft[f]["read"], "base_pred": base[f]["pred"], "pft1_pred": pft[f]["pred"],
                             "base_rank": base[f]["rank"], "pft1_rank": pft[f]["rank"],
                             "drug_change": "better" if ok(pft, f) and not ok(base, f) else "worse" if ok(base, f) and not ok(pft, f) else "same",
                             "read_change": "better" if pft[f]["exact"] and not base[f]["exact"] else "worse" if base[f]["exact"] and not pft[f]["exact"] else "same"})
        write_csv(OUT / f"diff_{part}.csv", diff)
        groups = [("전체", lambda r: True), ("K-000147 제외", lambda r: r["drug_N"] != "K-000147"), ("긴 알약", lambda r: r["aligned"] == "1"), ("그 외", lambda r: r["aligned"] != "1")]
        for name, cond in groups:
            fs = [f for f in base if cond(base[f])]
            b = sum(ok(pft, f) and not ok(base, f) for f in fs)
            w = sum(ok(base, f) and not ok(pft, f) for f in fs)
            table.append(line("base", part, name, [base[f] for f in fs], "", ""))
            table.append(line("pft1", part, name, [pft[f] for f in fs], b, w))
        by = collections.defaultdict(list)
        for f in base:
            by[base[f]["drug_N"]].append(f)
        for d, fs in sorted(by.items()):
            nb, np_ = sum(ok(base, f) for f in fs), sum(ok(pft, f) for f in fs)
            per.append({"part": part, "drug_N": d, "name": names[d], "n": len(fs), "base_correct": nb, "pft1_correct": np_, "change": np_ - nb})
    write_csv(OUT / "summary.csv", table)
    write_csv(OUT / "per_drug.csv", per)
    for s in table:
        print(s)
    print("--- 약별 변화")
    for p in per:
        if p["change"]:
            print(p["part"], p["drug_N"], p["name"], p["n"], p["base_correct"], "→", p["pft1_correct"], f"({p['change']:+d})")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
