# =====================================================================
#  topk_cns.py  —  약 후보 Top-k 정확도 (개발용 468장, 저장된 읽기로 다시 계산, OCR 안 돌림)
# =====================================================================
#
#  [Top-k]  상위 k 개 안에 정답 약이 있으면 맞은 것으로 침 (Al-Hussaeni et al., 2023 의 지표). k = 1~5
#  [점수]   주 방식 base 그대로 (match2 방식, λ 0.95, 길이 ≤ 30, w 0.02). 사진마다 저장된 base 답과 같은지 확인
#  [동점]   정답 약이 다른 약과 동점이면 가장 나쁜 등수로 침 (DB 순서가 앞이라 운 좋게 맞는 일을 막음)
#           그래서 k = 1 은 base 정확도와 같아야 함 (동점 = 판정 없음 = 오답). 다르면 멈춤
#           old_top3 = 예전 방식 (점수순 + DB 순서로 3개 자름) 을 나란히 기록
#  [k 고르기, 실행 전 고정]  K-000147 제외 묶음에서 k 를 1 늘려도 1%p 미만으로 오르면 그 앞의 k 에서 멈춤. 약이 36종뿐이라 확정이 아니라 후보
#  [저장]  exp_cns\results\topk_v1\  summary.csv, per_drug.csv, pred_dev.csv. 이미 있으면 덮어쓰지 않고 멈춤
#          평가용 사진·결과는 읽지 않음
#  [실행]  python exp_cns\topk_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import match2_cns as m2
import match_cns as m1

OUT = m1.cc.RESULTS / "topk_v1"
SAVED = m1.cc.RESULTS / "match_v3" / "base" / "pred_dev.csv"
SAVED_COL = "best_wA_0.02_pred"
LAM, MAX_LEN, W = 0.95, 30, 0.02                 # 사전 기록한 주 방식 값
KS = [1, 2, 3, 4, 5]
MIN_GAIN = 1.0                                   # k 고르기: 이보다 적게 오르면 멈춤 (%p)


def score_all(db):
    """사진마다 정답 등수(동점은 가장 나쁜 쪽)·상위 5개."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / "probs_dev.csv", encoding="utf-8-sig"))}
    ocr = [r for r in csv.DictReader(open(m1.PRED["dev"], encoding="utf-8-sig")) if r["file_name"] in probs]
    assert len(ocr) == len(probs) == 468, (len(ocr), len(probs))
    saved = {r["file_name"]: r[SAVED_COL] for r in csv.DictReader(open(SAVED, encoding="utf-8-sig"))}
    rows = []
    for k, r in enumerate(sorted(ocr, key=lambda x: x["file_name"]), 1):
        fn, drug = r["file_name"], r["drug_N"]
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM)
        scores = {d: T.get(d, 0.0) + W * A[d] for d in A}
        pred, _ = m1.decide(scores)
        assert pred == saved[fn], (fn, pred, saved[fn])    # base 저장 결과와 사진마다 같은지
        ranked = m1.rank(scores)
        truth_score = scores[drug]
        worst = sum(v >= truth_score - 1e-9 for v in scores.values())   # 정답과 동점인 약까지 앞에 있다고 침
        best = sum(v > truth_score + 1e-9 for v in scores.values()) + 1
        assert (worst == 1) == (pred == drug), fn           # k = 1 은 base 정답과 같아야 함
        rows.append({"file_name": fn, "drug_N": drug, "truth": r["truth"], "read": r["read"], "aligned": r["aligned"], "camera_la": r["camera_la"],
                     "base_pred": pred, "rank": worst, "best_rank_if_tie": best, "tied_with_truth": worst - best,
                     "old_top3": int(drug in [d for d, _ in ranked[:3]]),
                     "top5": " ".join(f"{d}:{v:.3f}" for d, v in ranked[:5])})
        if k % 100 == 0:
            print(f"  {k}/468", flush=True)
    return rows


def line(name, rs):
    n = len(rs)
    s = {"group": name, "n": n}
    for k in KS:
        s[f"top{k}"] = round(100 * sum(r["rank"] <= k for r in rs) / max(n, 1), 1)
    s["outside5"] = sum(r["rank"] > 5 for r in rs)
    s["old_top3"] = round(100 * sum(r["old_top3"] for r in rs) / max(n, 1), 1)
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
    rows = score_all(db)
    rest = [r for r in rows if r["drug_N"] != "K-000147"]
    table = [line("dev 전체", rows), line("dev K-000147 제외", rest),
             line("긴 알약", [r for r in rows if r["aligned"] == "1"]), line("그 외", [r for r in rows if r["aligned"] != "1"]),
             line("카메라 75", [r for r in rows if r["camera_la"] == "75"]), line("카메라 90", [r for r in rows if r["camera_la"] == "90"])]
    assert table[0]["top1"] == 87.2, table[0]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["drug_N"]].append(r)
    per = sorted((dict(line(d, v), name=names[d]) for d, v in by.items()), key=lambda x: (x["top1"], x["top5"]))
    write_csv(OUT / "summary.csv", table)
    write_csv(OUT / "per_drug.csv", per)
    write_csv(OUT / "pred_dev.csv", rows)
    for s in table:
        print(s)
    pick = KS[0]
    for k in KS[1:]:
        if table[1][f"top{k}"] - table[1][f"top{k - 1}"] < MIN_GAIN:
            break
        pick = k
    print("규칙으로 고른 k =", pick)
    print("--- 약별 (Top-1 이 100 이 아닌 약)")
    for s in per:
        if s["top1"] < 100:
            print(s["group"], s["name"], s["n"], [s[f"top{k}"] for k in KS], "5등 밖", s["outside5"])
    print("동점이 있는 사진:", sum(r["tied_with_truth"] > 0 for r in rows), "| 정답 등수 분포:", sorted(collections.Counter(min(r["rank"], 6) for r in rows).items()))
    print("--- 5등 밖 사진")
    for r in rows:
        if r["rank"] > 5:
            print(r["file_name"], r["truth"], "| 읽기", r["read"], "| 등수", r["rank"], "| 동점", r["tied_with_truth"], "|", r["top5"][:45])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
