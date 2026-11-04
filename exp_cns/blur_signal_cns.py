# =====================================================================
#  blur_signal_cns.py  —  흐림 점수를 '다시 찍기 안내'·'후보 여러 개 표시' 신호로 쓸 수 있는지 재기 (OCR 안 돌림)
# =====================================================================
#
#  [방식]  흐림 점수(라플라시안 분산, 작을수록 흐림, simple_prep_v3\blur_scores.csv)를 검증 부분의 4분위 경계로 4구간으로 나누고
#          개발용에도 같은 경계를 씀. 구간마다 약 정확도·판정 없음·틀림·후보 표시(display_cns 규칙)의 정답 포함률을 계산
#  [규칙, 실행 전 고정]  가장 흐린 구간의 약 정확도가 나머지 구간보다 검증·개발 모두 10%p 이상 낮으면 '다시 찍기 안내' 후보
#          약 단위로 몰린 결과인지 보려고 같은 약 안에서 비교한 차이(within_drug_gap)도 함께 저장
#  [대상]  pft1 (λ 0.9, w 0.02), 검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\blur_signal_v1\bins.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\blur_signal_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import display_cns as dc
import match_cns as m1

RES = m1.cc.RESULTS
OUT = RES / "blur_signal_v1"
GAP_MIN = 10.0                                                       # 판단 규칙의 기준 (%p)


def pct(v, n):
    return round(100 * v / max(n, 1), 1)


def within_drug_gap(rows, q):
    """같은 약 안에서 '나머지 구간 정확도 − 이 구간 정확도' 의 약별 평균 (양쪽에 사진이 있는 약만)."""
    by = collections.defaultdict(lambda: ([], []))
    for x in rows:
        by[x["drug"]][0 if x["bin"] == q else 1].append(x["ok"])
    gaps = [100 * (sum(b) / len(b) - sum(a) / len(a)) for a, b in by.values() if a and b]
    return (round(sum(gaps) / len(gaps), 1) if gaps else ""), len(gaps)


def bin_rows(part, group, rows, cuts):
    out = []
    for q in range(4):
        cur = [x for x in rows if x["bin"] == q]
        rest = [x for x in rows if x["bin"] != q]
        n = len(cur)
        drugs = collections.Counter(x["drug"] for x in cur)
        kinds = collections.Counter(x["kind"] for x in cur)
        gap, both = within_drug_gap(rows, q)
        out.append({"part": part, "group": group, "bin": f"Q{q + 1}" + ("(가장 흐림)" if q == 0 else ""),
                    "blur_le": cuts[q] if q < 3 else "", "n": n, "drugs": len(drugs),
                    "top_drug": drugs.most_common(1)[0][0] if drugs else "", "top_drug_n": drugs.most_common(1)[0][1] if drugs else 0,
                    "acc": pct(sum(x["ok"] for x in cur), n), "rest_acc": pct(sum(x["ok"] for x in rest), len(rest)),
                    "within_drug_gap": gap, "drugs_both": both,
                    "read_exact": pct(sum(x["exact"] for x in cur), n),
                    "none": sum(x["pred"] == "" for x in cur), "wrong": sum(x["pred"] not in ("", x["drug"]) for x in cur),
                    "include_truth": pct(sum(x["has"] for x in cur), n),
                    "mean_shown": round(sum(x["size"] for x in cur) / max(n, 1), 2),
                    "sure_share": pct(kinds["sure"], n), "sure_wrong": sum(x["kind"] == "sure" and not x["has"] for x in cur)})
    return out


def main():
    if (OUT / "bins.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    blur = {r["file_name"]: (r["part"], float(r["blur"]))
            for r in csv.DictReader(open(RES / "simple_prep_v3" / "blur_scores.csv", encoding="utf-8-sig"))}
    vals = sorted(v for p, v in blur.values() if p == "val")
    cuts = [vals[len(vals) * k // 4] for k in (1, 2, 3)]             # 검증 부분으로만 정한 경계
    print("구간 경계 (검증):", cuts)
    db = m1.load_db()
    table = []
    for part in ("val", "dev"):
        rows = []
        for r, pred, ranked in dc.score("pft1", part, db):
            b = blur[r["file_name"]][1]
            kind, sh = dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)
            rows.append({"drug": r["drug_N"], "bin": sum(b > c for c in cuts), "pred": pred, "ok": pred == r["drug_N"],
                         "exact": r["correct"] == "1", "kind": kind, "has": r["drug_N"] in sh, "size": len(sh)})
        table += bin_rows(part, "전체", rows, cuts)
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            table += bin_rows(part, "K-000147 제외", [x for x in rows if x["drug"] != "K-000147"], cuts)
    with open(OUT / "bins.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(t)
    q1 = {(t["part"], t["group"]): t for t in table if t["bin"].startswith("Q1")}
    gaps = {k: round(t["rest_acc"] - t["acc"], 1) for k, t in q1.items()}
    ok = gaps[("val", "전체")] >= GAP_MIN and gaps[("dev", "전체")] >= GAP_MIN
    print("가장 흐린 구간의 정확도 차이 (나머지 − Q1):", gaps)
    print("판단 규칙 통과:", ok)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
