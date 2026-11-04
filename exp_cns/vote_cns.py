# =====================================================================
#  vote_cns.py  —  방향 투표 규칙 V1 시험: 불확실한 12방향 사진에서만 최다 득표 읽기를 믿기 (pft1 저장 읽기로 계산, OCR 안 돌림)
# =====================================================================
#
#  [기준]  pft1 (λ 0.9, w 0.02), 저장된 답과 사진마다 같은지 확인 (display_cns.score)
#  [V1, 실행 전 고정]
#     대상 = 12방향 사진(축 정렬 아님) 중 불확실한 사진 (display_cns 규칙: 1등 < 0.95 또는 1·2등 차이 < 0.1, 동점 포함)
#     ① 방향별 읽기를 헷갈리는 글자 통일(fold) 후 세어, 단독 최다 득표이고 3표 이상인 읽기를 찾음
#     ② 그 읽기와 각인이 (fold 후) 같은 약이 있으면 그 약들 안에서만 고름. 여러 개면 겉모습 점수 A, 그래도 같으면 판정 없음
#     ③ 그런 읽기나 약이 없으면 기존 답 유지
#  [채택 규칙, 실행 전 고정]  기준 대비 (좋아짐 − 나빠짐) 이 검증 +5장 이상, 개발 0장 이상
#  [대상]  검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\vote_v1\  summary.csv, changed.csv, pred_val.csv, pred_dev.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\vote_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import display_cns as dc
import match7_cns as m7
import match_cns as m1
from correct_exp300 import fold

OUT = m1.cc.RESULTS / "vote_v1"
MIN_VOTES = 3
BETTER_MIN, DEV_MIN = 5, 0


def top_read(r):
    """단독 최다 득표이고 MIN_VOTES 표 이상인 읽기 (fold 후). 없으면 None."""
    votes = collections.Counter(fold(t) for t in (s.split(":")[1] for s in r["all_dirs"].split(" | ")) if t).most_common(2)
    if not votes or votes[0][1] < MIN_VOTES or (len(votes) > 1 and votes[1][1] == votes[0][1]):
        return None
    return votes[0][0]


def apply_vote(r, pred, ranked, prob, db, folded):
    """(새 답, 적용 여부)."""
    if r["aligned"] == "1" or dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)[0] == "sure":
        return pred, False
    t = top_read(r)
    same = [d for d in db if t in folded[d["drug_N"]]] if t else []
    if not same:
        return pred, False
    return m1.decide({d["drug_N"]: m1.attr_score(prob, d) for d in same})[0], True


def predict(part, db, folded):
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out, used = [], 0
    for r, pred, ranked in dc.score("pft1", part, db):
        new, on = apply_vote(r, pred, ranked, probs[r["file_name"]], db, folded)
        used += on
        out.append((r, {"base": (pred, ranked[:3]), "V1": (new, ranked[:3])}))
    print(f"  {part} 규칙 적용 {used}장", flush=True)
    return out


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    folded = {d["drug_N"]: {fold(x) for x in d["imprints"]} for d in db}
    table, changed = [], []
    res = {}
    for part in ("val", "dev"):
        out = res[part] = predict(part, db, folded)
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for name in ("base", "V1"):
                table.append({"part": part, "group": gname, "config": name, **m7.measure(its, name)})
        for r, p in out:
            k = m7.kind(r["drug_N"], p["base"][0], p["V1"][0])
            if k:
                changed.append({"part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                "all_dirs": r["all_dirs"], "change": k, "pred_base": p["base"][0], "pred_new": p["V1"][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "base_pred", "V1_pred"])
            for r, p in out:
                w.writerow([r["file_name"], r["drug_N"], r["truth"], r["read"], p["base"][0], p["V1"][0]])
    m7.write_csv(OUT / "summary.csv", table)
    if changed:
        m7.write_csv(OUT / "changed.csv", changed)
    for t in table:
        print(t)
    for c in sorted(changed, key=lambda c: (c["part"], c["change"], c["drug_N"])):
        print(c["part"], c["change"], c["drug_N"], c["truth"], "| 읽기", c["read"], "|", c["pred_base"] or "(없음)", "→", c["pred_new"] or "(없음)")
    get = lambda part: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == "V1")
    v, d = get("val"), get("dev")
    vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
    print(f"V1: 검증 {vd:+d} (+{v['better']}/-{v['worse']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}) → {'통과' if vd >= BETTER_MIN and dd >= DEV_MIN else '미달'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
