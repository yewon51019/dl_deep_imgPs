# =====================================================================
#  match2_cns.py  —  약 판별 v2: match_cns.py(v1) 의 각인 점수 T 를 세 가지 고침 (DB·겉모습 점수 A 는 v1 그대로)
# =====================================================================
#
#  [v1 판정없음 원인 → 고친 것]  (원인은 개발용 오답을 보고 찾았으므로, 숫자는 모두 검증 부분 val 에서 고름)
#     ① 다른 방향의 엉뚱한 짧은 읽기('B','F' 등)가 1글자 각인 약과 완전일치해 동점
#        → 고른 방향(확신도 최고) 후보와 다른 방향 후보를 나눠 T = max(T_고른, λ·T_다른)   λ = 1 이면 v1 과 같음
#     ② 14글자 넘는 읽기(COMPAGNEROXAN75MG)를 후보에서 뺌 → 길이 제한 30
#     ③ 'DB 각인이 읽은 글자 안에 통째로 들어 있음' 부분일치 없음
#        → rev: 2글자 이상 DB 각인 b 가 읽기 a 안에 있으면 max(유사도, 0.5 + 0.5·len(b)/len(a))  (V3 부분일치의 반대 방향)
#  [고르기]  val(65종 828장)에서 λ × rev(끄기/켜기) × w 를 정확도로 고름. 같으면 v1 에 가까운 쪽(λ 큰 쪽, rev 끄기, w 작은 쪽)
#  [측정]   개발용 468장: 고른 설정 + 하나씩만 바꾼 것(①만, ②③만) + v1 재현(λ 1, rev 끄기, 제한 14)
#  [저장]   exp_cns\results\match_v2\  pred_<part>.csv, summary.csv, grid_val.csv (val 의 모든 조합 정확도)
#  [실행]   python exp_cns\match2_cns.py
# =====================================================================

import csv
import itertools
import json
import os

import match_cns as m1
from correct_exp300 import MIN_SIM, fold, sim_v                     # match_cns 가 exp300 경로를 sys.path 에 넣어 둠

OUT = m1.cc.RESULTS / "match_v2"
LAMBDAS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.0]
MAX_LEN = 30


def split_candidates(r, max_len):
    """(고른 방향 후보, 다른 방향 후보). 고른 방향 = 읽은 글자 + 고른 방향 상자 글자의 순서 바꾸기."""
    main = {r["read"]}
    pieces = [b[4] for b in json.loads(r["best_boxes"] or "[]") if b[4]]
    if 2 <= len(pieces) <= 4:
        main |= {"".join(p) for p in itertools.permutations(pieces)}
    other = {s.split(":")[1] for s in r["all_dirs"].split(" | ")} - main
    keep = lambda c: [x for x in c if 1 <= len(x) <= max_len]
    return keep(main), keep(other)


def sim_rev(a, b):
    """V3 유사도 + 반대 방향 부분일치 (DB 각인 b 가 읽기 a 안에 통째로)."""
    s = sim_v(a, b, *m1.OPT)
    fa, fb = fold(a), fold(b)
    if 2 <= len(fb) < len(fa) and fb in fa:
        s = max(s, 0.5 + 0.5 * len(fb) / len(fa))
    return s


def text_scores(cands, db, rev):
    f = sim_rev if rev else (lambda a, b: sim_v(a, b, *m1.OPT))
    out = {}
    for d in db:
        s = max((f(c, imp) for c in cands for imp in d["imprints"]), default=0.0)
        if s >= MIN_SIM:
            out[d["drug_N"]] = s
    return out


def combine(tm, to, lam):
    T = dict(tm)
    for d, v in to.items():
        if lam * v >= MIN_SIM and lam * v > T.get(d, 0.0):
            T[d] = lam * v
    return T


def predict_all(part, db, configs):
    """configs = [(이름, λ, rev, 길이 제한)] → 사진마다 {이름_text / 이름_tie / 이름_wA_w: (1등, 1~3등)}."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(m1.PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs), (part, len(ocr), len(probs))
    out = []
    for k, (fn, r) in enumerate(sorted(ocr.items()), 1):
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        cache, res = {}, {}
        for name, lam, rev, ml in configs:
            if (rev, ml) not in cache:
                cm, co = split_candidates(r, ml)
                cache[(rev, ml)] = (text_scores(cm, db, rev), text_scores(co, db, rev))
            T = combine(*cache[(rev, ml)], lam)
            res[f"{name}_text"] = m1.decide(T)
            top = [d for d, v in T.items() if abs(v - max(T.values())) < 1e-9]
            res[f"{name}_tie"] = m1.decide({d: A[d] for d in top}) if len(top) > 1 else res[f"{name}_text"]
            for w in m1.W_GRID:
                res[f"{name}_wA_{w:g}"] = m1.decide({d: T.get(d, 0.0) + w * A[d] for d in A})
        out.append((r, res))
        if k % 200 == 0:
            print(f"  {part} {k}/{len(ocr)}")
    return out


def main():
    db = m1.load_db()
    m1.OUT = OUT                                                     # save_pred 가 쓰는 폴더 (v1 파일은 안 건드림)
    os.makedirs(OUT, exist_ok=True)
    grid = [(f"L{lam:g}_r{int(rev)}", lam, rev, MAX_LEN) for lam in LAMBDAS for rev in (False, True)]
    val = predict_all("val", db, grid + [("v1", 1.0, False, 14)])
    rows = []
    for name, lam, rev, _ in grid:
        for w in m1.W_GRID:
            rows.append(dict(config=name, lam=lam, rev=int(rev), w=w, acc=m1.metrics(val, f"{name}_wA_{w:g}")["acc"]))
    with open(OUT / "grid_val.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    best = max(rows, key=lambda x: (x["acc"], x["lam"], -x["rev"], -x["w"]))
    lam, rev, w = best["lam"], bool(best["rev"]), best["w"]
    print(f"val 에서 고름: λ = {lam:g}, rev = {int(rev)}, w = {w:g}  (val 정확도 {best['acc']})")
    v1w = max(m1.W_GRID, key=lambda x: (m1.metrics(val, f"v1_wA_{x:g}")["acc"], -x))
    dev_cfg = [("v1", 1.0, False, 14), ("best", lam, rev, MAX_LEN),
               ("only_lam", lam, False, 14), ("only_rev", 1.0, rev, MAX_LEN)]
    dev = predict_all("dev", db, dev_cfg)
    keys = ["v1_text", "v1_tie", f"v1_wA_{v1w:g}", "best_text", "best_tie", f"best_wA_{w:g}",
            f"only_lam_wA_{w:g}", f"only_rev_wA_{w:g}"]
    out_rows = []
    for part, out in (("val", val), ("dev", dev)):
        # val 에는 하나씩 바꾼 설정을 따로 안 돌렸으므로 v1 과 고른 설정(격자 이름)만
        pk = keys if part == "dev" else [k.replace("best_", best["config"] + "_") for k in keys[:6]]
        m1.save_pred(part, out, pk)
        for k in pk:
            mm = m1.metrics(out, k)
            out_rows.append(dict(part=part, method=k, **mm))
            print(f"  {part:3s} {k:16s} 정확도 {mm['acc']:5.1f}  3등안 {mm['top3']:5.1f}  판정없음 {mm['none']:3d}  "
                  f"약단위 {mm['drug_acc']:5.1f}  긴 {mm['long']:5.1f} 그외 {mm['other']:5.1f}  75° {mm['cam75']:5.1f} 90° {mm['cam90']:5.1f}")
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"저장: {OUT} (summary.csv, grid_val.csv, pred_val.csv, pred_dev.csv)")


if __name__ == "__main__":
    main()
