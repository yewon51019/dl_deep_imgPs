# =====================================================================
#  correct_exp300.py  —  각인 읽기 '보정' 실험 (OCR을 다시 돌리지 않음!)
# =====================================================================
#
#  [아이디어]
#   pred_dev.csv / pred_test.csv 에는 모델이 읽은 결과(여러 각도의 후보 'candidates')가 이미 저장돼 있어요.
#   그래서 "후보 중 무엇을 고르고, DB와 어떻게 비교하느냐"만 바꿔서 점수를 다시 계산합니다. (몇 초면 끝)
#
#  [시험하는 보정 방법]  (앞의 것을 누적해서 쌓아 감 → 한 줄씩 효과를 볼 수 있어요)
#   V0 기준선        : 지금 방식 (다수결로 고른 글자 1개 → 편집거리 유사도)
#   V1 +부분일치     : 'SAMJIN'처럼 정답의 '일부만' 읽었을 때, DB 각인 안에 들어있으면 점수를 올려 줌
#   V2 +헷갈리는 글자: 0↔O, 1↔I↔L, 5↔S, 8↔B, 2↔Z, 6↔G 를 같은 글자로 보고 비교
#   V3 +후보 전체(최고): 다수결 대신, 모든 각도 후보를 DB와 비교해 가장 잘 맞는 하나를 고름
#   V4 +후보 전체(합산): 약마다 모든 각도 후보의 점수를 더해서 가장 높은 약을 고름 (여러 번 일관되게 나온 쪽이 유리)
#   V5 +모델 합치기  : 여러 모델(--fuse)의 후보를 한꺼번에 합산
#
#  [공정성 규칙]  ← 중요!
#   · 방법은 '개발 세트(pred_dev.csv)' 점수로만 고릅니다.
#   · 시험 세트(pred_test.csv)는 고른 방법을 '한 번' 적용해 보는 용도예요. 보고 다시 고르면 안 돼요.
#
#  [실행 예]
#    python correct_exp300.py --dev pred_dev.csv --test pred_test.csv --db search_db.csv \
#        --base parseq:P0 --fuse parseq:P0,trocr:P3,easyocr:P0 --out corrected
# =====================================================================

import argparse
import csv
import json
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np

from evaluate_exp300 import drug_metrics, levenshtein as _lev

MIN_SIM = 0.3          # run_exp300.py 와 동일: 이보다 안 비슷하면 '판정 없음'
NONE = ""

# 헷갈리는 글자 묶음 (각 묶음의 첫 글자로 통일해서 비교)
CONFUSABLE = ["0OQD", "1IL", "5S", "8B", "2Z", "6G", "UV"]
FOLD = {c: g[0] for g in CONFUSABLE for c in g}


def fold(s):
    return "".join(FOLD.get(c, c) for c in s)


@lru_cache(maxsize=None)
def lev(a, b):
    return _lev(a, b)


def sim_edit(a, b):
    """기존 유사도: (L-편집거리)/(L+편집거리)"""
    if not a or not b:
        return 0.0
    L, ed = max(len(a), len(b)), lev(a, b)
    return (L - ed) / (L + ed)


def sim_v(a, b, use_partial, use_fold, alpha):
    """보정 옵션을 반영한 유사도.
       부분일치: a(읽은 것)가 b(DB 각인) 안에 통째로 들어 있으면 0.5 + 0.5*alpha*길이비율
                 (너무 짧은 1글자는 아무 데나 들어가므로 2글자 이상만 인정)"""
    if use_fold:
        a, b = fold(a), fold(b)
    s = sim_edit(a, b)
    if use_partial and 2 <= len(a) < len(b) and a in b:
        s = max(s, 0.5 + 0.5 * alpha * len(a) / len(b))      # 일부만 읽어도 최소 0.5점, 길게 읽을수록 높게
    return s


def load_db(path):
    db = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            db.append({"drug_N": r["drug_N"],
                       "imprints": [x for x in (r.get("front_norm", ""), r.get("back_norm", "")) if x]})
    return db


def drug_scores(text, db, opt):
    """한 글자열(text)이 각 약과 얼마나 비슷한지 {약: 점수} (앞/뒷면 중 높은 쪽)"""
    out = {}
    if not text:
        return out
    for d in db:
        s = max((sim_v(text, imp, *opt) for imp in d["imprints"]), default=0.0)
        if s >= MIN_SIM:
            out[d["drug_N"]] = s
    return out


def decide(score_map):
    """점수가 가장 높은 약. 1·2등 동점이거나 비어 있으면 '판정 없음'"""
    if not score_map:
        return NONE
    ranked = sorted(score_map.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        return NONE
    return ranked[0][0]


def candidates_of(row):
    return [c for c in json.loads(row["candidates"]) if 1 <= len(c) <= 12]


# ---------- 각 방법: (한 사진의 모델별 행들) -> 예측 약 ----------
def predict(rows, db, variant, alpha):
    """rows: 같은 사진에 대한 행 목록 (fuse 가 아니면 1개). variant: 'V0'..'V5'"""
    # V1: 부분일치만 / V2: 헷갈리는 글자만 / V3 이후: 둘 다 + 후보 전체 사용
    use_partial = variant in ("V1", "V3", "V4", "V5")
    use_fold = variant in ("V2", "V3", "V4", "V5")
    opt = (use_partial, use_fold, alpha)
    row = rows[0]
    if variant in ("V0", "V1", "V2"):                       # 다수결로 고른 글자 1개만 사용
        return decide(drug_scores(row["pred_imprint"], db, opt))
    if variant == "V3":                                     # 후보 전체 중 가장 잘 맞는 하나
        best = {}
        for c in candidates_of(row):
            for k, v in drug_scores(c, db, opt).items():
                best[k] = max(best.get(k, 0.0), v)
        return decide(best)
    total = defaultdict(float)                              # V4: 1개 모델 후보 합산 / V5: 여러 모델 합산
    for r in (rows if variant == "V5" else rows[:1]):
        for c in candidates_of(r):
            for k, v in drug_scores(c, db, opt).items():
                total[k] += v
    return decide(total)


VARIANTS = [("V0", "기준선(지금 방식)"), ("V1", "+부분일치"), ("V2", "+헷갈리는 글자(0/O 등)"),
            ("V3", "+부분일치·헷갈리는글자+후보전체(최고점)"), ("V4", "+후보전체 합산"), ("V5", "+모델 합치기")]


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group(rows, combos):
    """{사진: [조합별 행...]} (combos 순서대로, 하나라도 없는 사진은 제외)"""
    by = defaultdict(dict)
    for r in rows:
        if r["combo"] in combos:
            by[r["file_name"]][r["combo"]] = r
    return {fn: [d[c] for c in combos] for fn, d in by.items() if all(c in d for c in combos)}


def run_split(rows, db, base, fuse, alpha):
    """한 세트에 모든 방법을 적용 → {variant: (표 값, 예측 목록)}"""
    res = {}
    for v, _ in VARIANTS:
        combos = fuse if v == "V5" else [base]
        g = group(rows, combos)
        names = sorted(g)
        y_true = [g[n][0]["truth_drug"] for n in names]
        y_pred = [predict(g[n], db, v, alpha) or "(판정없음)" for n in names]
        res[v] = (drug_metrics(y_true, y_pred), names, y_true, y_pred)
    return res


def fix_break(res, v):
    """기준선 대비 새로 맞힌/새로 틀린 사진 수 (점수 변화가 우연인지 가늠하는 용도)"""
    _, n0, t0, p0 = res["V0"]
    _, n1, t1, p1 = res[v]
    if n0 != n1:                       # 사진 집합이 다르면(합치기에서 일부 모델 누락) 공통만 비교
        common = set(n0) & set(n1)
        d0 = {n: (t, p) for n, t, p in zip(n0, t0, p0) if n in common}
        d1 = {n: (t, p) for n, t, p in zip(n1, t1, p1) if n in common}
        pairs = [(d0[n], d1[n]) for n in common]
    else:
        pairs = [((t, a), (t, b)) for t, a, b in zip(t0, p0, p1)]
    fixed = sum(a[0] != a[1] and b[0] == b[1] for a, b in pairs)
    broke = sum(a[0] == a[1] and b[0] != b[1] for a, b in pairs)
    return fixed, broke


def print_table(title, res):
    print(f"\n[{title}]  (단위 %, 새로맞힘/새로틀림 = V0 대비 사진 수)")
    print(f"{'방법':<4} {'설명':<34} | {'정확도':>6} {'정밀도':>6} {'재현율':>6} {'F1':>6} | 새로맞힘 새로틀림")
    for v, desc in VARIANTS:
        if v not in res:
            continue
        (a, p, r, f), *_ = res[v]
        fx, bk = fix_break(res, v)
        print(f"{v:<4} {desc:<34} | {a*100:6.1f} {p*100:6.1f} {r*100:6.1f} {f*100:6.1f} | {fx:6d} {bk:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="개발 세트 예측 csv (방법을 고르는 용도)")
    ap.add_argument("--test", default=None, help="시험 세트 예측 csv (고른 방법을 한 번만 적용)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--base", default="parseq:P0", help="단일 모델 방법(V0~V4)에 쓸 조합")
    ap.add_argument("--fuse", default="parseq:P0,trocr:P3,easyocr:P0", help="V5에서 합칠 조합들(쉼표)")
    ap.add_argument("--alpha", type=float, default=1.0, help="부분일치 가중(0~1, 1이면 길이비율 그대로)")
    ap.add_argument("--out", default="corrected")
    a = ap.parse_args()
    fuse = [c.strip() for c in a.fuse.split(",") if c.strip()]
    db = load_db(a.db)
    os.makedirs(a.out, exist_ok=True)

    dev = run_split(load_rows(a.dev), db, a.base, fuse, a.alpha)
    print_table(f"개발 세트 / 기준 {a.base}, 합치기 {'+'.join(fuse)}", dev)
    best = max(dev, key=lambda v: (dev[v][0][3], -VARIANTS.index(next(x for x in VARIANTS if x[0] == v))))
    print(f"\n▶ 개발 세트 F1 이 가장 높은 방법: {best}  (동점이면 단순한 쪽)")

    test = None
    if a.test:
        test = run_split(load_rows(a.test), db, a.base, fuse, a.alpha)
        print_table("시험 세트 (참고: 개발 세트에서 고른 방법만 최종 성적으로 인정)", test)
        (ta, tp, tr, tf), *_ = test[best]
        print(f"\n★ 최종 보고용: 방법 {best} → 시험 세트 정확도 {ta*100:.1f} / 정밀도 {tp*100:.1f} / "
              f"재현율 {tr*100:.1f} / F1 {tf*100:.1f}   (기준선 V0 F1 {test['V0'][0][3]*100:.1f})")

    for name, res in (("dev", dev), ("test", test)):
        if res is None:
            continue
        with open(os.path.join(a.out, f"corrected_{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["variant", "file_name", "truth_drug", "pred_drug"])
            for v, _ in VARIANTS:
                _, names, yt, yp = res[v]
                for n, t, p in zip(names, yt, yp):
                    w.writerow([v, n, t, p])
    print(f"\n저장 폴더: {a.out}/ (corrected_dev.csv, corrected_test.csv)")


if __name__ == "__main__":
    main()
