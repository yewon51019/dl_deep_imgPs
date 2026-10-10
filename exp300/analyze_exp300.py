# =====================================================================
#  analyze_exp300.py  —  보정 실험 후속 분석 (OCR 다시 안 돌림, 몇 초)
# =====================================================================
#
#  [A. 분리 실험]  "후보 전체 사용"(V3)에서 부분일치·글자 통일이 각각 얼마나 기여했나?
#        후보만 / 후보+부분일치 / 후보+글자통일 / 후보+둘 다(=V3)  를 개발·시험 세트에서 비교
#        (부분일치 강도 alpha 0.5 / 1.0 도 같이)
#
#  [B. 남은 오류 분석]  V3 로도 틀린 사진을 원인별로 나눠 봄
#        ① 읽은 글자 없음      ② 읽긴 했는데 DB와 너무 다름(유사도<0.3)   ③ 동점이라 판정 못함
#        ④ DB상 각인이 똑같은 약끼리 헷갈림(각인만으로는 구분 불가)
#        ⑤ 정답 글자를 읽었는데 다른 약을 고름(고르는 방법 문제)
#        ⑥ 일부만 읽음   ⑦ 완전히 잘못 읽음
#      + 제형/앞뒷면/카메라각도/모양별 정확도, 가장 많이 틀린 약, 자주 헷갈린 약 쌍
#
#  [실행 예]
#    python analyze_exp300.py --dev pred_dev.csv --test pred_test.csv --db search_db.csv \
#        --base parseq:P0 --out analysis
# =====================================================================

import argparse
import csv
import os
from collections import Counter, defaultdict

from correct_exp300 import (MIN_SIM, candidates_of, decide, drug_scores, group, load_rows, drug_metrics)

NONE_LABEL = "(판정없음)"


def load_db_names(path):
    """검색 DB → {약ID: {'name':이름, 'imprints':[앞,뒤]}} (이름 열이 없으면 ID를 이름으로)"""
    db, names = [], {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            imps = [x for x in (r.get("front_norm", ""), r.get("back_norm", "")) if x]
            db.append({"drug_N": r["drug_N"], "imprints": imps})
            names[r["drug_N"]] = (r.get("dl_name") or r["drug_N"]).strip()
    return db, names


def predict_cands(row, db, opt):
    """V3 방식: 모든 각도 후보를 DB와 비교해 약마다 최고점 → 1등 (동점이면 판정없음). 점수표도 같이 돌려줌"""
    best = {}
    for c in candidates_of(row):
        for k, v in drug_scores(c, db, opt).items():
            best[k] = max(best.get(k, 0.0), v)
    return decide(best), best


# ----------------------------- A. 분리 실험 -----------------------------
ABLATIONS = [("후보 전체만(부분일치X, 글자통일X)", (False, False, 1.0)),
             ("+부분일치 (alpha=1.0)", (True, False, 1.0)),
             ("+부분일치 (alpha=0.5)", (True, False, 0.5)),
             ("+글자통일만", (False, True, 1.0)),
             ("+둘 다 (alpha=1.0) = V3", (True, True, 1.0)),
             ("+둘 다 (alpha=0.5)", (True, True, 0.5))]


def score_split(rows, db, base, opt):
    g = group(rows, [base])
    names = sorted(g)
    yt = [g[n][0]["truth_drug"] for n in names]
    yp = [predict_cands(g[n][0], db, opt)[0] or NONE_LABEL for n in names]
    return drug_metrics(yt, yp), yt, yp


def run_ablation(dev, test, db, base):
    base_pred = {}
    out = []
    for sp, rows in (("개발", dev), ("시험", test)):
        if rows is None:
            continue
        g = group(rows, [base])
        # V0 기준선 (다수결 1개, 기능 없음)
        n0 = sorted(g)
        yt0 = [g[n][0]["truth_drug"] for n in n0]
        yp0 = [decide(drug_scores(g[n][0]["pred_imprint"], db, (False, False, 1.0))) or NONE_LABEL for n in n0]
        base_pred[sp] = (yt0, yp0)
    print("\n[A. 분리 실험]  (단위 %, 새로맞힘/새로틀림 = V0 기준선 대비 사진 수)")
    head = f"{'방법':<34}"
    for sp in base_pred:
        head += f" | {sp+' 정확도':>9} {sp+' F1':>8} {'새맞':>4} {'새틀':>4}"
    print(head)
    line = f"{'V0 기준선(다수결 1개)':<34}"
    for sp, (yt, yp) in base_pred.items():
        a, p, r, f = drug_metrics(yt, yp)
        line += f" | {a*100:9.1f} {f*100:8.1f} {0:4d} {0:4d}"
    print(line)
    for name, opt in ABLATIONS:
        line = f"{name:<34}"
        for sp, rows in (("개발", dev), ("시험", test)):
            if rows is None:
                continue
            (a, p, r, f), yt, yp = score_split(rows, db, base, opt)
            y0 = base_pred[sp][1]
            fx = sum(t != b and t == n for t, b, n in zip(yt, y0, yp))
            bk = sum(t == b and t != n for t, b, n in zip(yt, y0, yp))
            line += f" | {a*100:9.1f} {f*100:8.1f} {fx:4d} {bk:4d}"
        print(line)


# ----------------------------- B. 남은 오류 분석 -----------------------------
def classify(row, db, drug_imps, opt):
    """한 사진의 결과를 '맞음' 또는 실패 원인으로 분류. (예측, 원인) 반환"""
    pred, scores = predict_cands(row, db, opt)
    truth = row["truth_drug"]
    if pred == truth:
        return pred, "맞음"
    cands = candidates_of(row)
    t_imp = row["truth_imprint"]
    if not pred:                                              # 판정 없음
        if not cands:
            return pred, "① 읽은 글자 없음"
        if not scores:
            return pred, "② 읽었지만 DB와 너무 다름(<0.3)"
        return pred, "③ 동점이라 판정 못함"
    if t_imp and t_imp in drug_imps.get(pred, []):            # 틀린 약이 정답과 같은 각인을 가짐
        return pred, "④ DB상 각인 동일(구분 불가)"
    if t_imp and t_imp in cands:
        return pred, "⑤ 정답 글자는 읽었는데 다른 약 선택"
    if t_imp and any(len(c) >= 2 and c in t_imp for c in cands):
        return pred, "⑥ 일부만 읽음"
    return pred, "⑦ 완전히 잘못 읽음"


def breakdown(rows_info, field):
    d = defaultdict(lambda: [0, 0])
    for r, ok in rows_info:
        d[r.get(field, "")][0] += ok
        d[r.get(field, "")][1] += 1
    return ", ".join(f"{k or '(빈값)'}={v[0]/v[1]*100:.0f}%({v[1]})" for k, v in sorted(d.items()))


def run_errors(rows, db, names, base, opt, out_dir, tag):
    g = group(rows, [base])
    drug_imps = {d["drug_N"]: d["imprints"] for d in db}
    info, cats, wrong_pairs, per_drug = [], Counter(), Counter(), defaultdict(lambda: [0, 0])
    out_rows = []
    for fn in sorted(g):
        r = g[fn][0]
        pred, cat = classify(r, db, drug_imps, opt)
        ok = cat == "맞음"
        info.append((r, ok))
        cats[cat] += 1
        per_drug[r["truth_drug"]][1] += 1
        per_drug[r["truth_drug"]][0] += ok
        if not ok and pred:
            wrong_pairs[(r["truth_drug"], pred)] += 1
        out_rows.append([fn, r["truth_drug"], names.get(r["truth_drug"], ""), r["truth_imprint"], pred or NONE_LABEL,
                         names.get(pred, ""), cat, r.get("form", ""), r.get("drug_dir", ""), r.get("camera_la", ""),
                         r.get("det_shape", ""), r["candidates"]])
    n = len(info)
    print(f"\n[B. 남은 오류 분석 — {tag} {n}장, 방법 V3, 기준 {base}]")
    print(f"  맞힘 {cats['맞음']}장 ({cats['맞음']/n*100:.1f}%),  틀림 {n-cats['맞음']}장")
    for k, v in sorted(cats.items()):
        if k != "맞음":
            print(f"   {k:<34} {v:3d}장  ({v/n*100:4.1f}%)")
    print("  제형별    :", breakdown(info, "form"))
    print("  앞/뒷면별 :", breakdown(info, "drug_dir"))
    print("  카메라각도:", breakdown(info, "camera_la"))
    print("  판정모양별:", breakdown(info, "det_shape"))
    worst = sorted(per_drug.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1]))[:8]
    print("  정확도 낮은 약 (이름 / 정답각인 / 맞힘/전체):")
    imp_of = {r["truth_drug"]: r["truth_imprint"] for r, _ in info}
    for k, (ok, tot) in worst:
        print(f"   - {names.get(k, k)[:24]:<24} {imp_of.get(k, ''):<14} {ok}/{tot}")
    if wrong_pairs:
        print("  자주 헷갈린 약 쌍 (정답 → 잘못 고른 약):")
        for (t, p), c in wrong_pairs.most_common(6):
            print(f"   - {names.get(t, t)[:18]} → {names.get(p, p)[:18]}  ×{c}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"errors_{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "truth_drug", "truth_name", "truth_imprint", "pred_drug", "pred_name", "원인",
                    "form", "drug_dir", "camera_la", "det_shape", "candidates"])
        w.writerows(out_rows)
    print(f"  → 사진별 원인 저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--db", required=True)
    ap.add_argument("--base", default="parseq:P0")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    db, names = load_db_names(a.db)
    dev = load_rows(a.dev)
    test = load_rows(a.test) if a.test else None
    run_ablation(dev, test, db, a.base)
    opt = (True, True, 1.0)                         # V3 설정
    for tag, rows in (("dev", dev), ("test", test)):
        if rows is not None:
            run_errors(rows, db, names, a.base, opt, a.out, tag)


if __name__ == "__main__":
    main()
