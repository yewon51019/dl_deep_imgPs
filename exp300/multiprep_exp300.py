# =====================================================================
#  multiprep_exp300.py  —  여러 전처리 후보 합치기 (P0 + P4), 개발용 51장에서만 (평가용 안 돌림)
# =====================================================================
#
#  [기준]  PARSeq 최종 방식 = 보정 V3 + 색 동점 깨기, P0 만 (test_final/final_dev.csv 와 같은 코드·같은 색 대표값)
#  [P4]   2배 확대 + CLAHE. 개발용 오류 7장 확인(check_dev_errors7.csv)에서 P1~P3 은 0장, P4 만 2장(LIT, ARICEPT)에서 정답 쪽 글자
#  [비교]  (OCR 은 P0·P4 를 한 번씩만 읽고, 아래는 저장된 후보로 고르는 방법만 다름)
#     B    : P0 후보만 (기준)
#     U    : P0 + P4 후보 전부 합쳐서 V3 + 색
#     F    : P0 로 먼저 정하고, P0 가 판정 없음이거나 V3 최고점이 LOW(0.5) 미만일 때만 P0 + P4 후보로 다시 정함
#     B1·U1·F1 : 위 셋에 S1 규칙(두 글자 이상 후보가 어떤 약과 맞으면 한 글자 후보는 뺌, boost_exp300.decide 와 같음)
#  [미리 정한 기준]  LOW = 0.5 (결과 보기 전에 정함). 고르는 기준: 개발용 정확도 최고 → 같으면 판정 없음이 적은 쪽 → 더 단순한 쪽.
#                   1장 = 2%p 라 1~2장 차이는 우연 범위. P4 는 오류 사진을 보고 고른 것이라 결과는 '개발용 추가 실험'으로만 보고.
#  [--rescore]  OCR 을 다시 돌리지 않고 저장된 cands_P0.csv / cands_P4.csv 로 다시 채점
#  [실행 예]  python exp300\multiprep_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#             --parseq exp300\results\test_final\final_dev.csv --out exp300\results\multiprep_cv410
# =====================================================================

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import torch                                         # Paddle 보다 먼저

import numpy as np

import combined_exp300 as cb
import correct_exp300 as corr
import final_exp300 as fin
import run_exp300 as base
from boost_exp300 import BoostOCR, decide
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

PREPS = ["P0", "P4"]
VARIANTS = ["B", "U", "F", "B1", "U1", "F1"]
LOW = 0.5
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def choose(var, c0, c4, fn, aspect, ab, dbrow, protos, cdb):
    """고르는 방법 하나 → (약, 근거, 쓴 후보)"""
    rule = "S1" if var.endswith("1") else "S0"
    kind = var[0]
    if kind == "B":
        return (*decide(rule, c0, fn, aspect, ab, dbrow, protos, cdb), c0)
    if kind == "U":
        return (*decide(rule, c0 + c4, fn, aspect, ab, dbrow, protos, cdb), c0 + c4)
    d, why = decide(rule, c0, fn, aspect, ab, dbrow, protos, cdb)
    v3 = cb.edit_scores([t for t in c0 if 1 <= len(t) <= 12], cdb)
    if d and v3 and max(v3.values()) >= LOW:
        return d, why, c0
    d, why = decide(rule, c0 + c4, fn, aspect, ab, dbrow, protos, cdb)
    return d, "P4 보탬:" + why, c0 + c4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--parseq", required=True, help="test_final/final_dev.csv (기준: PARSeq 최종 방식)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescore", action="store_true", help="OCR 없이 저장된 cands_*.csv 로 다시 채점")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                          # 개발용만
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    cdb = corr.load_db(a.db)
    ab = {it["file_name"]: cb.pill_ab(base.imread_any(it["image_path"])) for it in items}
    protos = {it["file_name"]: cb.color_protos(items, ab, dbrow, leave_out=it["file_name"]) for it in items}
    ref = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    cands, aspects, times, srcs = {}, {}, defaultdict(list), defaultdict(lambda: defaultdict(int))

    if a.rescore:
        for prep in PREPS:
            for r in csv.DictReader(open(os.path.join(a.out, f"cands_{prep}.csv"), encoding="utf-8-sig")):
                cl = json.loads(r["cands"])
                cands[(prep, r["file_name"])] = [c[0] for c in cl]
                aspects[r["file_name"]] = float(r["aspect"])
                times[prep].append(float(r["ms"]))
                for c in cl:
                    srcs[prep][c[3]] += 1
    else:
        ocr = BoostOCR(base.PARSeq(), "base")                               # CRAFT → PARSeq, 박스 없으면 알약 전체 (지금 방식 그대로)
        crops = {it["file_name"]: base.crop_pill(base.imread_any(it["image_path"])) for it in items}
        for prep in PREPS:
            rows = []
            for it in items:
                fn = it["file_name"]
                gray, aspect, color, ok = crops[fn]
                g = base.PREPS[prep][1](gray)
                ocr.log = []
                t0 = time.time()
                _, _, c12, _ = base.read_with_rotation(g, aspect, ocr, 30)
                times[prep].append((time.time() - t0) * 1000)
                for s, n in ocr.log:
                    srcs[prep][s] += 1
                cands[(prep, fn)] = [c[0] for c in c12]
                aspects[fn] = aspect
                rows.append({"prep": prep, "file_name": fn, "aspect": round(aspect, 4), "ms": round(times[prep][-1]),
                             "cands": json.dumps([[c[0], round(c[1], 1), c[2], s, n] for c, (s, n) in zip(c12, ocr.log)], ensure_ascii=False)})
            with open(os.path.join(a.out, f"cands_{prep}.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    for prep in PREPS:
        print(f"[{prep}] 사진당 평균 {np.mean(times[prep]) / 1000:.2f}초, 박스 출처(방향 수) {dict(srcs[prep])}")

    res = []
    for var in VARIANTS:
        for it in items:
            fn = it["file_name"]
            d, why, used = choose(var, cands[("P0", fn)], cands[("P4", fn)], fn, aspects[fn], ab[fn], dbrow, protos[fn], cdb)
            bc = fin.best_candidate([(t, 0.0, 0) for t in used], d, cdb) if d else None
            read = bc[0] if bc else ""
            res.append({"variant": var, "file_name": fn, "shape": ref[fn]["shape"], "truth_drug": it["drug_N"],
                        "pred_drug": d or NONE_LABEL, "drug_ok": int(d == it["drug_N"]), "read": read,
                        "judge": judge(read, base.norm(it["truth"])), "decided_by": why})
    with open(os.path.join(a.out, "results_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    chk = [r for r in res if r["variant"] == "B"]
    print(f"[확인] B 가 기준(PARSeq 최종 방식)과 다른 사진: {sum(r['pred_drug'] != ref[r['file_name']]['pred_drug'] for r in chk)}/{len(chk)}장")

    met, ud = [], []
    t0, t4 = np.mean(times["P0"]) / 1000, np.mean(times["P4"]) / 1000
    print(f"\n[개발용 51장]  기준 = B 정확도 {np.mean([r['drug_ok'] for r in chk]) * 100:.1f}%   1장 = 2.0%p")
    print(f"  {'방식':<4} " + " ".join(f"{k:>8}" for k in METRICS) + "  판정없음  둥근  길쭉   좋아짐/나빠짐  P4 씀")
    for var in VARIANTS:
        sub = [r for r in res if r["variant"] == var]
        acc, p, rc, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
        imp = np.mean([x["judge"] == "O" for x in sub])
        none = sum(x["pred_drug"] == NONE_LABEL for x in sub)
        rnd = np.mean([x["drug_ok"] for x in sub if x["shape"] == "둥근"])
        lng = np.mean([x["drug_ok"] for x in sub if x["shape"] == "길쭉"])
        up = [x["file_name"] for x in sub if x["drug_ok"] == 1 and ref[x["file_name"]]["drug_ok"] == "0"]
        dn = [x["file_name"] for x in sub if x["drug_ok"] == 0 and ref[x["file_name"]]["drug_ok"] == "1"]
        used4 = len(sub) if var[0] == "U" else sum(x["decided_by"].startswith("P4") for x in sub) if var[0] == "F" else 0
        sec = t0 + (t4 if var[0] == "U" else t4 * used4 / len(sub) if var[0] == "F" else 0)
        print(f"  {var:<4} " + " ".join(f"{v * 100:7.1f}%" for v in (imp, acc, p, rc, f1)) +
              f"  {none:6d}  {rnd * 100:5.1f} {lng * 100:5.1f}    +{len(up)} / -{len(dn)}      {used4:2d}장")
        met.append({"variant": var, **{k: round(v * 100, 1) for k, v in zip(METRICS, (imp, acc, p, rc, f1))},
                    "판정없음": none, "둥근_정확도": round(rnd * 100, 1), "길쭉_정확도": round(lng * 100, 1),
                    "좋아짐": len(up), "나빠짐": len(dn), "P4_쓴_사진": used4, "시간_초_장": round(float(sec), 2)})
        ud += [{"variant": var, "change": "좋아짐", "file_name": f} for f in up]
        ud += [{"variant": var, "change": "나빠짐", "file_name": f} for f in dn]
    for name, rows in (("metrics.csv", met), ("updown.csv", ud)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["variant"])
            w.writeheader()
            w.writerows(rows)
    print("\n[좋아진 / 나빠진 사진]")
    by = {(r["variant"], r["file_name"]): r for r in res}
    for x in ud:
        r = by[(x["variant"], x["file_name"])]
        print(f"  {x['variant']}: {x['change']} {x['file_name']}  정답 {r['truth_drug']} → {r['pred_drug']} ('{r['read']}', {r['decided_by']})")
    print(f"\n저장 완료: {a.out}")


if __name__ == "__main__":
    main()
