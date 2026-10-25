# =====================================================================
#  loose_exp300.py  —  넉넉하게 자르기 (L) 후보 추가, 개발용 51장에서만 (평가용 안 돌림)
# =====================================================================
#
#  [문제]  crop_pill 은 배경과 색이 다른 부분만 알약으로 봄 → 어두운 배경 + 진한 남색 절반(심발타 9543 30mg)은 배경으로 빠지고,
#          빠진 자리를 알약 중간 밝기로 칠해 '9543' 이 사라짐
#  [L 자르기]  1) segment_pill 마스크(지금과 같음)  2) 낮은 기준 t = min(Otsu 기준, 테두리 색 차이 99% 값 + 10) 으로 다시 나눠,
#              지금 마스크에 닿은 덩어리만 붙임  3) 붙인 넓이가 지금의 2.5배를 넘으면(그림자·배경 무늬) 지금 마스크 그대로
#              4) 최소 외접 사각형 + 10% 여유로 자르되, 바깥을 칠하지 않음  5) P0 (회색조, 긴 변 400)
#  [비교]  multiprep_cv410 의 저장된 P0·P4 후보를 그대로 쓰고 L 만 새로 읽음 (고르는 코드는 multiprep_exp300.choose / boost_exp300.decide)
#     U1  : P0 + P4 + S1 (multiprep 최고, 기준)        UL1 : P0 + P4 + L + S1
#     B   : P0 만 (PARSeq 최종 방식)                    BL  : P0 + L (S0)        L : L 만 (S0)
#     색·크기 동점 처리에 쓰는 비율(aspect)은 모두 지금 crop_pill 값 (기준과 같게)
#  [저장]  --out: cands_L.csv, results_dev.csv, metrics.csv, updown.csv, crop_check.csv (넓이 비율),
#          viz_loose_dev\ (넓이가 1.3배 넘게 바뀐 사진의 지금/L 자르기 비교 그림, 원본 사진 포함 → GitHub 에 안 올라감)
#  [실행 예]  python exp300\loose_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#             --parseq exp300\results\test_final\final_dev.csv --cands exp300\results\multiprep_cv410 --out exp300\results\loose_cv410
# =====================================================================

import argparse
import csv
import json
import os
import time

import torch                                         # Paddle 보다 먼저

import cv2
import numpy as np

import combined_exp300 as cb
import correct_exp300 as corr
import final_exp300 as fin
import run_exp300 as base
from boost_exp300 import BoostOCR, decide
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

VARIANTS = ["B", "L", "BL", "U1", "UL1"]
MAX_GROW = 2.5
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def loose_mask(bgr):
    """지금 마스크 + 낮은 기준으로 붙인 덩어리 → (마스크, 넓이 비율). 지금 마스크가 없으면 (None, 1)"""
    main = base.segment_pill(bgr)
    if main is None:
        return None, 1.0
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(5, int(min(h, w) * 0.04))                        # segment_pill 과 같은 테두리·색 차이
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.clip(np.linalg.norm(lab - np.median(border, axis=0), axis=2) * 2, 0, 255).astype(np.uint8)
    bd = np.concatenate([d[:b].ravel(), d[-b:].ravel(), d[:, :b].ravel(), d[:, -b:].ravel()])
    thr, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(max(thr, 20), max(np.percentile(bd, 99) + 10, 20))
    low = cv2.morphologyEx((d > t).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lbl = cv2.connectedComponents(low)
    ids = np.setdiff1d(np.unique(lbl[(main > 0) & (low > 0)]), [0])
    grown = ((main > 0) | np.isin(lbl, ids)).astype(np.uint8) * 255
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(main)
    cv2.drawContours(out, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    ratio = (out > 0).sum() / max((main > 0).sum(), 1)
    if ratio > MAX_GROW:
        return main, round(float(ratio), 2)
    return out, round(float(ratio), 2)


def loose_crop(bgr):
    """넉넉하게 자르기 → (흑백, 긴변/짧은변, 넓이 비율). crop_pill 과 같은 회전·10% 여유, 바깥은 칠하지 않음"""
    mask, ratio = loose_mask(bgr)
    if mask is None:
        g, asp, _, _ = base.crop_pill(bgr)
        return g, asp, 1.0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    H, W = bgr.shape[:2]
    rot = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.getRectSubPix(rot, (int(rw * 1.1) + 2, int(rh * 1.1) + 2), (cx, cy))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), rw / max(rh, 1.0), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--parseq", required=True, help="test_final/final_dev.csv (기준: PARSeq 최종 방식)")
    ap.add_argument("--cands", required=True, help="multiprep_cv410 (저장된 P0·P4 후보)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "viz_loose_dev"), exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                          # 개발용만
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    cdb = corr.load_db(a.db)
    ab = {it["file_name"]: cb.pill_ab(base.imread_any(it["image_path"])) for it in items}
    protos = {it["file_name"]: cb.color_protos(items, ab, dbrow, leave_out=it["file_name"]) for it in items}
    ref = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    cands, aspect = {}, {}
    for prep in ("P0", "P4"):
        for r in csv.DictReader(open(os.path.join(a.cands, f"cands_{prep}.csv"), encoding="utf-8-sig")):
            cands[(prep, r["file_name"])] = [c[0] for c in json.loads(r["cands"])]
            aspect[r["file_name"]] = float(r["aspect"])

    ocr = BoostOCR(base.PARSeq(), "base")
    rows, crops, times = [], [], []
    for it in items:
        fn = it["file_name"]
        bgr = base.imread_any(it["image_path"])
        g, asp, ratio = loose_crop(bgr)
        ocr.log = []
        t0 = time.time()
        _, _, c12, _ = base.read_with_rotation(base.PREPS["P0"][1](g), asp, ocr, 30)
        times.append((time.time() - t0) * 1000)
        cands[("L", fn)] = [c[0] for c in c12]
        rows.append({"file_name": fn, "aspect_L": round(asp, 4), "area_ratio": ratio, "ms": round(times[-1]),
                     "cands": json.dumps([[c[0], round(c[1], 1), c[2], s, n] for c, (s, n) in zip(c12, ocr.log)], ensure_ascii=False)})
        crops.append({"file_name": fn, "area_ratio": ratio, "aspect_now": round(aspect[fn], 3), "aspect_L": round(asp, 3),
                      "used_L_mask": int(ratio <= MAX_GROW)})
        if ratio > 1.3:                                                    # 바뀐 사진만 그림으로 확인
            g0 = base.crop_pill(bgr)[0]
            hh = 300
            pair = [cv2.resize(x, (int(x.shape[1] * hh / x.shape[0]), hh)) for x in (g0, g)]
            cv2.imencode(".png", np.hstack([pair[0], np.full((hh, 10), 255, np.uint8), pair[1]]))[1].tofile(
                os.path.join(a.out, "viz_loose_dev", f"{fn[:-4]}_ratio{ratio}.png"))
    for name, rr in (("cands_L.csv", rows), ("crop_check.csv", crops)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    print(f"[L] 사진당 평균 {np.mean(times) / 1000:.2f}초, 넓이 1.3배 넘게 바뀐 사진 {sum(c['area_ratio'] > 1.3 for c in crops)}장"
          f" (그중 {MAX_GROW}배 초과라 지금 마스크 그대로 {sum(c['area_ratio'] > MAX_GROW for c in crops)}장)")

    pick = {"B": ("S0", ["P0"]), "L": ("S0", ["L"]), "BL": ("S0", ["P0", "L"]), "U1": ("S1", ["P0", "P4"]), "UL1": ("S1", ["P0", "P4", "L"])}
    res = []
    for var in VARIANTS:
        rule, preps = pick[var]
        for it in items:
            fn = it["file_name"]
            texts = sum((cands[(p, fn)] for p in preps), [])
            d, why = decide(rule, texts, fn, aspect[fn], ab[fn], dbrow, protos[fn], cdb)
            bc = fin.best_candidate([(t, 0.0, 0) for t in texts], d, cdb) if d else None
            read = bc[0] if bc else ""
            res.append({"variant": var, "file_name": fn, "shape": ref[fn]["shape"], "truth_drug": it["drug_N"],
                        "truth": base.norm(it["truth"]), "pred_drug": d or NONE_LABEL, "drug_ok": int(d == it["drug_N"]),
                        "read": read, "judge": judge(read, base.norm(it["truth"])), "decided_by": why})
    with open(os.path.join(a.out, "results_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    print(f"[확인] B 가 기준과 다른 사진: {sum(r['pred_drug'] != ref[r['file_name']]['pred_drug'] for r in res if r['variant'] == 'B')}/51장")

    met, ud = [], []
    print(f"\n  {'방식':<4} " + " ".join(f"{k:>8}" for k in METRICS) + "  판정없음  둥근  길쭉   좋아짐/나빠짐(기준 B 대비)  심발타 각인")
    for var in VARIANTS:
        sub = [r for r in res if r["variant"] == var]
        acc, p, rc, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
        imp = np.mean([x["judge"] == "O" for x in sub])
        none = sum(x["pred_drug"] == NONE_LABEL for x in sub)
        rnd = np.mean([x["drug_ok"] for x in sub if x["shape"] == "둥근"])
        lng = np.mean([x["drug_ok"] for x in sub if x["shape"] == "길쭉"])
        up = [x["file_name"] for x in sub if x["drug_ok"] == 1 and ref[x["file_name"]]["drug_ok"] == "0"]
        dn = [x["file_name"] for x in sub if x["drug_ok"] == 0 and ref[x["file_name"]]["drug_ok"] == "1"]
        cym = [f"{x['read']}({x['judge']})" for x in sub if x["truth_drug"] == "K-021930"]
        print(f"  {var:<4} " + " ".join(f"{v * 100:7.1f}%" for v in (imp, acc, p, rc, f1)) +
              f"  {none:6d}  {rnd * 100:5.1f} {lng * 100:5.1f}    +{len(up)} / -{len(dn)}                 {' '.join(cym)}")
        met.append({"variant": var, **{k: round(v * 100, 1) for k, v in zip(METRICS, (imp, acc, p, rc, f1))},
                    "판정없음": none, "둥근_정확도": round(rnd * 100, 1), "길쭉_정확도": round(lng * 100, 1),
                    "좋아짐": len(up), "나빠짐": len(dn), "심발타_읽음": " ".join(cym)})
        ud += [{"variant": var, "change": "좋아짐", "file_name": f} for f in up]
        ud += [{"variant": var, "change": "나빠짐", "file_name": f} for f in dn]
    for name, rr in (("metrics.csv", met), ("updown.csv", ud)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]) if rr else ["variant"])
            w.writeheader()
            w.writerows(rr)
    by = {(r["variant"], r["file_name"]): r for r in res}
    print("\n[좋아진 / 나빠진 사진]")
    for x in ud:
        r = by[(x["variant"], x["file_name"])]
        print(f"  {x['variant']}: {x['change']} {x['file_name']}  정답 {r['truth_drug']} → {r['pred_drug']} ('{r['read']}', {r['decided_by']})")
    print("\n[각인 판정이 U1 과 달라진 사진 (UL1)]")
    for it in items:
        u, l = by[("U1", it["file_name"])], by[("UL1", it["file_name"])]
        if u["judge"] != l["judge"]:
            print(f"  {it['file_name']}  정답 '{u['truth']}'  U1 '{u['read']}'({u['judge']}) → UL1 '{l['read']}'({l['judge']})")
    print(f"\n저장 완료: {a.out}")


if __name__ == "__main__":
    main()
