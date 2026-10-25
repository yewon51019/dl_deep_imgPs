# =====================================================================
#  boost_exp300.py  —  (1) 글자 찾기 보강 + (2) 짧은 글자 충돌 막기, 개발용 51장에서만 (평가용 안 돌림)
# =====================================================================
#
#  [기준]  PARSeq 최종 방식 = 보정 V3 + 색 동점 깨기 (test_final/final_dev.csv 와 같은 코드·같은 색 대표값, 개발용은 자기 사진 뺀 대표값)
#  [1 글자 찾기 보강]  CRAFT 가 어떤 방향에서 박스를 0개 찾았을 때만 다시 찾음 (박스를 찾은 방향은 지금과 똑같음)
#     base   : 지금 그대로 (박스 없으면 알약 전체를 한 줄로 읽음)
#     up2    : 2배 확대 + CLAHE 이미지에서 CRAFT 로 다시 찾고, 좌표를 원래 크기로 돌려 원본(P0)에서 읽음
#     lowthr : CRAFT 검출 기준을 낮춰 다시 찾기 (text 0.7→0.4, low_text 0.4→0.3, link 0.4→0.3)
#     paddle : PaddleOCR 검출 상자를 대신 씀 — 대표 설정 paddleocr_t0 와 같게 검출 기준 thresh·box_thresh = 0.0
#              (알약 넓이 절반이 넘는 상자는 제외), 읽기는 PARSeq
#     all    : up2 → lowthr → paddle 순서로 하나라도 찾으면 사용
#  [2 짧은 글자 충돌 막기]  OCR 재실행 없이 동점 처리 규칙만 바꿈
#     S0 지금 그대로 / S1 한 글자 후보는 두 글자 이상 후보가 아무 약과도 안 맞을 때만 /
#     S2 동점이면 색보다 먼저 '그 약을 지지하는 방향 수'(V3 점수 0.5 이상 후보 수)가 많은 쪽 / S12 = S1 + S2
#     S3 한 글자 후보는 같은 글자가 2개 이상의 방향에서 나왔을 때만 씀 / S123 = S1 + S2 + S3
#  [--rescore]  OCR 을 다시 돌리지 않고 저장된 cands_*.csv 로 규칙만 다시 채점
#  [고르는 기준, 미리 정함]  개발용 정확도 최고 → 같으면 판정 없음이 적은 쪽 → 더 단순한 쪽. 1장 = 2%p 라 1~2장 차이는 우연 범위.
#  [저장]  --out: cands_{방식}.csv (방향별 후보·박스 출처), metrics.csv, updown.csv
#  [실행 예]  python exp300\boost_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#             --parseq exp300\results\test_final\final_dev.csv --parseq-vote exp300\results\pred_dev_parseq_P0_align_vote_cv410.csv ^
#             --out exp300\results\boost_cv410
# =====================================================================

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import torch                                         # Paddle 보다 먼저

import cv2
import numpy as np

import align_exp300 as al
import combined_exp300 as cb
import correct_exp300 as corr
import final_exp300 as fin
import run_exp300 as base
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

MODES = ["base", "up2", "lowthr", "paddle", "all"]
RULES = ["S0", "S1", "S2", "S12", "S3", "S123"]
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def boxes_from(hl, fl, W, H, scale=1.0):
    """EasyOCR detect 결과 → (x0,y0,x1,y1) 목록 (run_exp300.craft_boxes 와 같은 변환, scale 로 원래 크기로)"""
    boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in hl[0]]
    for poly in fl[0]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append((x0, y0, x1, y1))
    return out


_PD_T0 = {}


def paddle_t0_polys(g):
    """PaddleOCR 검출(대표 설정 paddleocr_t0 와 같게 검출 기준 thresh·box_thresh = 0.0)의 4점 상자.
       너무 작거나(넓이 40 미만) 알약 전체를 잡은(사진 넓이 절반 초과) 상자는 뺌. 읽기는 PARSeq 라 text_rec_score_thresh 는 해당 없음"""
    if "m" not in _PD_T0:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import TextDetection
        for kw in ({"enable_mkldnn": False}, {}):
            try:
                _PD_T0["m"] = TextDetection(thresh=0.0, box_thresh=0.0, **kw)
                break
            except (TypeError, ValueError):
                continue
    res = list(_PD_T0["m"].predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))[0]
    h, w = g.shape
    polys = [np.asarray(p, np.float32).reshape(-1, 2) for p in res["dt_polys"]]
    return [p for p in polys if 40 <= cv2.contourArea(p) <= al.PADDLE_MAX_AREA * h * w]


def alt_boxes(mode, g):
    H, W = g.shape
    r = base.easy_reader()
    if mode == "up2":
        g2 = base.clahe(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        return boxes_from(*r.detect(g2), W, H, scale=2.0)
    if mode == "lowthr":
        return boxes_from(*r.detect(g, text_threshold=0.4, low_text=0.3, link_threshold=0.3), W, H)
    if mode == "paddle":
        out = []
        for p in paddle_t0_polys(g):
            x0, y0 = np.floor(p.min(0)).astype(int)
            x1, y1 = np.ceil(p.max(0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                out.append((int(x0), int(y0), int(x1), int(y1)))
        return out
    if mode == "all":
        for m in ("up2", "lowthr", "paddle"):
            b = alt_boxes(m, g)
            if b:
                return b
        return []
    return []


class BoostOCR:
    """PARSeq 읽기. CRAFT 박스가 0개일 때만 mode 의 방법으로 다시 찾음. 호출마다 (박스 출처, 박스 수) 기록"""
    def __init__(self, parseq, mode):
        self.p, self.mode, self.log = parseq, mode, []

    def __call__(self, g):
        boxes, src = base.craft_boxes(g), "craft"
        if not boxes and self.mode != "base":
            boxes = alt_boxes(self.mode, g)
            src = self.mode if boxes else "none"
        elif not boxes:
            src = "none"
        n = len(boxes)
        boxes = boxes or [(0, 0, g.shape[1], g.shape[0])]
        items = []
        for x0, y0, x1, y1 in boxes:                      # _CraftThenRead.read_boxes 와 같은 읽기·이어 붙이기
            t, c = self.p.read_crop(g[y0:y1, x0:x1])
            t = "".join(ch for ch in t if ch.isalnum())
            if t:
                items.append((x0, y0, t, c))
        self.log.append((src, n))
        return base.join_boxes(items)


def decide(rule, texts, fn, aspect, ab, dbrow, protos, cdb):
    """보정 V3 + 색 동점 깨기 (+ 규칙 S1/S2). → (약, 근거)"""
    texts = [t for t in texts if 1 <= len(t) <= 12]
    if rule in ("S3", "S123"):                             # 한 글자 후보는 같은 글자가 2개 이상의 방향에서 나왔을 때만
        texts = [t for t in texts if len(t) >= 2 or texts.count(t) >= 2]
    if rule in ("S1", "S12", "S123"):
        long_t = [t for t in texts if len(t) >= 2]
        if cb.edit_scores(long_t, cdb):                    # 두 글자 이상 후보가 어떤 약이라도 맞으면 한 글자 후보는 뺌
            texts = long_t
    v3map = cb.edit_scores(texts, cdb)
    if rule in ("S0", "S1", "S3"):
        return fin.v3_color(v3map, ab, aspect, dbrow, protos)
    e = corr.decide(v3map)
    if e:
        return e, "글자"
    tied = cb.top_set(v3map)
    if len(tied) >= 2:
        sup = {d: sum(corr.drug_scores(t, cdb, cb.V3_OPT).get(d, 0) >= 0.5 for t in texts) for d in tied}
        top = max(sup.values())
        best = [d for d in tied if sup[d] == top]
        if len(best) == 1:
            return best[0], "지지 방향 수"
        d, why = cb.tiebreak(best, ab, aspect, dbrow, protos)
        if d:
            return d, why
    return "", "판정없음"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--parseq", required=True, help="test_final/final_dev.csv (기준: PARSeq 최종 방식)")
    ap.add_argument("--parseq-vote", required=True, help="기존 후보 (base 가 같은지 확인용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescore", action="store_true", help="OCR 없이 저장된 cands_*.csv 로 규칙만 다시 채점")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                          # 개발용만
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    cdb = corr.load_db(a.db)
    ab = {it["file_name"]: cb.pill_ab(base.imread_any(it["image_path"])) for it in items}
    protos = {it["file_name"]: cb.color_protos(items, ab, dbrow, leave_out=it["file_name"]) for it in items}
    crops = {it["file_name"]: base.crop_pill(base.imread_any(it["image_path"])) for it in items}
    ref = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    old = {r["file_name"]: json.loads(r["candidates"]) for r in csv.DictReader(open(a.parseq_vote, encoding="utf-8-sig"))}
    cands, times, srcs = {}, defaultdict(list), defaultdict(lambda: defaultdict(int))
    if a.rescore:                                          # 저장된 후보로 규칙만 다시 채점
        for mode in MODES:
            for r in csv.DictReader(open(os.path.join(a.out, f"cands_{mode}.csv"), encoding="utf-8-sig")):
                cl = json.loads(r["cands"])
                cands[(mode, r["file_name"])] = (float(r["aspect"]), [c[0] for c in cl])
                times[mode].append(float(r["ms"]))
                for c in cl:
                    srcs[mode][c[3]] += 1
            print(f"[{mode}] 저장된 후보 읽음, 사진당 평균 {np.mean(times[mode]) / 1000:.2f}초, 박스 출처 {dict(srcs[mode])}")
    parseq = None if a.rescore else base.PARSeq()

    for mode in ([] if a.rescore else MODES):
        ocr = BoostOCR(parseq, mode)
        rows = []
        for it in items:
            fn = it["file_name"]
            gray, aspect, color, ok = crops[fn]
            g0 = base.PREPS["P0"][1](gray)
            ocr.log = []
            t0 = time.time()
            _, _, c12, _ = base.read_with_rotation(g0, aspect, ocr, 30)
            times[mode].append((time.time() - t0) * 1000)
            for s, n in ocr.log:
                srcs[mode][s] += 1
            cands[(mode, fn)] = (aspect, [c[0] for c in c12])
            rows.append({"mode": mode, "file_name": fn, "aspect": round(aspect, 4), "ms": round(times[mode][-1]),
                         "cands": json.dumps([[c[0], round(c[1], 1), c[2], s, n] for c, (s, n) in zip(c12, ocr.log)], ensure_ascii=False)})
        with open(os.path.join(a.out, f"cands_{mode}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if mode == "base":
            diff = sum(cands[("base", fn)][1] != old[fn] for fn in old)
            print(f"[확인] base 후보가 기존 저장 후보와 다른 사진: {diff}/{len(old)}장")
        print(f"[{mode}] 읽기 완료, 사진당 평균 {np.mean(times[mode]) / 1000:.2f}초, 박스 출처 {dict(srcs[mode])}")

    # 채점
    res = []
    for mode in MODES:
        for rule in RULES:
            for it in items:
                fn = it["file_name"]
                aspect, texts = cands[(mode, fn)]
                d, why = decide(rule, texts, fn, aspect, ab[fn], dbrow, protos[fn], cdb)
                cl = [(t, 0.0, 0) for t in texts]
                bc = fin.best_candidate(cl, d, cdb) if d else None
                read = bc[0] if bc else ""
                res.append({"mode": mode, "rule": rule, "file_name": fn, "shape": ref[fn]["shape"], "truth_drug": it["drug_N"],
                            "pred_drug": d or NONE_LABEL, "drug_ok": int(d == it["drug_N"]), "read": read,
                            "judge": judge(read, base.norm(it["truth"])), "decided_by": why})
    with open(os.path.join(a.out, "results_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    chk = [r for r in res if r["mode"] == "base" and r["rule"] == "S0"]
    print(f"[확인] base × S0 가 기준(PARSeq 최종 방식)과 다른 사진: {sum(r['pred_drug'] != ref[r['file_name']]['pred_drug'] for r in chk)}/{len(chk)}장")

    met, ud = [], []
    print(f"\n[개발용 51장]  기준 = PARSeq 최종 방식(base × S0) 정확도 {np.mean([r['drug_ok'] for r in chk]) * 100:.1f}%   1장 = 2.0%p")
    print(f"  {'글자찾기 × 규칙':<16} " + " ".join(f"{k:>8}" for k in METRICS) + "  판정없음  둥근  길쭉   좋아짐/나빠짐   시간/장")
    for mode in MODES:
        for rule in RULES:
            sub = [r for r in res if r["mode"] == mode and r["rule"] == rule]
            acc, p, rc, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
            imp = np.mean([x["judge"] == "O" for x in sub])
            none = sum(x["pred_drug"] == NONE_LABEL for x in sub)
            rnd = np.mean([x["drug_ok"] for x in sub if x["shape"] == "둥근"])
            lng = np.mean([x["drug_ok"] for x in sub if x["shape"] == "길쭉"])
            up = [x["file_name"] for x in sub if x["drug_ok"] == 1 and ref[x["file_name"]]["drug_ok"] == "0"]
            dn = [x["file_name"] for x in sub if x["drug_ok"] == 0 and ref[x["file_name"]]["drug_ok"] == "1"]
            sec = np.mean(times[mode]) / 1000
            print(f"  {mode + ' × ' + rule:<16} " + " ".join(f"{v * 100:7.1f}%" for v in (imp, acc, p, rc, f1)) +
                  f"  {none:6d}  {rnd * 100:5.1f} {lng * 100:5.1f}    +{len(up)} / -{len(dn)}       {sec:.2f}초")
            met.append({"mode": mode, "rule": rule, **{k: round(v * 100, 1) for k, v in zip(METRICS, (imp, acc, p, rc, f1))},
                        "판정없음": none, "둥근_정확도": round(rnd * 100, 1), "길쭉_정확도": round(lng * 100, 1),
                        "좋아짐": len(up), "나빠짐": len(dn), "시간_초_장": round(float(sec), 2)})
            ud += [{"mode": mode, "rule": rule, "change": "좋아짐", "file_name": f} for f in up]
            ud += [{"mode": mode, "rule": rule, "change": "나빠짐", "file_name": f} for f in dn]
    for name, rows in (("metrics.csv", met), ("updown.csv", ud)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["mode"])
            w.writeheader()
            w.writerows(rows)
    print("\n[좋아진 / 나빠진 사진]")
    for x in ud:
        print(f"  {x['mode']} × {x['rule']}: {x['change']} {x['file_name']}")
    print(f"\n저장 완료: {a.out}")


if __name__ == "__main__":
    main()
