# =====================================================================
#  match_cns.py  —  약 판별: 각인 읽기 + 모양·제형·색 CNN → 검색 DB 에서 약 이름 고르기
# =====================================================================
#
#  [DB]  search_db_cns.csv = 학습용(trainall 323종) + 개발용(36종) + 정신계 약제 목록(cns_drugs_summary.xlsx, 852종)
#        → 겹침 0 이라 1,211종. 사진이 있는 약만 넣으면 후보가 적어 점수가 부풀려지므로, 사진 없는 852종을 '방해 후보'로 넣음.
#        (xlsx 에는 평가용 36종의 라벨 줄도 있지만 사진은 안 씀 — 실제로 쓸 때도 DB 에는 모든 약이 들어 있음)
#        각인 = norm_label(print_front / print_back) (영문·숫자만, 마크·한글만이면 빈칸), 모양·제형·색 = attr_cnn_cns.attrs 와 같은 묶음
#  [각인 점수 T]  exp300 보정 V3 그대로: 부분일치 + 헷갈리는 글자(0/O 등) + 후보 전체 중 최고점, 0.3 미만은 0
#        후보 = 모든 방향의 읽은 글자(all_dirs) + 고른 방향 상자 글자를 순서 바꿔 이어 붙인 것(상자 4개까지, 순서 뒤집힘 대비)
#        약마다 앞면·뒷면 각인 중 높은 쪽 (사진이 어느 면인지 모르므로)
#  [겉모습 점수 A]  log P(모양) + log P(제형) + Σ색 12칸 log(p 또는 1−p)  (확률은 0.02~0.98 로 자름)
#  [방식]
#     text      각인만 (1등 동점이면 판정 없음)
#     text_tie  각인 → 1등 동점이면 A 가 높은 약 (exp300 '색 동점 깨기'의 CNN 판)
#     text_wA   T + w·A  (w 는 검증 부분 val 에서 고름 — 개발용으로 고르지 않음)
#     attr      A 만 (참고)
#  [나누기]  w 고르기 = 학습용 검증 부분 val 65종 828장 (attr_cnn_v1 labels.csv 의 part=val, 읽기는 ocr_v1\trainall)
#            측정 = 개발용 36종 468장 (ocr_v1\dev\paddle_ft1_u2.0)
#  [저장]  exp_cns\results\match_v1\  search_db_cns.csv, pred_<part>.csv (1~3등 약·점수), summary.csv
#  [실행]  python exp_cns\match_cns.py --step db
#          python exp_cns\match_cns.py --step score
# =====================================================================

import argparse
import collections
import csv
import glob
import itertools
import json
import math
import os
import sys

import openpyxl

import common_cns as cc
import ocr_common_cns as oc
from attr_cnn_cns import COLORS, FORMS, SHAPES, attrs, drug_info

sys.path.insert(0, str(cc.ROOT / "exp300"))
from correct_exp300 import MIN_SIM, sim_v                            # noqa: E402  exp300 V3 와 같은 유사도

OUT = cc.RESULTS / "match_v1"
XLSX = cc.ROOT / "pill_image_testset" / "cns_drugs_summary.xlsx"
ATTR = cc.RESULTS / "attr_cnn_v1"
PRED = {"val": oc.OCR_OUT / "trainall" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv",
        "dev": oc.OCR_OUT / "dev" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"}
W_GRID = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
OPT = (True, True, 1.0)                                              # V3: 부분일치, 헷갈리는 글자, alpha 1


# ---------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------
def step_db():
    rows = []
    info = drug_info(cc.TRAIN_ROOT)
    tr ={it["drug_N"] for it in oc.load_split("trainall", 0)}
    dv = {it["drug_N"] for it in oc.load_split("dev", 0)}
    lab = {}                                                         # 약마다 첫 JSON (각인은 앞면·뒷면 라벨 둘 다 씀)
    for p in sorted(glob.glob(str(cc.TRAIN_ROOT / "labels" / "*.json"))):
        im = json.load(open(p, encoding="utf-8"))["images"][0]
        lab.setdefault(im["drug_N"], im)
    for d in sorted(tr | dv):
        im = lab[d]
        s, f, c, _ = info[d]
        rows.append(dict(drug_N=d, name=im["dl_name"], source="train" if d in tr else "dev",
                         front=cc.norm_label(im["print_front"])[0], back=cc.norm_label(im["print_back"])[0],
                         shape=s, form=f, colors="|".join(c)))
    ws = openpyxl.load_workbook(XLSX, read_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    head = next(it)
    seen = {r["drug_N"] for r in rows}
    n_x = 0
    for v in it:
        im = dict(zip(head, v))
        d = im["drug_N"]
        if not d or d in seen:
            continue
        seen.add(d)
        s, f, c = attrs(im, strict=False)
        rows.append(dict(drug_N=d, name=im["dl_name"], source="cns",
                         front=cc.norm_label(im["print_front"])[0], back=cc.norm_label(im["print_back"])[0],
                         shape=s, form=f, colors="|".join(c)))
        n_x += 1
    os.makedirs(OUT, exist_ok=True)
    with open(OUT / "search_db_cns.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    noimp = sum(not (r["front"] or r["back"]) for r in rows)
    same = collections.Counter(tuple(sorted(x for x in (r["front"], r["back"]) if x)) for r in rows if r["front"] or r["back"])
    print(f"저장: {OUT / 'search_db_cns.csv'}  {len(rows)}종 (학습 {len(tr)} + 개발 {len(dv)} + 정신계 목록 {n_x})")
    print(f"  영문·숫자 각인이 없는 약 {noimp}종 / 각인(앞뒤)이 다른 약과 똑같은 약 {sum(v for v in same.values() if v > 1)}종")


# ---------------------------------------------------------------------
# 점수
# ---------------------------------------------------------------------
def load_db():
    db = list(csv.DictReader(open(OUT / "search_db_cns.csv", encoding="utf-8-sig")))
    for r in db:
        r["imprints"] = [x for x in (r["front"], r["back"]) if x]
        r["cols"] = set(x for x in r["colors"].split("|") if x)
    return db


def candidates(r):
    """한 사진의 각인 후보: 모든 방향의 읽은 글자 + 고른 방향 상자 글자의 순서 바꾸기."""
    c = {s.split(":")[1] for s in r["all_dirs"].split(" | ")}
    pieces = [b[4] for b in json.loads(r["best_boxes"] or "[]") if b[4]]
    if 2 <= len(pieces) <= 4:
        c |= {"".join(p) for p in itertools.permutations(pieces)}
    return [x for x in c if 1 <= len(x) <= 14]


def text_scores(cands, db):
    out = {}
    for d in db:
        s = max((sim_v(c, imp, *OPT) for c in cands for imp in d["imprints"]), default=0.0)
        if s >= MIN_SIM:
            out[d["drug_N"]] = s
    return out


def attr_score(p, d):
    cl = lambda x: min(max(x, 0.02), 0.98)
    a = math.log(cl(p[f"s_{d['shape']}"])) + math.log(cl(p[f"f_{d['form']}"]))
    for c in COLORS:
        a += math.log(cl(p[f"c_{c}"]) if c in d["cols"] else 1 - cl(p[f"c_{c}"]))
    return a


def rank(scores):
    return sorted(scores.items(), key=lambda kv: -kv[1])


def decide(scores):
    """1등 약 (동점·비어 있음 = 판정 없음 '')."""
    r = rank(scores)
    if not r or (len(r) > 1 and abs(r[0][1] - r[1][1]) < 1e-9):
        return "", r[:3]
    return r[0][0], r[:3]


def predict_all(part, db):
    """사진마다 {방식: (1등, 1~3등)} — w 는 W_GRID 전부."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs), (part, len(ocr), len(probs))
    out = []
    for k, (fn, r) in enumerate(sorted(ocr.items()), 1):
        T = text_scores(candidates(r), db)
        A = {d["drug_N"]: attr_score(probs[fn], d) for d in db}
        res = {"text": decide(T), "attr": decide(A)}
        top = [d for d, v in T.items() if T and abs(v - max(T.values())) < 1e-9]
        res["text_tie"] = decide({d: A[d] for d in top}) if len(top) > 1 else res["text"]
        for w in W_GRID:
            res[f"wA_{w:g}"] = decide({d: T.get(d, 0.0) + w * A[d] for d in A})
        out.append((r, res))
        if k % 200 == 0:
            print(f"  {part} {k}/{len(ocr)}")
    return out


def metrics(out, key):
    n = len(out)
    hit = [r["drug_N"] == res[key][0] for r, res in out]
    top3 = [r["drug_N"] in [d for d, _ in res[key][1]] for r, res in out]
    none = sum(res[key][0] == "" for _, res in out)
    by = collections.defaultdict(list)
    for (r, _), h in zip(out, hit):
        by[r["drug_N"]].append(h)
    drug_acc = sum(sum(v) * 2 > len(v) for v in by.values()) / len(by)   # 약마다 사진 과반 맞힘
    g = lambda f: [h for (r, _), h in zip(out, hit) if f(r)]
    pct = lambda v: round(100 * sum(v) / max(len(v), 1), 1)
    return dict(acc=pct(hit), top3=pct(top3), none=none, drug_acc=round(100 * drug_acc, 1),
                long=pct(g(lambda r: r["aligned"] == "1")), other=pct(g(lambda r: r["aligned"] != "1")),
                cam75=pct(g(lambda r: r["camera_la"] == "75")), cam90=pct(g(lambda r: r["camera_la"] == "90")),
                ocr_exact=pct([r["correct"] == "1" for r, _ in out]), n=n)


def save_pred(part, out, keys):
    with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "drug_N", "truth", "read", "ocr_correct"] +
                   [f"{k}_{x}" for k in keys for x in ("pred", "ok", "top3")])
        for r, res in out:
            row = [r["file_name"], r["drug_N"], r["truth"], r["read"], r["correct"]]
            for k in keys:
                p, t3 = res[k]
                row += [p, int(p == r["drug_N"]), " ".join(f"{d}:{v:.3f}" for d, v in t3)]
            w.writerow(row)


def step_score():
    db = load_db()
    print(f"DB {len(db)}종")
    val = predict_all("val", db)
    wk = max(W_GRID, key=lambda w: (metrics(val, f"wA_{w:g}")["acc"], -w))   # 같으면 작은 w
    print("val 에서 w 고르기:", {f"{w:g}": metrics(val, f"wA_{w:g}")["acc"] for w in W_GRID}, "→ w =", f"{wk:g}")
    dev = predict_all("dev", db)
    keys = ["text", "text_tie", f"wA_{wk:g}", "attr"]
    rows = []
    for part, out in (("val", val), ("dev", dev)):
        save_pred(part, out, keys)
        for k in keys:
            m = metrics(out, k)
            rows.append(dict(part=part, method=k, **m))
            print(f"  {part:3s} {k:10s} 정확도 {m['acc']:5.1f}  3등안 {m['top3']:5.1f}  판정없음 {m['none']:3d}  "
                  f"약단위 {m['drug_acc']:5.1f}  긴 {m['long']:5.1f} 그외 {m['other']:5.1f}  75° {m['cam75']:5.1f} 90° {m['cam90']:5.1f}"
                  f"  (읽기 완전일치 {m['ocr_exact']})")
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {OUT / 'summary.csv'}, pred_val.csv, pred_dev.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["db", "score"])
    a = ap.parse_args()
    {"db": step_db, "score": step_score}[a.step]()


if __name__ == "__main__":
    main()
