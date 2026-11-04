# =====================================================================
#  error_cause_cns.py  —  약 판별 오답 원인 분류 (개발용, 주 방식 base = match_v3\base)
# =====================================================================
#
#  [대상]  match_v3\base\pred_dev.csv 에서 약을 틀린 사진 (판정없음 + 다른 약)
#  [자동 분류]  (앞 칸부터 먼저 해당하는 것, 유사도 = exp300 V3 sim_v)
#     A  각인 같은 약   : 1등 동점에 정답 약이 있고, 동점 약들이 정답 각인과 똑같은 각인을 가짐 (겉모습 점수도 같음 = DB 모양·제형·색도 같음)
#     B1 다른 면 읽음   : 읽은 글자가 정답과 거의 안 닮았고(< 0.3) 같은 약의 다른 면 각인과는 0.7 이상 (라벨 앞뒤 뒤바뀜 의심)
#     B2 다른 약 각인   : 읽은 글자가 정답과 거의 안 닮았고 DB 다른 약 각인과 0.8 이상 (사진·라벨 불일치 의심, 예 K-000147)
#     C  순서·방향      : 읽은 글자를 정렬하면 정답과 같음
#     D  일부만 읽음    : 정답 글자의 절반 이상이 읽은 글자에 있음
#     E0 상자 없음 / E 엉뚱하게 읽음
#     같이 적음: kind(판정없음/다른 약), 정답 약 3등 안, 긴 알약(축 정렬) 여부, 제형, 카메라, 동점·1등 약 이름
#  [저장]  match_v3\base\errors\error_list.csv (note 칸 = 사람·Claude 가 그림 보고 적는 칸), error_summary.csv,
#          sheet_01.png ~ (한 장에 6장: dt 그림 + 정답·읽음·1~3등, 자동 분류 순으로)
#  [실행]  python exp_cns\error_cause_cns.py
# =====================================================================

import collections
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_cns as cc
import match_cns as m1
from correct_exp300 import sim_v

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = cc.RESULTS / "match_v3" / "base"
OUT = BASE / "errors"
OCR = cc.RESULTS / "ocr_v1" / "dev" / "paddle_ft1_u2.0" / "orig_alignlong_line"
KEY = "best_wA_0.02"
PER = 6
sim = lambda a, b: sim_v(a, b, *m1.OPT) if a and b else 0.0


def classify(r, o, db):
    t, rd = r["truth"], r["read"]
    top = [x.rsplit(":", 1) for x in r[f"{KEY}_top3"].split()]
    top = [(d, float(v)) for d, v in top]
    me = db[r["drug_N"]]
    tied = [d for d, v in top if top and abs(v - top[0][1]) < 1e-9]
    if r[f"{KEY}_pred"] == "" and r["drug_N"] in tied:
        if any(t in db[d]["imprints"] for d in tied if d != r["drug_N"]):
            return "A"
    if sim(rd, t) < 0.3:
        other = [x for x in me["imprints"] if x != t]
        if any(sim(rd, x) >= 0.7 for x in other):
            return "B1"
        if any(sim(rd, x) >= 0.8 for d in db.values() if d["drug_N"] != r["drug_N"] for x in d["imprints"]):
            return "B2"
    if rd and sorted(rd) == sorted(t):
        return "C"
    if o["fallback"] == "1" or not rd:
        return "E0"
    if t and sum(c in rd for c in t) >= len(t) * 0.5:
        return "D"
    return "E"


def main():
    db = {d["drug_N"]: d for d in m1.load_db()}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(OCR / "pred.csv", encoding="utf-8-sig"))}
    rows = [r for r in csv.DictReader(open(BASE / "pred_dev.csv", encoding="utf-8-sig")) if r[f"{KEY}_ok"] == "0"]
    os.makedirs(OUT, exist_ok=True)
    out = []
    for r in rows:
        o = ocr[r["file_name"]]
        top = [x.rsplit(":", 1) for x in r[f"{KEY}_top3"].split()]
        out.append(dict(
            file_name=r["file_name"], drug_N=r["drug_N"], name=db[r["drug_N"]]["name"], truth=r["truth"], read=r["read"],
            cause=classify(r, o, db), kind="판정없음" if r[f"{KEY}_pred"] == "" else "다른 약",
            pred=r[f"{KEY}_pred"], pred_name=db[r[f"{KEY}_pred"]]["name"] if r[f"{KEY}_pred"] else "",
            in_top3=int(r["drug_N"] in [d for d, _ in top]),
            top3=" | ".join(f"{db[d]['name'][:14]} {float(v):.2f}" for d, v in top),
            other_face="|".join(x for x in db[r["drug_N"]]["imprints"] if x != r["truth"]),
            long=o["aligned"], form=o["form"], camera=o["camera_la"], face=o["face"],
            all_dirs=o["all_dirs"], note=""))
    order = ["A", "B1", "B2", "C", "D", "E0", "E"]
    out.sort(key=lambda x: (order.index(x["cause"]), x["file_name"]))
    with open(OUT / "error_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["no"] + list(out[0]))
        w.writeheader()
        for i, x in enumerate(out, 1):
            w.writerow(dict(no=i, **x))
    # 요약: 원인 × (판정없음/다른 약), 원인 × (긴/그 외)
    c = collections.Counter((x["cause"], x["kind"]) for x in out)
    lg = collections.Counter((x["cause"], x["long"]) for x in out)
    with open(OUT / "error_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cause", "total", "판정없음", "다른 약", "긴 알약", "그 외", "정답 3등 안", "약 종류"])
        for k in order:
            xs = [x for x in out if x["cause"] == k]
            if xs:
                w.writerow([k, len(xs), c[(k, "판정없음")], c[(k, "다른 약")], lg[(k, "1")], lg[(k, "0")],
                            sum(x["in_top3"] for x in xs), len({x["drug_N"] for x in xs})])
    for k in order:
        xs = [x for x in out if x["cause"] == k]
        if xs:
            print(f"  {k:2s} {len(xs):2d}장  약 {len({x['drug_N'] for x in xs})}종  " +
                  ", ".join(sorted({f"{x['drug_N']}({x['truth']})" for x in xs})))
    # 그림: 한 장에 PER 장 (dt 그림 = 고른 방향 상자·읽은 글자 + 원본)
    n_sheet = (len(out) + PER - 1) // PER
    for s in range(n_sheet):
        part = out[s * PER:(s + 1) * PER]
        fig, ax = plt.subplots(PER, 1, figsize=(13, 3.3 * PER))
        for i, a in enumerate(ax):
            a.axis("off")
            if i >= len(part):
                continue
            x = part[i]
            im = cc.imread(str(OCR / "dt" / x["file_name"]))
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            no = s * PER + i + 1
            a.set_title(f"#{no} [{x['cause']}] {x['file_name'][:24]} {x['name'][:18]}  ({x['kind']}, 3등안 {x['in_top3']})\n"
                        f"정답 {x['truth']} (다른 면 {x['other_face'] or '-'}) / 읽음 {x['read'] or '-'}  → {x['pred_name'][:16] or '판정없음'}\n"
                        f"1~3등: {x['top3']}", fontsize=9, loc="left")
        fig.suptitle(f"약 판별 오답 (개발용, base) {s + 1}/{n_sheet} — [자동 분류]", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{s + 1:02d}.png", dpi=80)
        plt.close(fig)
    print(f"저장: {OUT} (error_list.csv {len(out)}장, error_summary.csv, sheet_01~{n_sheet:02d}.png)")


if __name__ == "__main__":
    main()
