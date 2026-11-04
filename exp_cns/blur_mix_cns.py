# =====================================================================
#  blur_mix_cns.py  —  원본 + K0L17 섞는 비율 튜닝: 전체에 적용 vs 흐린 사진에만 적용, 약 정확도로 판단 (평가용 안 씀)
# =====================================================================
#
#  [왜]  흐린 사진 시험(simple_prep_v2)에서 Mix30K0 찾기+읽기만 규칙을 통과했지만, 개발용 전체에서는 글자 틀림이 늘었다 (92→96).
#        흐린 사진에만 쓰면 손해를 줄일 수 있는지, 읽기 정답률이 아니라 약 정확도로 본다.
#  [설정]  비율 15·30·50% × 적용 범위 3가지 = 9조합 (전체 / 흐림 점수 ≤ 검증 25% 값 / ≤ 검증 50% 값)
#          흐림 점수 = hard_prep_cns.blur_score (정답 안 봄). 적용 안 하는 사진은 원본 결과를 그대로 씀.
#          전처리 사진으로 상자 찾기 + 읽기, 검출기 det_ft1 (unclip 2.0), pft1, 축 정렬 + 줄 묶기
#          겉모습 CNN 점수 A 는 원본 사진 그대로 (attr_cnn_v1 probs_val·probs_dev)
#  [약 판별]  pft1 의 고정 설정 λ 0.9, w 0.02, 길이 제한 30, 반대 방향 부분일치 끔 (match3 에서 고른 값, 기준 검증 78.3 / 개발 87.4 재현 확인)
#  [규칙, 결과를 보기 전에 정함]
#     고르기 = 검증 828장 약 정확도 최고 (같으면 비율 낮은 쪽, 적용 범위 좁은 쪽)
#     채택 = 개발 468장 약 정확도가 기준 이상 + 검증·개발 합쳐 NF(K-014467) 빼고 좋아짐 > 나빠짐.
#     개발용에서 +5장(약 1%p) 미만이면 우연 범위로 봄
#  [저장]  prep_v2\final\<이름>\params.json, prep_v2\final_valdev\<이름>\
#          results\simple_prep_v3\blur_scores.csv, pred_<source>.csv, compare.csv, changed.csv (이미 있으면 멈춤)
#  [실행]  python exp_cns\blur_mix_cns.py --step make      전처리 사진 저장 (.venv)
#          (paddle_boxes_cns.py · ocr_cns.py --split valdev 실행 뒤)
#          python exp_cns\blur_mix_cns.py --step compare   약 판별 + 표
# =====================================================================

import argparse
import csv
import json
import os

import cv2
import numpy as np

import common_cns as cc
import hard_prep_cns as hp
import match_cns as m1
import match2_cns as m2
import ocr_common_cns as oc
import prep_steps as ps
import sweep_cns as sw

OUT = cc.RESULTS / "simple_prep_v3"
K0 = hp.K0
RATIOS = {0.15: "Mix15K0_K0L17x0.15", 0.3: "Mix30K0_K0L17x0.3", 0.5: "Mix50K0_K0L17x0.5"}
SCOPES = {"전체": None, "흐림25": 25, "흐림50": 50}                     # None = 모든 사진, 숫자 = 검증 흐림 점수의 그 백분위 값 이하만
LAM, W = 0.9, 0.02
BASE_ACC = {"val": 78.3, "dev": 87.4}                                # pft1 기준 (match_v3 compare.csv)
OCR = oc.OCR_OUT
NF = "K-014467"


def make():
    """K0L17 을 먼저 만들고, 비율별로 원본과 섞어 저장."""
    items = oc.load_split("valdev", 0)
    assert not (oc.FINAL_VALDEV / K0).exists(), f"이미 있음, 덮어쓰지 않음: {oc.FINAL_VALDEV / K0}"
    os.makedirs(oc.FINAL_VALDEV / K0)
    p = json.load(open(oc.FINAL / K0 / "params.json", encoding="utf-8"))["params"]
    for it in items:
        old = oc.FINAL_DEV / K0 / it["file_name"]                    # 개발용은 이미 만든 사진을 그대로
        out = cc.imread(str(old)) if old.exists() else sw.apply6(cc.imread(it["image_path"]), p)
        cc.imwrite(str(oc.FINAL_VALDEV / K0 / it["file_name"]), out)
    print(f"저장: {oc.FINAL_VALDEV / K0}  ({len(items)}장)")
    for r, name in RATIOS.items():
        os.makedirs(oc.FINAL / name, exist_ok=True)
        f = oc.FINAL / name / "params.json"
        if not f.exists():
            with open(f, "w", encoding="utf-8") as fh:
                json.dump({"code": name, "from": K0, "params": {"kind": "mix", "from": K0, "mix": r}}, fh, ensure_ascii=False, indent=2)
        os.makedirs(oc.FINAL_VALDEV / name)
        for it in items:
            base = cc.fit_side(cc.imread(it["image_path"]), ps.SIDE)
            k0 = cc.imread(str(oc.FINAL_VALDEV / K0 / it["file_name"]))
            cc.imwrite(str(oc.FINAL_VALDEV / name / it["file_name"]), cv2.addWeighted(base, 1 - r, k0, r, 0))
        print(f"저장: {oc.FINAL_VALDEV / name}  ({len(items)}장)")


def predict(rows, probs, db):
    """pred.csv 줄들 → {파일: (1등 약 또는 '', 읽은 글자)}. 설정은 맨 위 고정값."""
    out = {}
    for r in rows:
        fn = r["file_name"]
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        cm, co = m2.split_candidates(r, m2.MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM)
        out[fn] = m1.decide({d: T.get(d, 0.0) + W * A[d] for d in A})[0]
    return out


def compare():
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    os.makedirs(OUT, exist_ok=True)
    db = m1.load_db()
    probs = {}
    part_of = {}
    for part in ("val", "dev"):
        for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig")):
            probs[r["file_name"]] = {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
            part_of[r["file_name"]] = part
    items = oc.load_split("valdev", 0)
    assert {it["file_name"] for it in items} == set(probs)
    score = {it["file_name"]: hp.blur_score(it["image_path"]) for it in items}
    with open(OUT / "blur_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "part", "blur"])
        w.writerows((fn, part_of[fn], round(s, 2)) for fn, s in sorted(score.items()))
    val_scores = [s for fn, s in score.items() if part_of[fn] == "val"]
    cuts = {k: (None if v is None else float(np.percentile(val_scores, v))) for k, v in SCOPES.items()}
    print("흐림 기준:", cuts)
    # 원본 결과: 검증·개발 기존 pred.csv
    base_rows = []
    for split in ("trainval", "dev"):
        base_rows += list(csv.DictReader(open(OCR / split / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv", encoding="utf-8-sig")))
    info = {r["file_name"]: r for r in base_rows}
    sources = {"orig": base_rows}
    for r, name in RATIOS.items():
        sources[r] = list(csv.DictReader(open(OCR / "valdev" / "paddle_ft1_u2.0" / f"{name}_alignlong_line_pft1" / "pred.csv", encoding="utf-8-sig")))
    preds = {}
    for k, rows in sources.items():
        preds[k] = predict(rows, probs, db)
        with open(OUT / f"pred_{k}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "pred"])
            w.writerows((r["file_name"], r["drug_N"], r["truth"], r["read"], preds[k][r["file_name"]]) for r in rows)
        print(f"약 판별 완료: {k}")
    ok = lambda pr, fn: pr[fn] == info[fn]["drug_N"]
    files = {p: [fn for fn in probs if part_of[fn] == p] for p in ("val", "dev")}
    acc = lambda pr, p: round(100 * sum(ok(pr, fn) for fn in files[p]) / len(files[p]), 1)
    base = preds["orig"]
    print(f"기준 재현: 검증 {acc(base, 'val')} (기록 {BASE_ACC['val']}) / 개발 {acc(base, 'dev')} (기록 {BASE_ACC['dev']})")
    table = [dict(조합="원본 (기준)", 비율=0, 범위="", 적용장수=0, 검증정확도=acc(base, "val"), 개발정확도=acc(base, "dev"),
                  검증판정없음=sum(base[fn] == "" for fn in files["val"]), 개발판정없음=sum(base[fn] == "" for fn in files["dev"]),
                  검증좋아짐=0, 검증나빠짐=0, 개발좋아짐=0, 개발나빠짐=0, NF뺀좋아짐=0, NF뺀나빠짐=0)]
    changed = []
    for r, name in RATIOS.items():
        for scope, cut in cuts.items():
            use = {fn for fn in probs if cut is None or score[fn] <= cut}
            pr = {fn: preds[r][fn] if fn in use else base[fn] for fn in probs}
            win = lambda fs: [fn for fn in fs if ok(pr, fn) and not ok(base, fn)]
            lose = lambda fs: [fn for fn in fs if ok(base, fn) and not ok(pr, fn)]
            combo = f"{int(r * 100)}% {scope}"
            all_f = list(probs)
            table.append(dict(조합=combo, 비율=r, 범위=scope, 적용장수=len(use), 검증정확도=acc(pr, "val"), 개발정확도=acc(pr, "dev"),
                              검증판정없음=sum(pr[fn] == "" for fn in files["val"]), 개발판정없음=sum(pr[fn] == "" for fn in files["dev"]),
                              검증좋아짐=len(win(files["val"])), 검증나빠짐=len(lose(files["val"])),
                              개발좋아짐=len(win(files["dev"])), 개발나빠짐=len(lose(files["dev"])),
                              NF뺀좋아짐=sum(info[fn]["drug_N"] != NF for fn in win(all_f)), NF뺀나빠짐=sum(info[fn]["drug_N"] != NF for fn in lose(all_f))))
            for fn, tag in [(f, "좋아짐") for f in win(all_f)] + [(f, "나빠짐") for f in lose(all_f)]:
                changed.append([combo, part_of[fn], fn, info[fn]["drug_N"], info[fn]["truth"], tag, info[fn]["read"], next(x for x in sources[r] if x["file_name"] == fn)["read"], base[fn], pr[fn]])
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["combo", "part", "file_name", "drug_N", "truth", "change", "read_orig", "read_mix", "pred_orig", "pred_new"])
        w.writerows(changed)
    for row in table:
        print(row)
    order = {"흐림25": 0, "흐림50": 1, "전체": 2}
    best = max(table[1:], key=lambda x: (x["검증정확도"], -x["비율"], -order[x["범위"]]))
    pass_dev = best["개발정확도"] >= table[0]["개발정확도"]
    pass_nf = best["NF뺀좋아짐"] > best["NF뺀나빠짐"]
    print(f"고른 조합(검증 최고): {best['조합']} → 개발 기준 이상 {pass_dev}, NF 빼고 좋아짐>나빠짐 {pass_nf} → {'채택' if pass_dev and pass_nf else '채택 안 함'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["make", "compare"])
    {"make": make, "compare": compare}[ap.parse_args().step]()
