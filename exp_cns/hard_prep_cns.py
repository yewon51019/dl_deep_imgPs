# =====================================================================
#  hard_prep_cns.py  —  흐린 사진에서 전처리가 글자를 더 읽게 하는지 시험 (평가용 237장은 쓰지 않음)
# =====================================================================
#
#  [왜]  simple_prep_v1 의 468장 전체 숫자에는 흐린 사진의 효과가 묻히고, 순서 변화도 섞인다.
#  [사진 고르기]  '못 읽은 사진'만 고르면 나빠질 사진이 빠져 결과가 좋게만 나온다 (DBNet++ 86장 +9/−1 → 468장 +15/−79).
#     그래서 정답을 보지 않는 흐림 점수(400 맞춤 사진 가운데 50% 밝기의 라플라시안 분산, 낮을수록 흐림)로 고른다.
#     val_blur = 검증 828장의 낮은 쪽 25% (판단 기준)
#     dev_blur = 개발 468장의 낮은 쪽 25%, 라벨 문제 K-035766·K-000147 제외 (같은 방향인지 확인)
#     exp51    = exp300 개발용 51장 (넉넉 자르기 crop_v2) (참고)
#     ref:*    = 사용자가 지정한 약 (참고, 약별로 따로 보고). 학습에 쓰인 약이 섞여 있어 판단에는 쓰지 않음
#  [비교]  orig, Sclahe2, Sclahe4, Mix30K0, K0L17, K0L17R6 × 읽기만 / 찾기+읽기. 검출기 det_ft1 (unclip 2.0), pft1, 축 정렬 + 줄 묶기
#  [세 칸]  exact = 완전일치, order = 글자 구성이 정답과 같거나 상자 순서를 바꾸면 정답, wrong = 글자가 틀림
#  [채택 규칙, 결과를 보기 전에 정함]  val_blur 에서 (글자 고침 − 글자 망침) ≥ +5 이고 dev_blur 에서 ≥ 0 인 설정만 튜닝 후보
#     고침 = 원본 wrong → exact·order, 망침 = 원본 exact·order → wrong
#  [저장]  results\simple_prep_v2\groups.csv, check_blur24.png, compare.csv, changed.csv, prep_v2\final_hardset\<이름>\ (이미 있으면 멈춤)
#  [실행]  python exp_cns\hard_prep_cns.py --step select     사진 고르기 + 확인 그림
#          python exp_cns\hard_prep_cns.py --step make       전처리 사진 저장 (.venv 에서, Paddle 환경은 저장된 사진만 읽음)
#          (paddle_boxes_cns.py · ocr_cns.py --split hardset 실행 뒤)
#          python exp_cns\hard_prep_cns.py --step compare    표 만들기
# =====================================================================

import argparse
import csv
import json
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_cns as cc
import ocr_common_cns as oc
import read_order_cns as ro
import simple_prep_cns as sp
import sweep_cns as sw

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = cc.RESULTS / "simple_prep_v2"
OCR = cc.RESULTS / "ocr_v1" / "hardset" / "paddle_ft1_u2.0"
K0 = sp.K0
K6 = "K0L17R6-q2-Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12_auto_link17_low85_rim0.6"
SCL2, SCL4, MIX = "Sclahe2_L_c2_t8", "Sclahe4_L_c4_t8", sp.MIX_NAME
PREPS = [SCL2, SCL4, MIX, K0, K6]
BAD_LABEL = {"K-035766", "K-000147"}                                 # 개발용 라벨 문제 약 (사진·라벨 불일치, 앞뒤 뒤바뀜)
REF_DRUGS = {"K-000746", "K-016779", "K-017237", "K-027741", "K-014467", "K-000650"}   # 사용자가 지정한 약 (사진 파일로 지정한 4종 포함)
REF_TRUTH = {"F", "LIT", "E10", "ARICEPT", "SAMJINOSTEMIN"}          # 사용자가 지정한 각인 (아리셉트정은 사진이 없어 아리셉트에비스정)
NEED = 5                                                             # val_blur 에서 필요한 (고침 − 망침)
GROUPS_MAIN = ["val_blur", "dev_blur", "exp51"]


def blur_score(path):
    """흐림 점수: 400 맞춤 사진 가운데 50% 밝기의 라플라시안 분산 (낮을수록 흐림). 정답은 보지 않음."""
    L = cv2.cvtColor(cc.fit_side(cc.imread(path), 400), cv2.COLOR_BGR2LAB)[..., 0][100:300, 100:300]
    return float(cv2.Laplacian(L, cv2.CV_64F).var())


def select():
    """묶음을 정해 groups.csv 로 저장. 사진 하나가 여러 묶음에 들어갈 수 있어 (사진, 묶음) 한 줄씩."""
    assert not oc.HARDSET_CSV.exists(), f"이미 있음, 덮어쓰지 않음: {oc.HARDSET_CSV}"
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split, group in (("trainval", "val_blur"), ("dev", "dev_blur")):
        items = [it for it in oc.load_split(split, 0) if group != "dev_blur" or it["drug_N"] not in BAD_LABEL]
        sc = {it["file_name"]: blur_score(it["image_path"]) for it in items}
        cut = np.percentile(list(sc.values()), 25)
        rows += [dict(file_name=it["file_name"], group=group, drug_N=it["drug_N"], truth=it["truth"],
                      blur=round(sc[it["file_name"]], 2), image_path=it["image_path"]) for it in items if sc[it["file_name"]] <= cut]
        print(f"{group}: {len(items)}장 중 낮은 쪽 25% 기준 {cut:.2f} → {sum(r['group'] == group for r in rows)}장")
    for it in oc.load_split("exp300dev51_crop2", 0):
        rows.append(dict(file_name=it["file_name"], group="exp51", drug_N=it["drug_N"], truth=it["truth"],
                         blur=round(blur_score(it["image_path"]), 2), image_path=it["image_path"]))
    for split in ("dev", "trainall", "exp300dev51_crop2"):
        for it in oc.load_split(split, 0):
            if it["drug_N"] in REF_DRUGS or it["truth"] in REF_TRUTH:
                rows.append(dict(file_name=it["file_name"], group=f"ref:{it['truth']}", drug_N=it["drug_N"], truth=it["truth"],
                                 blur=round(blur_score(it["image_path"]), 2), image_path=it["image_path"]))
    with open(oc.HARDSET_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {oc.HARDSET_CSV}  (줄 {len(rows)}, 사진 {len({r['file_name'] for r in rows})}장)")
    for g in sorted({r["group"] for r in rows}):
        v = [r for r in rows if r["group"] == g]
        print(f"  {g}: {len(v)}장, 약 {len({r['drug_N'] for r in v})}종, 점수 중앙값 {np.median([r['blur'] for r in v]):.1f}")
    pick = random.Random(42).sample([r for r in rows if r["group"] == "val_blur"], 24)
    fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
    for a, r in zip(ax.flat, sorted(pick, key=lambda r: r["blur"])):
        a.imshow(cv2.cvtColor(cc.fit_side(cc.imread(r["image_path"]), 240), cv2.COLOR_BGR2RGB))
        a.set_title(f"{r['truth']}\n점수 {r['blur']:.1f}", fontsize=9)
        a.axis("off")
    fig.suptitle("흐림 점수가 낮은 사진 24장 (검증용 낮은 쪽 25% 에서 무작위, 점수 낮은 순)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "check_blur24.png", dpi=80)
    plt.close(fig)


def make():
    """hardset 사진의 전처리 결과를 만들어 저장. K0L17 을 먼저 만들어야 Mix30K0 을 만들 수 있음."""
    items = oc.load_split("hardset", 0)
    for name in (K0, K6, SCL2, SCL4, MIX):
        d = oc.FINAL_HARD / name
        assert not d.exists(), f"이미 있음, 덮어쓰지 않음: {d}"
        os.makedirs(d)
        p = json.load(open(oc.FINAL / name / "params.json", encoding="utf-8"))["params"]
        for it in items:
            src = cc.imread(it["image_path"])
            if name in (K0, K6):
                out = sw.apply6(src, p)
            elif name == MIX:
                out = sp.mix_prep(src, cc.imread(str(oc.FINAL_HARD / K0 / it["file_name"])))
            else:
                out = sp.simple_prep(src, p)
            cc.imwrite(str(d / it["file_name"]), out)
        print(f"저장: {d}  ({len(items)}장)")


def state(r):
    """exact / order / wrong (맨 위 설명)."""
    if r["correct"] == "1":
        return "exact"
    return "order" if sorted(r["read"]) == sorted(r["truth"]) or ro.order_free(r) else "wrong"


def load(folder):
    return {r["file_name"]: r for r in csv.DictReader(open(OCR / folder / "pred.csv", encoding="utf-8-sig"))}


def compare():
    """묶음별·조건별 표 + 맞고 틀림이 달라진 사진 + 채택 규칙 판단."""
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    groups = {}
    for r in csv.DictReader(open(oc.HARDSET_CSV, encoding="utf-8-sig")):
        groups.setdefault(r["group"], []).append(r["file_name"])
    base = load("orig_alignlong_line_pft1")
    runs = {"원본 (기준)": base}
    for n in PREPS:
        runs[f"{n[:7]} 읽기만"] = load(f"{n}_alignlong_boxorig_line_pft1")
        runs[f"{n[:7]} 찾기+읽기"] = load(f"{n}_alignlong_line_pft1")
    # 기준이 기존 결과와 같은지 확인: dev·검증 사진의 읽은 글자 비교 (상자를 다시 계산했으므로 달라지면 보고)
    for split, g in (("dev", "dev_blur"), ("trainval", "val_blur")):
        old = {r["file_name"]: r for r in csv.DictReader(open(cc.RESULTS / "ocr_v1" / split / "paddle_ft1_u2.0" / "orig_alignlong_line_pft1" / "pred.csv", encoding="utf-8-sig"))}
        print(f"기준 재현 {g}: {sum(old[f]['read'] == base[f]['read'] for f in groups[g])}/{len(groups[g])}장 같음")
    table = []
    for g, fs in groups.items():
        for name, d in runs.items():
            st = {f: state(d[f]) for f in fs}
            b = {f: state(base[f]) for f in fs}
            table.append({"묶음": g, "조건": name, "장수": len(fs),
                          "완전일치": sum(v == "exact" for v in st.values()),
                          "순서만": sum(v == "order" for v in st.values()),
                          "글자틀림": sum(v == "wrong" for v in st.values()),
                          "고침": sum(b[f] == "wrong" and st[f] != "wrong" for f in fs),
                          "망침": sum(b[f] != "wrong" and st[f] == "wrong" for f in fs),
                          "큰상자": sum(any((x[2] - x[0]) * (x[3] - x[1]) >= 80000 for x in json.loads(d[f]["best_boxes"] or "[]")) for f in fs),
                          "박스0개": sum(d[f]["fallback"] == "1" for f in fs)})
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["condition", "file_name", "drug_N", "truth", "state_orig", "state_new", "read_orig", "read_new"])
        for name, d in list(runs.items())[1:]:
            for fn in sorted(base):
                if state(base[fn]) != state(d[fn]):
                    w.writerow([name, fn, base[fn]["drug_N"], base[fn]["truth"], state(base[fn]), state(d[fn]), base[fn]["read"], d[fn]["read"]])
    net = lambda g, c: next(r["고침"] - r["망침"] for r in table if r["묶음"] == g and r["조건"] == c)
    for r in table:
        print(r)
    picks = [c for c in list(runs)[1:] if net("val_blur", c) >= NEED and net("dev_blur", c) >= 0]
    print(f"튜닝 후보: {picks or '없음'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["select", "make", "compare"])
    {"select": select, "make": make, "compare": compare}[ap.parse_args().step]()
