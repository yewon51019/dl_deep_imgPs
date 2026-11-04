# =====================================================================
#  simple_prep_cns.py  —  간단한 전처리 하나씩 + 이전 전처리(K0L17)를 약하게 한 것을 OCR 점수로 비교 (개발용 468장)
# =====================================================================
#
#  [왜]  지금까지 OCR 로 잰 전처리는 10단계 안팎의 묶음(K0L17 등)뿐이고, 단계별 설정은 사람 눈으로 골랐다.
#        CLAHE 하나, 블러 하나만 넣었을 때의 OCR 점수는 잰 적이 없다.
#  [비교하는 전처리]  모두 긴 변 400 맞춤(원본과 같은 좌표) 뒤에 적용, 색(Lab a·b)은 그대로
#     Sclahe2  밝기 L 에 CLAHE clip 2, tile 8
#     Sclahe4  밝기 L 에 CLAHE clip 4, tile 8
#     Sblur3k  밝기 L 에 가우시안 k3 (σ 는 OpenCV 자동 = 0.8)
#     Sb3cl2x  가우시안 k3 → CLAHE clip 2, tile 8
#     WeakAll  K0L17 의 모든 단계를 약하게 (채널 auto 그대로, 값은 WEAK)
#     WeakLch  WeakAll 에서 채널만 L 로 (K-000065 빨간 DA 가 auto 채널에서 사라졌던 문제)
#     Mix30K0  원본 70% + K0L17 30% 섞기 (단계별 값을 정하지 않고 한꺼번에 약하게)
#  [적용 위치]  읽기만 = 상자는 원본에서 찾은 것 그대로 + 전처리 사진을 읽음 (폴더 <이름>_alignlong_boxorig_line_pft1)
#               찾기+읽기 = 전처리 사진에서 상자도 다시 찾음 (폴더 <이름>_alignlong_line_pft1)
#               검출기 = det_ft1 (unclip 2.0), 읽기 = 파인튜닝 PARSeq (pft1), 축 정렬 + 줄 묶기
#  [채택 규칙, 결과를 보기 전에 정함]  기준(원본 323장)보다 10장 이상 더 맞히고 좋아짐 > 나빠짐 인 설정만 후보.
#     후보는 검증용 828장에서 같은 방향인지 다시 확인. 평가용은 쓰지 않음.
#  [주의]  전처리 결과는 cv2 버전에 따라 달라질 수 있어 --step make 는 .venv 에서 만들어 저장하고, Paddle 환경은 저장된 사진만 읽는다.
#          폴더 이름 앞 7글자가 결과 폴더 이름에 들어가므로(ocr_cns --box-prep) 앞 7글자를 서로 다르게 지었다.
#  [저장]  prep_v2\final\<이름>\params.json, prep_v2\final_dev\<이름>\  (이미 있으면 멈춤)
#          exp_cns\results\simple_prep_v1\check_8.png, compare.csv, changed.csv (이미 있으면 멈춤)
#  [실행]  python exp_cns\simple_prep_cns.py --step make      전처리 설정·사진 + 확인 그림 만들기
#          (paddle_boxes_cns.py · ocr_cns.py 실행 뒤)
#          python exp_cns\simple_prep_cns.py --step compare   표 만들기
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

import common_cns as cc
import det_ft3_compare_cns as dc                                     # 저장된 결과 읽기(load), 큰 상자(has_big), 비율(pct)
import ocr_common_cns as oc
import prep_steps as ps
import read_order_cns as ro
import sweep_cns as sw

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

K0 = "K0L17-c3-Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12_auto_link17_low85"
# 모든 단계를 약하게: 가우시안 k7 s1 → k3 s0.8 / 평균 k5 → 끔 / CLAHE clip 3 → 1.5 / 샤프닝 2 → 0.5 / 마무리 흐림 6 → 2 / 획 잇기 세기 40 → 20
WEAK = {"k": 3, "s": 0.8, "gauss": "k3s0.8", "mean_k": 1, "clip": 1.5, "amount": 0.5, "final_h": 2, "link_strength": 20}
MIX = 0.3                                                            # Mix30K0 에서 K0L17 의 비중
SIMPLE = {"Sclahe2_L_c2_t8": {"blur_k": 0, "clip": 2.0, "tile": 8},
          "Sclahe4_L_c4_t8": {"blur_k": 0, "clip": 4.0, "tile": 8},
          "Sblur3k_L_k3": {"blur_k": 3, "clip": 0, "tile": 0},
          "Sb3cl2x_L_k3_c2_t8": {"blur_k": 3, "clip": 2.0, "tile": 8}}
WEAK_K0 = {"WeakAll_chauto": WEAK, "WeakLch_chL": {**WEAK, "ch": "L"}}
MIX_NAME = "Mix30K0_K0L17x0.3"
NAMES = list(SIMPLE) + list(WEAK_K0) + [MIX_NAME]
OCR = cc.RESULTS / "ocr_v1" / "dev" / "paddle_ft1_u2.0"
OUT = cc.RESULTS / "simple_prep_v1"
BASE_CORRECT = 323                                                   # 원본 + pft1 의 맞은 장수 (orig_alignlong_line_pft1\pred.csv)
NEED = 10                                                            # 후보가 되는 데 필요한 추가 장수


def simple_prep(bgr, p):
    """긴 변 400 맞춤 → 밝기 L 에 (blur_k 가 있으면) 가우시안 → (clip 이 있으면) CLAHE → 원본 색을 입힘."""
    base = cc.fit_side(bgr, ps.SIDE)
    L = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[..., 0]
    if p["blur_k"]:
        L = cv2.GaussianBlur(L, (p["blur_k"], p["blur_k"]), 0)
    if p["clip"]:
        L = ps.hist(L, "clahe", p["clip"], p["tile"])
    return ps.recolor(base, L)


def weak_prep(bgr, p):
    """K0L17 경로를 약한 값으로. apply6 은 획 잇기 세기를 못 받으므로(40 고정) 같은 순서로 직접 부른다."""
    return ps.link_strokes(sw.apply_spot(bgr, p), p["link_len"], p["link_low"], strength=p["link_strength"])


def mix_prep(bgr, k0):
    """원본 70% + 이미 만든 K0L17 사진 30%."""
    return cv2.addWeighted(cc.fit_side(bgr, ps.SIDE), 1 - MIX, k0, MIX, 0)


def make():
    """설정(params.json)과 개발용 468장 전처리 사진을 저장 → oc.Prep 이 저장된 사진을 그대로 읽는다."""
    assert not (OUT / "check_8.png").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'check_8.png'}"
    k0 = json.load(open(oc.FINAL / K0 / "params.json", encoding="utf-8"))
    params = {n: {"kind": "simple", **p} for n, p in SIMPLE.items()}
    params.update({n: {**k0["params"], "kind": "weak", **ch} for n, ch in WEAK_K0.items()})
    params[MIX_NAME] = {"kind": "mix", "from": K0, "mix": MIX}
    for n in NAMES:                                                  # 하나라도 있으면 만들기 전에 멈춤
        assert not (oc.FINAL / n).exists(), f"이미 있음, 덮어쓰지 않음: {oc.FINAL / n}"
    items = oc.load_split("dev", 0)
    for n in NAMES:
        p = params[n]
        os.makedirs(oc.FINAL / n)
        os.makedirs(oc.FINAL_DEV / n)
        with open(oc.FINAL / n / "params.json", "w", encoding="utf-8") as f:
            json.dump({"code": n, "from": K0 if p["kind"] != "simple" else "", "params": p}, f, ensure_ascii=False, indent=2)
        for it in items:
            src = cc.imread(it["image_path"])
            if p["kind"] == "simple":
                out = simple_prep(src, p)
            elif p["kind"] == "weak":
                out = weak_prep(src, p)
            else:
                out = mix_prep(src, cc.imread(str(oc.FINAL_DEV / K0 / it["file_name"])))
            cc.imwrite(str(oc.FINAL_DEV / n / it["file_name"]), out)
        print(f"저장: {oc.FINAL_DEV / n}  ({len(items)}장)")
    save_check(items)


def save_check(items):
    """약 8종에서 1장씩(시드 42): 원본 | K0L17 | 새 전처리 7개."""
    os.makedirs(OUT, exist_ok=True)
    by = {}
    for it in items:
        by.setdefault(it["drug_N"], []).append(it)
    rng = random.Random(42)
    pick = [rng.choice(by[d]) for d in rng.sample(sorted(by), 8)]
    cols = ["원본", "K0L17"] + NAMES
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(2 * len(cols), 2.1 * len(pick)))
    for i, it in enumerate(pick):
        for j, c in enumerate(cols):
            if c == "원본":
                im = cc.fit_side(cc.imread(it["image_path"]), 400)
            else:
                im = cc.imread(str(oc.FINAL_DEV / (K0 if c == "K0L17" else c) / it["file_name"]))
            ax[i, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(c.split("_")[0], fontsize=10)
        ax[i, 0].text(0, -8, f"{it['drug_N']} {it['truth']}", fontsize=8)
    fig.suptitle("간단한 전처리 비교 (개발용 8장)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "check_8.png", dpi=70)
    plt.close(fig)
    print(f"저장: {OUT / 'check_8.png'}")


def compare():
    """조건별 표 + 기준과 맞고 틀림이 달라진 사진 + 채택 규칙 판단."""
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    os.makedirs(OUT, exist_ok=True)
    runs = {"원본 (기준)": "orig_alignlong_line_pft1"}
    for n in NAMES:
        runs[f"{n[:7]} 읽기만"] = f"{n}_alignlong_boxorig_line_pft1"
        runs[f"{n[:7]} 찾기+읽기"] = f"{n}_alignlong_line_pft1"
    data = {k: dc.load(OCR / v) for k, v in runs.items()}
    base = data["원본 (기준)"]
    assert sum(r["correct"] == "1" for r in base.values()) == BASE_CORRECT      # 기존 표와 같은 결과를 읽는지 확인
    table = []
    for name, d in data.items():
        v = list(d.values())
        win = sum(d[f]["correct"] == "1" and base[f]["correct"] != "1" for f in d)
        lose = sum(d[f]["correct"] != "1" and base[f]["correct"] == "1" for f in d)
        n_ok = sum(r["correct"] == "1" for r in v)
        table.append({"조건": name, "맞음": n_ok, "완전일치": dc.pct(v),
                      "순서무시": round(100 * sum(ro.order_free(r) for r in v) / len(v), 1),
                      "좋아짐": win, "나빠짐": lose,
                      "긴알약": dc.pct([r for r in v if r["aligned"] == "1"]),
                      "그외": dc.pct([r for r in v if r["aligned"] != "1"]),
                      "카메라75": dc.pct([r for r in v if r["camera_la"] == "75"]),
                      "카메라90": dc.pct([r for r in v if r["camera_la"] == "90"]),
                      "박스0개": sum(r["fallback"] == "1" for r in v), "큰상자": sum(dc.has_big(r) for r in v),
                      "후보": int(n_ok >= BASE_CORRECT + NEED and win > lose)})
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["condition", "file_name", "drug_N", "truth", "change", "read_orig", "read_new"])
        for name in list(data)[1:]:
            for fn in sorted(base):
                a, b = base[fn]["correct"] == "1", data[name][fn]["correct"] == "1"
                if a != b:
                    w.writerow([name, fn, base[fn]["drug_N"], base[fn]["truth"], "좋아짐" if b else "나빠짐",
                                base[fn]["read"], data[name][fn]["read"]])
    for row in table:
        print(row)
    print(f"후보: {[r['조건'] for r in table if r['후보']] or '없음'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["make", "compare"])
    {"make": make, "compare": compare}[ap.parse_args().step]()
