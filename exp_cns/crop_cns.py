# =====================================================================
#  crop_cns.py  —  잘리지 않은 전체 사진에서 알약을 컬러로 잘라 저장 (평가용 exp300test 288장, 사후 분석용)
# =====================================================================
#
#  [왜 필요한가]
#    학습·개발용 사진은 이미 알약에 맞게 잘린 약 230×230 이다. 그런데 exp300test 는 976×1280 전체 사진이고,
#    base 에는 자르는 단계가 없다. 그래서 평가용 글자가 작아졌다
#    (고른 방향에서 가장 높은 상자의 중앙값: 개발용 149 픽셀 / 평가용 42 픽셀).
#
#  [자르는 순서]
#    1. 알약 영역 찾기 (--mask 로 고름)
#       base  = exp300 segment_pill. 사진 테두리 색과 다른 부분을 알약으로 봄 (결과 폴더 crop_v1)
#               약점: 어두운 배경 + 진한 캡슐 절반은 배경으로 빠짐 (심발타 30mg 은 길이의 53% 를 못 찾음)
#       loose = exp300 넉넉 자르기 loose_mask. 기존 영역에 낮은 기준으로 잡은 '붙어 있는' 덩어리를 더함 (결과 폴더 crop_v2)
#               넓이가 2.5배를 넘게 커지면 그림자·배경으로 보고 기존 영역을 그대로 씀. 기준값은 exp300 그대로, 여기서 바꾸지 않음
#    2. 찾은 영역을 감싸는 기울지 않은 네모 + 사방 여유(네모의 긴 변 × 0.15) 로 자름
#       여유 0.15: 원형 알약의 넓이 비율이 약 0.46 이 되어 개발용 중앙값 0.48 과 비슷함
#    3. 자른 그대로 저장. exp300 crop_pill 과 달리 흑백 변환·배경 칠하기·돌리기를 하지 않음
#       (겉모습 CNN 은 색이 필요하고, 학습·개발용 사진에는 배경이 남아 있고, 방향은 다음 단계에서 정함)
#    영역을 못 찾으면 가운데 50% 를 쓰고 crop_info.csv 의 found 를 0 으로 표시 (crop_pill 과 같은 처리)
#
#  [잘림 검사, --mask loose 일 때만]
#    라벨의 실제 길이(mm)로 '알약 길이의 몇 % 를 못 찾았는지' 계산해 crop_compare.csv 에 저장.
#    짧은 변 픽셀 ÷ 라벨 단축 mm 의 카메라별 중앙값 = 1mm 당 픽셀 → 예상 긴 변 = 라벨 장축 mm × 1mm 당 픽셀
#    빠진 비율 = (예상 긴 변 − 찾은 긴 변) ÷ 예상 긴 변.  라벨은 이 검사에만 쓰고 자를 때는 쓰지 않음. OCR 점수도 보지 않음.
#
#  [저장]  --mask base  : results\crop_v1\exp300test\images\, crop_info.csv, results\crop_v1\check_24.png
#          --mask loose : results\crop_v2\exp300test\images\, crop_info.csv, results\crop_v2\check_24.png,
#                         results\crop_v2\crop_compare.csv (기존·넉넉 사진별 비교), results\crop_v2\check_changed.png (자르기 전후 비교)
#          이미 있으면 덮어쓰지 않고 멈춤. 사진이 든 PNG 는 .gitignore 로 올라가지 않음
#  [실행]  python exp_cns\crop_cns.py --mask loose   (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import argparse
import collections
import csv
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_cns as cc
import ocr_common_cns as oc
from run_exp300 import segment_pill

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SPLIT = "exp300test"
MARGIN = 0.15                                                        # 사방 여유 = 찾은 네모의 긴 변 × 0.15
FOLDER = {"base": "crop_v1", "loose": "crop_v2"}                     # 영역 찾는 방법별 결과 폴더
DB_V3 = cc.ROOT / "exp300" / "results" / "search_db_v3.csv"          # 라벨 장축·단축 mm (잘림 검사에만 씀)


def find_mask(bgr, how):
    """알약 영역 찾기 → (마스크 또는 None, 넓이 비율). 넓이 비율 = 넉넉 영역 ÷ 기존 영역 (base 는 항상 1)."""
    if how == "base":
        return segment_pill(bgr), 1.0
    from loose_exp300 import loose_mask                              # 무거운 모듈이라 넉넉 자르기일 때만 불러옴
    return loose_mask(bgr)


def crop_box(bgr, mask):
    """마스크 → 자를 네모 (x0, y0, x1, y1). 마스크가 없으면 가운데 50%. 사진 밖으로는 넘어가지 않음."""
    h, w = bgr.shape[:2]
    if mask is None:
        return w // 4, h // 4, w * 3 // 4, h * 3 // 4
    x, y, bw, bh = cv2.boundingRect(mask)                            # 영역을 감싸는 기울지 않은 네모
    m = int(round(max(bw, bh) * MARGIN))
    return max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)


def sides(mask):
    """마스크 → (긴 변, 짧은 변) 픽셀. 알약에 맞춰 기울어진 네모(minAreaRect)로 재서 알약이 비스듬해도 실제 길이가 나옴."""
    c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    rw, rh = cv2.minAreaRect(c)[1]
    return max(rw, rh), min(rw, rh)


def save_check(items, folder, path):
    """약 24종에서 1장씩(시드 42) 자른 결과를 한 그림으로."""
    by = {}
    for it in items:
        by.setdefault(it["drug_N"], []).append(it)
    rng = random.Random(42)
    fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
    for a, d in zip(ax.flat, rng.sample(sorted(by), 24)):
        it = rng.choice(by[d])
        im = cc.imread(str(folder / "images" / it["file_name"]))
        a.imshow(cv2.cvtColor(cc.fit_side(im, 240), cv2.COLOR_BGR2RGB))
        a.set_title(f"{d} 정답 {it['truth']}\n{im.shape[1]}×{im.shape[0]}, 카메라 {it['camera_la']}°", fontsize=9)
        a.axis("off")
    fig.suptitle(f"자른 사진 24장 확인 (평가용 exp300test, 컬러·배경 유지, 여유 15%, {folder.parent.name})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_changed(cmp_rows, new_dir, path):
    """넓이가 많이 커진 사진 12장(한 약에서 최대 2장)을 기존 | 넉넉 으로 나란히."""
    pick, seen = [], collections.Counter()
    for r in sorted(cmp_rows, key=lambda r: -r["grow"]):
        if seen[r["drug_N"]] < 2 and len(pick) < 12:
            pick.append(r)
            seen[r["drug_N"]] += 1
    fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
    for a in ax.flat:
        a.axis("off")
    for i, r in enumerate(pick):
        src = cc.imread(r["image_path"])                             # 기존 자르기는 원본에서 바로 다시 만듦 (crop_v1 과 같은 계산)
        old = src[r["y0_base"]:r["y1_base"], r["x0_base"]:r["x1_base"]]
        new = cc.imread(str(new_dir / "images" / r["file_name"]))
        a0, a1 = ax.flat[2 * i], ax.flat[2 * i + 1]
        a0.imshow(cv2.cvtColor(cc.fit_side(old, 240), cv2.COLOR_BGR2RGB))
        a0.set_title(f"기존: {r['truth'][:14]}\n빠진 길이 {r['missing_base']:.0f}%", fontsize=9)
        a1.imshow(cv2.cvtColor(cc.fit_side(new, 240), cv2.COLOR_BGR2RGB))
        a1.set_title(f"넉넉: 넓이 {r['grow']:.2f}배\n빠진 길이 {r['missing_loose']:.0f}%", fontsize=9)
    fig.suptitle("자르기 전후 비교 (평가용 exp300test, 왼쪽 기존 | 오른쪽 넉넉, 넓이가 많이 커진 순)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def compare_rows(items):
    """사진마다 기존·넉넉 영역을 라벨 길이와 비교한 줄 목록 (crop_compare.csv)."""
    db = {r["drug_N"]: r for r in csv.DictReader(open(DB_V3, encoding="utf-8-sig"))}
    rows = []
    for it in items:
        src = cc.imread(it["image_path"])
        m0, _ = find_mask(src, "base")
        m1, grow = find_mask(src, "loose")
        l0, s0 = sides(m0)
        l1, s1 = sides(m1)
        x0, y0, x1, y1 = crop_box(src, m0)
        rows.append(dict(file_name=it["file_name"], drug_N=it["drug_N"], truth=it["truth"], camera_la=int(it["camera_la"]),
                         label_long_mm=float(db[it["drug_N"]]["leng_long"]), label_short_mm=float(db[it["drug_N"]]["leng_short"]),
                         long_base=round(l0, 1), short_base=round(s0, 1), long_loose=round(l1, 1), short_loose=round(s1, 1),
                         grow=float(grow), x0_base=x0, y0_base=y0, x1_base=x1, y1_base=y1, image_path=it["image_path"]))
    # 1mm 당 픽셀: 짧은 변으로 구함 (캡슐 절반이 빠져도 짧은 변은 거의 안 줄어듦), 카메라별 중앙값
    scale = {c: float(np.median([r["short_base"] / r["label_short_mm"] for r in rows if r["camera_la"] == c]))
             for c in {r["camera_la"] for r in rows}}
    for r in rows:
        exp = r["label_long_mm"] * scale[r["camera_la"]]             # 예상 긴 변 픽셀
        r["expected_long"] = round(exp, 1)
        r["missing_base"] = round(100 * (exp - r["long_base"]) / exp, 1)     # 기존 영역에서 빠진 길이 %
        r["missing_loose"] = round(100 * (exp - r["long_loose"]) / exp, 1)   # 넉넉 영역에서 빠진 길이 %
    return rows, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="loose", choices=["base", "loose"], help="base = 기존 영역(crop_v1), loose = 넉넉 자르기(crop_v2)")
    a = ap.parse_args()
    out = cc.RESULTS / FOLDER[a.mask]
    out_split = out / SPLIT
    info_csv = out_split / "crop_info.csv"
    assert not info_csv.exists(), f"이미 있음, 덮어쓰지 않음: {info_csv}"
    os.makedirs(out_split / "images", exist_ok=True)
    items = oc.load_split(SPLIT, 0)
    rows = []
    for it in items:
        src = cc.imread(it["image_path"])
        mask, grow = find_mask(src, a.mask)
        x0, y0, x1, y1 = crop_box(src, mask)
        cc.imwrite(str(out_split / "images" / it["file_name"]), src[y0:y1, x0:x1])
        rows.append(dict(file_name=it["file_name"], drug_N=it["drug_N"], camera_la=it["camera_la"], found=int(mask is not None), grow=grow,
                         src_w=src.shape[1], src_h=src.shape[0], x0=x0, y0=y0, x1=x1, y1=y1, crop_w=x1 - x0, crop_h=y1 - y0))
    with open(info_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_check(items, out_split, out / "check_24.png")
    side = [max(r["crop_w"], r["crop_h"]) for r in rows]
    print(f"저장: {out_split}  {len(rows)}장, 못 찾음 {sum(1 - r['found'] for r in rows)}장, "
          f"긴 변 중앙값 {np.median(side):.0f} (10% {np.percentile(side, 10):.0f}, 90% {np.percentile(side, 90):.0f})")
    if a.mask == "loose":
        cmp_rows, scale = compare_rows(items)
        with open(out / "crop_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in cmp_rows[0] if k != "image_path"], extrasaction="ignore")
            w.writeheader()
            w.writerows(cmp_rows)
        save_changed(cmp_rows, out_split, out / "check_changed.png")
        count = lambda k, t: sum(r[k] > t for r in cmp_rows)
        print(f"1mm 당 픽셀 {scale}")
        print(f"빠진 길이 >10% / >20% / >30%: 기존 {count('missing_base', 10)} / {count('missing_base', 20)} / {count('missing_base', 30)}, "
              f"넉넉 {count('missing_loose', 10)} / {count('missing_loose', 20)} / {count('missing_loose', 30)}")
        print(f"넓이 2.5배 초과(기존 영역으로 돌아감) {sum(r['grow'] > 2.5 for r in cmp_rows)}장, 1.3배 초과 {sum(r['grow'] > 1.3 for r in cmp_rows)}장")


if __name__ == "__main__":
    main()
