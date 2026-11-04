# =====================================================================
#  prep_variants_cns.py  —  K0L17 의 채널을 밝기(L)로 바꾼 전처리 비교 (재학습 전에 효과가 있는지 먼저 봄, 개발용 468장)
# =====================================================================
#
#  [왜]  K0L17 은 사진마다 대비가 가장 큰 채널을 고른다 (학습 사진 254장 중 191장이 초록).
#        K-000065 의 빨간 반쪽 'DA' 가 초록 채널을 고르는 첫 단계에서 거의 사라지고, 블러를 줄여도 안 살아나지만
#        밝기(L) 채널로는 살아남을 그림으로 확인함. 사진 1장 확인이라 468장에서 효과를 잰다.
#  [비교하는 전처리]  나머지 설정은 모두 K0L17 그대로
#     K0L17Lc  채널 auto → L
#     K0L17LB  채널 auto → L + 블러 줄임 (가우시안 k7 s1 → k3 s0.8, 평균 필터 k5 → 끔, 마무리 흐림 6 → 0)
#  [비교 방법]  검출기 = 원본으로 학습한 det_ft1 (unclip 2.0) 그대로, 전처리 사진에서 박스 찾기, 읽기는 원본 사진 + 사전학습 PARSeq
#  [재학습 규칙, 결과를 보기 전에 정함]  K0L17 박스(308장 맞음, 65.8%)보다 5장 이상 더 맞힌 설정이 있으면
#     그중 가장 많이 맞힌 설정으로 det_ft3 재학습. 없으면 재학습 안 함. 평가용은 쓰지 않음.
#  [주의]  전처리 결과는 cv2 버전에 따라 달라질 수 있어 --step make 는 .venv 에서 만들어 저장하고, Paddle 환경은 저장된 사진만 읽는다.
#          폴더 이름 앞 7글자가 결과 폴더 이름에 들어가므로(ocr_cns --box-prep) 앞 7글자를 서로 다르게 지었다.
#  [저장]  prep_v2\final\<이름>\params.json, prep_v2\final_dev\<이름>\  (이미 있으면 멈춤)
#          exp_cns\results\prep_variants\compare.csv, changed.csv (이미 있으면 멈춤)
#  [실행]  python exp_cns\prep_variants_cns.py --step make      전처리 설정·사진 만들기
#          (paddle_boxes_cns.py · ocr_cns.py 실행 뒤)
#          python exp_cns\prep_variants_cns.py --step compare   표 만들기
# =====================================================================

import argparse
import csv
import json
import os

import common_cns as cc
import det_ft3_compare_cns as dc                                     # 저장된 결과 읽기(load), 큰 상자(has_big), 비율(pct)
import ocr_common_cns as oc
import read_order_cns as ro

K0 = "K0L17-c3-Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12_auto_link17_low85"
LESS_BLUR = {"k": 3, "s": 0.8, "gauss": "k3s0.8", "mean_k": 1, "final_h": 0}
VARIANTS = {"K0L17Lc_chL_link17_low85": {"ch": "L"},
            "K0L17LB_chL_lessblur_link17_low85": {"ch": "L", **LESS_BLUR}}
OCR = cc.RESULTS / "ocr_v1" / "dev" / "paddle_ft1_u2.0"
OUT = cc.RESULTS / "prep_variants"
K0_CORRECT = 308                                                     # K0L17 박스 + 원본 읽기의 맞은 장수 (read_order_compare.csv)
NEED = 5                                                             # 재학습에 필요한 추가 장수


def make():
    """params.json 을 K0L17 에서 복사해 값만 바꾸고, 개발용 468장을 oc.Prep 으로 만들어 저장."""
    base = json.load(open(oc.FINAL / K0 / "params.json", encoding="utf-8"))
    items = oc.load_split("dev", 0)
    for name, change in VARIANTS.items():
        d = oc.FINAL / name
        assert not d.exists(), f"이미 있음, 덮어쓰지 않음: {d}"
        os.makedirs(d)
        cfg = {**base, "code": name, "from": K0, "params": {**base["params"], **change}}
        cfg.pop("n_images", None)                                    # 이 폴더에는 평가용 사진을 만들지 않음
        with open(d / "params.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.makedirs(oc.FINAL_DEV / name, exist_ok=True)
        prep = oc.Prep(name, "dev")
        for it in items:
            prep(it, cc.imread(it["image_path"]))                    # 저장 폴더에 사진이 쌓임
        print(f"저장: {oc.FINAL_DEV / name}  ({len(items)}장)")


def compare():
    """조건별 표 + 원본 박스와 맞고 틀림이 달라진 사진 + 재학습 규칙 판단."""
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    os.makedirs(OUT, exist_ok=True)
    runs = {"원본 박스 (기준)": "orig_alignlong_line", "K0L17 박스": "orig_alignlong_boxK0L17-c_line"}
    runs.update({f"{n[:7]} 박스": f"orig_alignlong_box{n[:7]}_line" for n in VARIANTS})
    data = {k: dc.load(OCR / v) for k, v in runs.items()}
    base, k0 = data["원본 박스 (기준)"], data["K0L17 박스"]
    assert sum(r["correct"] == "1" for r in k0.values()) == K0_CORRECT    # 기존 표와 같은 결과를 읽는지 확인
    table = []
    for name, d in data.items():
        v = list(d.values())
        win = lambda ref: sum(d[f]["correct"] == "1" and ref[f]["correct"] != "1" for f in d)
        lose = lambda ref: sum(d[f]["correct"] != "1" and ref[f]["correct"] == "1" for f in d)
        table.append({"조건": name, "맞음": sum(r["correct"] == "1" for r in v), "완전일치": dc.pct(v),
                      "순서무시": round(100 * sum(ro.order_free(r) for r in v) / len(v), 1),
                      "원본 대비 좋아짐": win(base), "원본 대비 나빠짐": lose(base),
                      "K0L17 대비 좋아짐": win(k0), "K0L17 대비 나빠짐": lose(k0),
                      "긴알약": dc.pct([r for r in v if r["aligned"] == "1"]),
                      "그외": dc.pct([r for r in v if r["aligned"] != "1"]),
                      "카메라75": dc.pct([r for r in v if r["camera_la"] == "75"]),
                      "카메라90": dc.pct([r for r in v if r["camera_la"] == "90"]),
                      "박스0개": sum(r["fallback"] == "1" for r in v), "큰상자": sum(dc.has_big(r) for r in v)})
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["condition", "file_name", "drug_N", "truth", "change_vs_K0L17", "read_K0L17", "read_new"])
        for name in list(data)[2:]:
            for fn in sorted(data[name]):
                a, b = k0[fn]["correct"] == "1", data[name][fn]["correct"] == "1"
                if a != b:
                    w.writerow([name, fn, k0[fn]["drug_N"], k0[fn]["truth"], "좋아짐" if b else "나빠짐",
                                k0[fn]["read"], data[name][fn]["read"]])
    for row in table:
        print(row)
    new = table[2:]
    best = max(new, key=lambda r: r["맞음"])
    ok = best["맞음"] >= K0_CORRECT + NEED
    print(f"가장 좋은 설정 {best['조건']} {best['맞음']}장 (K0L17 {K0_CORRECT}장) → {'재학습 조건 충족' if ok else '재학습 안 함'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["make", "compare"])
    {"make": make, "compare": compare}[ap.parse_args().step]()
