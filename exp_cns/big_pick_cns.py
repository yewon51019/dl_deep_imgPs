# =====================================================================
#  big_pick_cns.py  —  '큰 상자 방향 확신도 깎기' P 고르기 (OCR 다시 안 돌리고, 저장된 모든 방향 읽기 + 상자로 흉내)
# =====================================================================
#
#  [왜]  K-003049 90_240 처럼 상자가 알약 전체(넓이 ≥ 400×400 의 절반)를 덮으면 PARSeq 가 'PHOTOGRAPHY'·'CANADA' 같은 단어를
#        높은 확신도로 지어내고, 확신도 최고 방향 규칙이 그걸 고름 (다른 방향엔 상자를 찾은 FCH 92.2 가 있었음)
#  [규칙]  방향마다 점수 = (글자 있음, 확신도 × (1 − P if 큰 상자 있음 else 1)) 최고  (P = 0 이면 지금과 같음)
#  [고르기]  검증(trainval 828장) 읽기 완전일치로 P 고름 (같으면 작은 P), 개발용은 측정만
#  [읽는 것]  ocr_v1\<trainval|dev>\paddle_ft1_u2.0\orig_alignlong_line\pred.csv 의 all_dirs + boxes.json (trainall / dev)
#  [실행]  python exp_cns\big_pick_cns.py   → 고른 P 로  ocr_cns.py --big-pen P  를 다시 돌림
# =====================================================================

import csv
import json

import common_cns as cc

O = cc.RESULTS / "ocr_v1"
PS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
BIG = 0.5 * 400 * 400


def sim(split, boxsplit):
    R = list(csv.DictReader(open(O / split / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv", encoding="utf-8-sig")))
    bx = json.load(open(O / boxsplit / "paddle_ft1_u2.0" / "orig_alignlong" / "boxes.json", encoding="utf-8"))
    out = {}
    for P in PS:
        ok = 0
        for r in R:
            dirs = []
            for s in r["all_dirs"].split(" | "):
                ang, t, c = s.split(":")
                big = any((b[2] - b[0]) * (b[3] - b[1]) >= BIG for b in bx[r["file_name"]][ang]["boxes"])
                # 동점이면 실제로 고른 방향(r['angle']) 우선 — all_dirs 의 확신도는 소수 첫째 자리로 반올림돼 동점이 생김
                dirs.append((bool(t), float(c) * ((1 - P) if big else 1), ang == r["angle"], t))
            ok += max(dirs, key=lambda d: d[:3])[3] == r["truth"]
        out[P] = 100 * ok / len(R)
    return out


def main():
    v = sim("trainval", "trainall")
    d = sim("dev", "dev")
    best = max(PS, key=lambda p: (round(v[p], 6), -p))
    print("검증 읽기:", {f"{p:g}": round(x, 1) for p, x in v.items()})
    print("개발 읽기:", {f"{p:g}": round(x, 1) for p, x in d.items()}, f"→ P = {best:g} (검증으로 고름)")


if __name__ == "__main__":
    main()
