# =====================================================================
#  det_ft_compare_cns.py  —  기본 Paddle 검출기 vs 파인튜닝 검출기 상자 비교 그림 (CLAUDE.md 10장 2-5 ④)
# =====================================================================
#
#  [하는 일]  개발용에서 '파인튜닝으로 좋아진 사진'과 '나빠진 사진'을 시드 42 로 무작위 n 장씩 골라
#             한 줄 = [기본 검출기: 고른 방향 사진 + 상자 + 읽은 글자 | 파인튜닝: 같은 방식] 으로 그림
#             (좋은 예만 고르지 않게 무작위. 각자 '확신도 최고 방향'이 다를 수 있어 방향도 따로)
#  [읽는 것]  ocr_v1\dev\<det>\orig_alignlong_line\pred.csv (고른 방향·읽은 글자) + <det>\orig_alignlong\boxes.json (그 방향 상자)
#  [저장]    exp_cns\results\det_ft1\compare_boxes.png  (원본 사진이 들어가므로 .gitignore 로 안 올림)
#  [실행]    python exp_cns\det_ft_compare_cns.py --ft paddle_ft1_u2.0
#            python exp_cns\det_ft_compare_cns.py --base paddle_ft1_u2.0 --ft paddle_ft2e13_u2.0 --out det_ft2\compare_ft1_ep13.png
#            (--out 을 주면 좋아짐·나빠짐 전체 목록 CSV 도 같은 이름으로 저장)
# =====================================================================

import argparse
import csv
import json
import random

import cv2
import numpy as np

import common_cns as cc
import ocr_common_cns as oc


def load(det):
    d = oc.OCR_OUT / "dev" / det
    pred = {r["file_name"]: r for r in csv.DictReader(open(d / "orig_alignlong_line" / "pred.csv", encoding="utf-8-sig"))}
    boxes = json.load(open(d / "orig_alignlong" / "boxes.json", encoding="utf-8"))
    return pred, boxes


def panel(img, r, boxes, title, tile=300):
    """고른 방향으로 돌린 사진 + 그 방향 상자(초록) + 위 글씨(검출기, 읽은 글자, 맞음 여부)."""
    ang = r["angle"]
    rot = oc.rotate_color(img, float(ang))
    for x0, y0, x1, y1 in boxes[r["file_name"]][ang]["boxes"]:
        cv2.rectangle(rot, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 0), 2)
    s = tile / max(rot.shape[:2])
    rot = cv2.resize(rot, (round(rot.shape[1] * s), round(rot.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cell = np.full((tile, tile, 3), 255, np.uint8)
    cell[:rot.shape[0], :rot.shape[1]] = rot
    bar = np.full((44, tile, 3), 255, np.uint8)
    ok = r["correct"] == "1"
    cv2.putText(bar, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(bar, f"read: {r['read'] or '-'}  {'O' if ok else 'X'}  ({ang} deg)", (6, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0) if ok else (0, 0, 220), 1)
    return np.vstack([bar, cell])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", default="paddle_ft1_u2.0", help="비교할 파인튜닝 검출기 이름")
    ap.add_argument("--n", type=int, default=4, help="좋아진·나빠진 사진 각각 몇 장")
    ap.add_argument("--base", default="paddle_t0", help="비교 기준 검출기 이름")
    ap.add_argument("--out", default=None, help="results 아래 저장 경로 (없으면 det_ft1/compare_boxes.png)")
    a = ap.parse_args()
    bp, bb = load(a.base)
    fp, fb = load(a.ft)
    names = sorted(fp)
    up = [k for k in names if fp[k]["correct"] == "1" and bp[k]["correct"] == "0"]
    dn = [k for k in names if fp[k]["correct"] == "0" and bp[k]["correct"] == "1"]
    rng = random.Random(42)
    pick = [("better", k) for k in rng.sample(up, min(a.n, len(up)))] + [("worse", k) for k in rng.sample(dn, min(a.n, len(dn)))]
    items = {it["file_name"]: it for it in oc.load_split("dev", 0)}
    prep = oc.Prep("orig", "dev")
    rows = []
    for tag, k in pick:
        img, _ = prep(items[k], cc.imread(items[k]["image_path"]))
        head = np.full((26, 600 + 12, 3), 255, np.uint8)
        cv2.putText(head, f"[{tag}] {k}  truth: {bp[k]['truth']}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        pair = [np.pad(panel(img, p[k], b, n), ((3, 3), (3, 3), (0, 0)), constant_values=128)
                for p, b, n in ((bp, bb, a.base), (fp, fb, a.ft))]
        rows.append(np.vstack([head, np.hstack(pair)]))
    cols = [np.vstack(rows[i::2]) for i in range(2)] if len(rows) > 1 else rows   # 두 열로 (그림이 너무 길지 않게)
    h = max(c.shape[0] for c in cols)
    cols = [np.pad(c, ((0, h - c.shape[0]), (0, 10), (0, 0)), constant_values=255) for c in cols]
    title = np.full((36, sum(c.shape[1] for c in cols), 3), 255, np.uint8)
    cv2.putText(title, f"{a.base} vs {a.ft} boxes (dev, random seed 42: {a.n} better + {a.n} worse, of +{len(up)} / -{len(dn)})",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    out = cc.RESULTS / (a.out or "det_ft1/compare_boxes.png")
    if a.out:                                                        # 무작위로 뽑지 않은 전체 목록
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["change", "file_name", "truth", "base_read", "base_angle", "ft_read", "ft_angle"])
            for tag, ks in (("better", up), ("worse", dn)):
                for k in ks:
                    w.writerow([tag, k, bp[k]["truth"], bp[k]["read"], bp[k]["angle"], fp[k]["read"], fp[k]["angle"]])
    cc.imwrite(str(out), np.vstack([title, np.hstack(cols)]))
    print(f"저장: {out}  (좋아짐 {len(up)} / 나빠짐 {len(dn)} 중 각 {a.n}장)")


if __name__ == "__main__":
    main()
