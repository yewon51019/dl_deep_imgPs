# =====================================================================
#  viz_boxes_exp300.py  —  글자 검출 박스 그림 (EasyOCR CRAFT / PaddleOCR 기본 / PaddleOCR 검출 0.0)
# =====================================================================
#
#  [하는 일]
#   300장 중 30장을 뽑아서(알약 회전각 6종 × 5장, 카메라 각도 60/90° 반반),
#   '잘라낸 알약' 위에 세 가지 방식의 검출 박스를 그려 PNG 로 저장해요.
#   · CRAFT          : PARSeq / TrOCR 이 쓰는 글자 위치 찾기 (전처리 P0)
#   · Paddle 기본    : PaddleOCR 기본 검출 기준 (전처리 P1)
#   · Paddle 0.0     : PaddleOCR 검출 기준 0.0   (전처리 P1)
#   본 실험 점수·CSV 는 건드리지 않아요. (사진 30장을 한 번씩만 읽음, 몇 분)
#
#  [저장]  --out 폴더
#   boxes_NN_....png : 사진별 2×2 비교 (잘라낸 알약 / CRAFT / Paddle 기본 / Paddle 0.0)
#   sheet_1.png ...  : 6장씩 모은 그림 (보고서용)
#   boxes.csv        : 사진별·방식별 박스 개수, 좌표, 읽은 글자
#
#  [주의]  박스는 알약을 돌리지 않은 상태(0°)에서 한 번만 찾은 것이에요.
#          본 실험은 둥근 알약을 12방향으로 돌려 읽으므로, 이 그림은 '대표 한 장면'입니다.
#
#  [실행 예]
#     python viz_boxes_exp300.py --manifest manifest.csv --out viz_boxes --n 30 --seed 42
# =====================================================================

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

import run_exp300 as base
from run_exp300_paddle import PaddleOCRModel, PaddleOCRModelT0     # 두 설정 등록 + P5~P10 추가
from run_exp300 import imread_any, crop_pill

COLORS = {"craft": (60, 180, 60), "paddle": (220, 90, 30), "paddle_t0": (30, 140, 240)}   # BGR
TITLES = {"craft": "EasyOCR CRAFT", "paddle": "Paddle default", "paddle_t0": "Paddle thr=0.0"}
METHODS = ["craft", "paddle", "paddle_t0"]
CELL_W = 420


# ----------------------------- 사진 고르기 -----------------------------
def sample_rows(rows, n=30, seed=42):
    """회전각(rot_deg)별로 똑같이 n/6장씩, 카메라 각도는 두 값이 반반이 되게 뽑음 (시드 고정)"""
    rng = random.Random(seed)
    by_rot = defaultdict(list)
    for r in rows:
        by_rot[str(r.get("rot_deg", "")).strip()].append(r)
    rots = sorted(by_rot)
    cams = [c for c, _ in Counter(r.get("camera_la", "") for r in rows).most_common(2)]
    per = n // len(rots)
    extra = n - per * len(rots)                       # 나누어떨어지지 않으면 앞쪽 회전각에 한 장씩 더
    picked = []
    for i, rot in enumerate(rots):
        pool = by_rot[rot][:]
        rng.shuffle(pool)
        k = per + (1 if i < extra else 0)
        first = cams[i % 2] if cams else ""            # 회전각마다 번갈아 한쪽 카메라 각도를 더 많이 (전체는 반반)
        second = cams[(i + 1) % 2] if len(cams) > 1 else ""
        a = [r for r in pool if r.get("camera_la") == first][: (k + 1) // 2]
        b = [r for r in pool if r.get("camera_la") == second][: k - len(a)]
        rest = [r for r in pool if r not in a and r not in b]
        b += rest[: k - len(a) - len(b)]               # 한쪽이 모자라면 아무거나 채움
        picked += a + b
    rng.shuffle(picked)
    return picked


# ----------------------------- 검출 -----------------------------
def craft_detect(g):
    return [list(b) for b in base.craft_boxes(g)], []            # 상자 [(x0,y0,x1,y1)], 글자 없음


def paddle_detect(model, g):
    """PaddleOCR 한 설정으로 (상자 목록, 읽은 글자 목록). 상자는 (x0,y0,x1,y1) 바깥 사각형, 폴리곤도 같이"""
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    polys, texts = [], []
    if model.v3:
        for res in model.ocr.predict(rgb):
            rp = res.get("rec_polys")
            if rp is None or len(rp) == 0:
                rp = res.get("dt_polys")
            polys += [np.asarray(p).reshape(-1, 2) for p in (rp if rp is not None else [])]
            texts += [str(t) for t in res.get("rec_texts", [])]
    else:
        lines = model.ocr.ocr(rgb, cls=False)
        for box, (t, c) in (lines[0] if lines and lines[0] else []):
            polys.append(np.asarray(box).reshape(-1, 2))
            texts.append(str(t))
    boxes = [[float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())] for p in polys]
    return boxes, texts, polys


# ----------------------------- 그리기 -----------------------------
def ascii_only(s):
    return "".join(ch if ord(ch) < 128 else "?" for ch in str(s))


def draw_panel(g, method, boxes, texts, polys=None, err=None):
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    col = COLORS[method]
    if polys:
        for p in polys:
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, col, 2)
    else:
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
    for k, (b, t) in enumerate(zip(boxes, texts)):
        cv2.putText(img, ascii_only(t)[:14], (int(b[0]), max(12, int(b[1]) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    head = f"{TITLES[method]}  boxes={len(boxes)}" + (f"  ERR" if err else "")
    return img, head


def pad_cell(img, head, col, w, h):
    cell = np.full((h + 30, w, 3), 255, np.uint8)
    y = 30 + (h - img.shape[0]) // 2
    x = (w - img.shape[1]) // 2
    cell[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(cell, head, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.rectangle(cell, (0, 0), (w - 1, h + 29), (190, 190, 190), 1)
    return cell


def compose(g, results, caption):
    """2×2 : 잘라낸 알약 / CRAFT / Paddle 기본 / Paddle 0.0"""
    H, W = g.shape[:2]
    w, h = max(CELL_W, W + 20), H + 20
    cells = [pad_cell(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "cropped pill (P0)", (60, 60, 60), w, h)]
    for m in METHODS:
        img, head, _ = results[m]
        cells.append(pad_cell(img, head, COLORS[m], w, h))
    grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    bar = np.full((34, grid.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def make_sheets(files, out_dir, per_sheet=6, cols=3):
    for s in range(0, len(files), per_sheet):
        ims = [cv2.imread(f) for f in files[s:s + per_sheet]]
        ims = [cv2.resize(i, (800, int(i.shape[0] * 800 / i.shape[1]))) for i in ims]
        rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
        hs = [max(i.shape[0] for i in r) for r in rows]
        canvas = np.full((sum(hs), 800 * cols, 3), 255, np.uint8)
        y = 0
        for r, h in zip(rows, hs):
            for k, im in enumerate(r):
                canvas[y:y + im.shape[0], k * 800:k * 800 + im.shape[1]] = im
            y += h
        cv2.imwrite(os.path.join(out_dir, f"sheet_{s // per_sheet + 1}.png"), canvas)


# ----------------------------- 실행 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="viz_boxes")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    picked = sample_rows(rows, a.n, a.seed)
    cnt_rot, cnt_cam = defaultdict(int), defaultdict(int)
    for r in picked:
        cnt_rot[r.get("rot_deg", "")] += 1
        cnt_cam[r.get("camera_la", "")] += 1
    print(f"뽑은 사진 {len(picked)}장 | 회전각별 {dict(sorted(cnt_rot.items()))} | 카메라각도별 {dict(sorted(cnt_cam.items()))}")

    print("모델 불러오는 중... (EasyOCR CRAFT, PaddleOCR 두 설정)")
    base.easy_reader()
    models = {"paddle": PaddleOCRModel(), "paddle_t0": PaddleOCRModelT0()}
    p0, p1 = base.PREPS["P0"][1], base.PREPS["P1"][1]

    csv_rows, files = [], []
    for k, r in enumerate(picked, 1):
        img = imread_any(r["image_path"])
        if img is None:
            print(f"  [경고] 이미지를 못 읽음: {r['image_path']}")
            continue
        crop, *_ = crop_pill(img)
        g0, g1 = p0(crop), p1(crop)
        results = {}
        for m in METHODS:
            t0, err, polys = time.time(), None, None
            try:
                if m == "craft":
                    boxes, texts = craft_detect(g0)
                    g = g0
                else:
                    boxes, texts, polys = paddle_detect(models[m], g1)
                    g = g1
            except Exception as e:                       # 한 방식이 실패해도 나머지는 계속
                boxes, texts, g, err = [], [], g1, f"{type(e).__name__}: {e}"
                print(f"  [경고] {r['file_name'][:20]} {m}: {err}")
            im, head = draw_panel(g, m, boxes, texts, polys, err)
            results[m] = (im, head, boxes)
            csv_rows.append({"file_name": r["file_name"], "rot_deg": r.get("rot_deg", ""), "camera_la": r.get("camera_la", ""),
                             "form": r.get("form", ""), "truth": r.get("truth", ""), "method": m,
                             "prep": "P0" if m == "craft" else "P1", "n_boxes": len(boxes), "n_text": len(texts),
                             "ms": int((time.time() - t0) * 1000), "error": err or "",
                             "boxes": json.dumps([[round(v) for v in b] for b in boxes]), "texts": "|".join(texts)})
        cap = f"{k:02d}  rot={r.get('rot_deg', '')}  camera={r.get('camera_la', '')}  truth={ascii_only(r.get('truth', ''))}"
        path = os.path.join(a.out, f"boxes_{k:02d}_rot{r.get('rot_deg', '')}_cam{r.get('camera_la', '')}.png")
        cv2.imwrite(path, compose(g0, results, cap))
        files.append(path)
        print(f"  {k:2d}/{len(picked)}  " + "  ".join(f"{TITLES[m]}={len(results[m][2])}" for m in METHODS))

    make_sheets(files, a.out)
    with open(os.path.join(a.out, "boxes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    print("\n[방식별 요약]")
    print(f"{'방식':<16} {'사진':>4} {'평균 박스':>9} {'박스 0개':>8} {'오류':>4}")
    for m in METHODS:
        rs = [x for x in csv_rows if x["method"] == m]
        if rs:
            print(f"{TITLES[m]:<16} {len(rs):4d} {np.mean([x['n_boxes'] for x in rs]):9.1f} "
                  f"{sum(x['n_boxes'] == 0 for x in rs):8d} {sum(bool(x['error']) for x in rs):4d}")
    print(f"\n저장 완료: {a.out}/  (사진별 PNG {len(files)}장, sheet_*.png, boxes.csv)")


if __name__ == "__main__":
    main()
