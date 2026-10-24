# =====================================================================
#  viz_align_exp300.py  —  각인 정렬 실험: 12방향 다수결(지금) vs ①~⑥ 비교 그림 + 지표 (PARSeq + P0)
# =====================================================================
#
#  [하는 일]
#   사진마다 run_exp300.py / align_exp300.py 의 '방향 고르기'를 전부 한 번에 돌려서 정답 각인과 비교해요.
#     vote       : 12방향 다수결 (지금 방식)                 conf       : ② 12방향 중 신뢰도 최고
#     box        : ① CRAFT 상자 (+ 없으면 ③)                 pbox       : ① PaddleOCR 상자 (+ 없으면 ③), 위아래는 신뢰도
#     pbox_cls   : ① PaddleOCR 상자, 위아래는 Paddle 분류기   stroke     : ③ 획 방향만
#     polar      : ④ 극좌표 펼치기                            polar_conf : ④+② 펼친 띠 + 12방향 중 신뢰도 최고
#     cnn        : ⑤ 각도 분류기 (가짜 사진 학습)             tps        : ⑥ TPS 펴기 (가짜 사진 학습) + 12방향
#   두 길 합치기 (⑥ 은 휜 글씨 전용으로 보고, 곧은 글씨용으로 따로 튜닝하지 않음)
#     conf_perm  : 길 A = ② 방향 + 그 방향의 박스 글자를 여러 순서로 붙여 DB 와 가장 비슷한 것
#     two_conf   : 길 A 와 길 B(⑥ TPS) 중 PARSeq 확신도가 높은 쪽 (길쭉한 알약은 A 만)
#     two_db     : 길 A 와 길 B 중 dbscore(DB 후보 중 최고, 글자당 평균 로그확률)가 높은 쪽
#                  A 이미지 = 고른 방향의 박스를 읽는 순서대로 이어 붙인 것, B 이미지 = TPS 로 펴진 띠
#   12방향 읽기는 한 번만 하고 vote / conf / polar_conf 가 같이 써요 (같은 사진은 언제 읽어도 같은 결과).
#   길쭉한 알약은 모든 방식이 0°/180° 만 읽으므로 차이는 둥근 알약에서만 나요.
#
#  [저장]  --out 폴더
#   align_NN_....png  : 사진별 2줄×5칸 [잘라낸 알약 | 다수결 | ② | ①CRAFT | ①Paddle / ③ | ④ | ④+② | ⑤ | ⑥] + 요약 칸
#   sheet_N.png       : 6장씩 모은 그림
#   align_results.csv : 사진별·방식별 고른 각도, 읽은 글자, 판정, 찾은 약, 걸린 시간
#   align_metrics.csv : 방식별 지표 (전체 / 둥근 / 길쭉 / 둥근 카메라 60°·90°) + 정렬 정답률 + 평균 시간
#   ※ PNG 에는 원본 사진이 들어 있어 GitHub 에 올리지 않아요 (.gitignore).
#
#  [정렬이 맞았는지]  같은 약·같은 면이면 '고른 각도 ± 파일 이름 회전각'이 일정해야 해요.
#     정렬 정답률 = 같은 약·면의 다른 사진들로 정한 기준값과 20° 안쪽인 사진 비율 (180° 뒤집힘은 같은 것으로 봄,
#     PARSeq 가 뒤집힌 글자도 읽기 때문). 회전각은 이 확인에만 쓰고, 읽기·고르기에는 쓰지 않아요.
#
#  [실행 예]  (저장소 맨 위 폴더에서)
#     python exp300\viz_align_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db.csv --out exp300\results\viz_align2
# =====================================================================

import argparse
import csv
import itertools
import os
import time
from collections import defaultdict

import torch                                         # Paddle 보다 먼저 (align_exp300 참고)

import cv2
import numpy as np

import align_exp300 as al
import run_exp300 as base
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import JCOL, ascii_only, judge

METHODS = ["vote", "conf", "box", "pbox", "pbox_cls", "stroke", "polar", "polar_conf", "cnn", "tps",
           "conf_perm", "two_conf", "two_db"]
PANELS = [["crop", "vote", "conf", "box", "pbox"], ["stroke", "polar", "polar_conf", "cnn", "tps"],
          ["conf_perm", "two_conf", "two_db", None, None]]
TITLES = {"vote": "12-dir vote (now)", "conf": "(2) max conf", "box": "(1) CRAFT box+(3)", "pbox": "(1) Paddle box+(3)",
          "pbox_cls": "(1) Paddle + ori-cls", "stroke": "(3) stroke only", "polar": "(4) polar unwrap",
          "polar_conf": "(4)+(2) polar|12-dir", "cnn": "(5) angle CNN", "tps": "(6) TPS + 12-dir",
          "conf_perm": "A: (2)+box order perm", "two_conf": "A|B(6) by PARSeq conf", "two_db": "A|B(6) by dbscore"}
COLORS = {"vote": (60, 180, 60), "conf": (200, 60, 160), "box": (30, 140, 240), "pbox": (220, 90, 30),
          "pbox_cls": (160, 60, 20), "stroke": (0, 170, 200), "polar": (130, 0, 180), "polar_conf": (90, 40, 140),
          "cnn": (40, 40, 200), "tps": (0, 120, 120), "conf_perm": (150, 80, 200), "two_conf": (20, 100, 60),
          "two_db": (0, 80, 160)}                                                           # BGR
TWO_PATH = {"conf_perm", "two_conf", "two_db"}     # 이 방식들은 약을 직접 정함 (DB 매칭을 내부에서 끝냄)
LABELS = {**base.ALIGNS, "conf_perm": "길A: ② + 박스 순서 바꾸기", "two_conf": "두 길(A | ⑥) PARSeq 확신도로",
          "two_db": "두 길(A | ⑥) dbscore 로"}
CELL = 360
HEAD, FOOT = 28, 54
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


class Recorder:
    """PARSeq 를 감싸서 읽을 때마다 (돌린 사진, 상자, 상자별 글자)를 기억 (그림 그리기용). 읽는 결과는 그대로"""
    def __init__(self, parseq):
        self.m, self.log = parseq, []

    def __call__(self, g):
        t, c, boxes, texts = self.m.read_boxes(g)
        self.log.append((g, boxes, texts))
        return t, c


def perm_best(bt, db):
    """박스 글자들을 여러 순서로 붙여 DB 와 가장 비슷한 것 → (글자, 약, 유사도). 박스 4개 이하면 모든 순서, 넘으면 원래·거꾸로"""
    texts = [t for _, t in bt]
    if len(texts) < 2:
        cands = [base.norm("".join(texts))]
    else:
        order = itertools.permutations(range(len(texts))) if len(texts) <= 4 else [tuple(range(len(texts))),
                                                                                  tuple(reversed(range(len(texts))))]
        cands = [base.norm("".join(texts[i] for i in p)) for p in order]
    best = ("", "", 0.0)
    for t in cands:
        if t:
            _, d, _, s, _ = base.identify(t, "", "", db)
            if d and s > best[2]:
                best = (t, d, s)
    return best


def db_image_score(scorer, g, boxes):
    """그 사진(이미 돌린 것)에서 박스를 읽는 순서대로 이어 붙인 이미지(박스 없으면 사진 전체)의 dbscore 최고값 (글자당 평균 로그확률)"""
    from dbscore_exp300 import hconcat, ordered_boxes
    full = not boxes or (len(boxes) == 1 and tuple(boxes[0]) == (0, 0, g.shape[1], g.shape[0]))
    crops = [g] if full else [g[y0:y1, x0:x1] for x0, y0, x1, y1 in ordered_boxes(list(boxes))]
    img = crops[0] if len(crops) == 1 else hconcat(crops)
    return float(scorer.score([img])[1].max())


def add_two_path(res, aspect, db, scorer):
    """'두 길 합치기': 길 A = ② 방향 + 박스 순서 바꾸기, 길 B = ⑥ TPS (둥근 알약만). 고르는 기준 = PARSeq 확신도 / dbscore"""
    c = res["conf"]
    g, boxes, texts = c["view_src"]
    tA, dA, _ = perm_best([(b, t) for b, t in zip(boxes, texts) if t], db)
    if not tA:                                                        # 순서를 바꿔도 DB 와 안 맞으면 ② 그대로
        tA, dA = c["read"], base.identify(c["read"], "", "", db)[1]
    A = {"read": tA, "conf": c["conf"], "angle": c["angle"], "cands": c["cands"], "log": c["log"], "view_src": c["view_src"],
         "info": {"src": "A"}, "ms": c["ms"], "drug": dA}
    res["conf_perm"] = dict(A)
    if aspect > 1.3:                                                  # 길쭉: TPS 를 안 쓰므로 A 만
        res["two_conf"], res["two_db"] = dict(A), dict(A)
        return
    t = res["tps"]
    B = {**t, "drug": base.identify(t["read"], "", "", db)[1], "info": {**t["info"], "src": "B"}}
    sA, sB = db_image_score(scorer, g, boxes), float(scorer.score([t["info"]["rect"]])[1].max())
    for name, pickB in (("two_conf", B["conf"] > A["conf"]), ("two_db", sB > sA)):
        r = dict(B if pickB else A)
        r["info"] = {**r["info"], "dbA": round(sA, 3), "dbB": round(sB, 3), "confA": round(A["conf"], 1),
                     "confB": round(B["conf"], 1)}
        r["ms"] = A["ms"] + t["ms"]
        res[name] = r


def run_methods(g0, aspect, rec, step, db=None, scorer=None):
    """모든 방식 실행 → {방식: dict(read, conf, angle, cands, view_src, info, ms)}"""
    res = {}

    def call(name, fn):
        start, t0 = len(rec.log), time.time()
        t, c, cands, ang, info = fn()
        res[name] = {"read": t, "conf": c, "angle": ang, "cands": cands, "info": info,
                     "log": rec.log[start:], "ms": (time.time() - t0) * 1000}

    call("vote", lambda: base.read_pill(g0, aspect, rec, step, "vote"))
    log12, c12 = res["vote"]["log"], res["vote"]["cands"]
    best = base.pick_by_conf(c12)                                     # ② = 같은 12번 읽기에서 신뢰도 최고
    res["conf"] = {"read": best[0], "conf": best[1], "angle": best[2], "cands": c12, "log": log12,
                   "info": {"src": "rot", "theta": ""}, "ms": res["vote"]["ms"]}
    call("box", lambda: base.read_aligned(g0, aspect, rec))
    call("pbox_cls", lambda: al.read_pbox(g0, aspect, rec, use_cls=True))
    pc = res["pbox_cls"]                                               # 같은 읽기에서 위아래를 신뢰도로 고른 것 = pbox
    best = base.pick_by_conf(pc["cands"])
    res["pbox"] = {**pc, "read": best[0], "conf": best[1], "angle": best[2]}
    call("stroke", lambda: al.read_stroke(g0, aspect, rec))
    call("polar", lambda: al.read_polar(g0, aspect, rec))
    if aspect > 1.3:                                                   # 길쭉: ④ 는 0/180 그대로 → ④+② 도 같음
        res["polar_conf"] = dict(res["polar"])
    else:
        cands = res["polar"]["cands"] + c12
        best = base.pick_by_conf(cands)
        res["polar_conf"] = {"read": best[0], "conf": best[1], "angle": best[2], "cands": cands,
                             "log": res["polar"]["log"] + log12, "info": {**res["polar"]["info"], "src": "polar+rot"},
                             "ms": res["polar"]["ms"] + res["vote"]["ms"]}
    call("cnn", lambda: al.read_cnn(g0, aspect, rec))
    call("tps", lambda: al.read_tps(g0, aspect, rec, step))
    for r in res.values():                                             # 고른 후보가 몇 번째 읽기였나 → 그림
        idx = [(t, c, a) for t, c, a in r["cands"]].index((r["read"], r["conf"], r["angle"]))
        r["view_src"] = r["log"][idx] if idx < len(r["log"]) else None
    add_two_path(res, aspect, db, scorer)
    return res


# ----------------------------- 그리기 -----------------------------
def fit(img, size=CELL):
    s = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def view_read(r, g0, col):
    """고른 방향의 사진 + 그 사진에서 찾은 상자와 글자. ⑥ 은 돌린 알약 위에 펴진 띠를 붙여 보여 줌"""
    if r["view_src"] is None:                                          # ⑥ TPS
        g = al.rotate_same(g0, r["angle"])
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        rect = r["info"].get("rect")
        if rect is not None:
            strip = cv2.cvtColor(cv2.resize(rect, (img.shape[1], int(img.shape[1] * 32 / 128))), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(strip, (0, 0), (strip.shape[1] - 1, strip.shape[0] - 1), col, 3)
            img = np.vstack([img, strip])
        return fit(img), "-"
    g, boxes, texts = r["view_src"]
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    full = len(boxes) == 1 and tuple(boxes[0]) == (0, 0, g.shape[1], g.shape[0])
    if not full:
        for (x0, y0, x1, y1), t in zip(boxes, texts):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
            cv2.putText(img, ascii_only(t)[:12], (int(x0), max(14, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2, cv2.LINE_AA)
    return fit(img), (0 if full else len(boxes))


def arrow(img, theta, col, scale):
    if theta in ("", None):
        return
    h, w = img.shape[:2]
    t = np.radians(float(theta))
    d = np.array([np.cos(t), np.sin(t)]) * min(h, w) * scale
    c = np.array([w / 2, h / 2])
    cv2.arrowedLine(img, tuple(np.int32(c - d)), tuple(np.int32(c + d)), col, 2, cv2.LINE_AA, tipLength=0.08)


def view_crop(g, res):
    """잘라낸 알약(0°): ③ 획(노랑), ① CRAFT 상자(주황)·Paddle 상자(파랑), 각 방식이 추정한 글줄 방향 화살표"""
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    st = res["stroke"]["info"].get("stroke")
    if st is not None:
        img[st] = (0.5 * img[st] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    for m in ("box", "pbox"):
        for p in res[m]["info"].get("polys") or []:
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, COLORS[m], 2)
    arrow(img, res["box"]["info"].get("theta"), COLORS["box"], 0.44)
    arrow(img, res["pbox"]["info"].get("theta"), COLORS["pbox"], 0.38)
    arrow(img, res["stroke"]["info"].get("theta"), COLORS["stroke"], 0.32)
    arrow(img, res["cnn"]["info"].get("theta"), COLORS["cnn"], 0.26)
    return fit(img)


def cell(img, head, col, foots):
    out = np.full((HEAD + CELL + FOOT, CELL + 16, 3), 255, np.uint8)
    y, x = HEAD + (CELL - img.shape[0]) // 2, 8 + (CELL - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(out, head, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
    for k, (t, c) in enumerate(foots):
        cv2.putText(out, t[:34], (6, HEAD + CELL + 20 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (190, 190, 190), 1)
    return out


def ang_txt(a):
    return a if isinstance(a, str) else f"{a:g}"


def summary(res, truth, truth_drug, height, width=430):
    col = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(col, (0, 0), (width - 1, height - 1), (190, 190, 190), 1)

    def put(y, t, c=(60, 60, 60), s=0.5, th=1):
        cv2.putText(col, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)

    put(28, "SUMMARY", (0, 0, 0), 0.7, 2)
    put(56, f"truth : {ascii_only(truth)}  ({truth_drug})", s=0.52, th=2)
    y = 64
    for m in METHODS:
        r = res[m]
        ok = r["pred_drug"] == truth_drug
        y += 38
        put(y, f"{TITLES[m]}", COLORS[m], 0.5, 2)
        y += 22
        put(y, f"  {ascii_only(r['read'])[:13] or '(empty)'} [{r['judge']}]  drug {'OK' if ok else 'NG'}"
               f"  {ang_txt(r['angle'])}", JCOL[r["judge"]] if ok or r["judge"] == "O" else (0, 0, 200), 0.48, 2)
    y += 34
    put(y, "O=exact ~=contains X=wrong | last = angle", (120, 120, 120), 0.45)
    y += 20
    put(y, "arrows on crop: line dir per method (colors)", (120, 120, 120), 0.45)
    return col


def compose(g0, res, row, caption):
    truth = base.norm(row["truth"])
    rows = []
    for prow in PANELS:
        cells = []
        for m in prow:
            if m is None:                                              # 빈 칸
                cells.append(np.full((HEAD + CELL + FOOT, CELL + 16, 3), 255, np.uint8))
                continue
            if m == "crop":
                cells.append(cell(view_crop(g0, res), "cropped pill (P0, 0deg)", (60, 60, 60),
                                  [(f"truth: {ascii_only(truth)}", (60, 60, 60)),
                                   (f"src: box={res['box']['info'].get('src')} paddle={res['pbox']['info'].get('src')}",
                                    (90, 90, 90))]))
                continue
            r = res[m]
            extra = f"  cls p180={r['info'].get('p180')}" if m == "pbox" and "p180" in r["info"] else ""
            if m in ("two_conf", "two_db") and "dbA" in r["info"]:
                i = r["info"]
                extra = (f"  {r['info']['src']}: conf {i['confA']:.0f}/{i['confB']:.0f}" if m == "two_conf"
                         else f"  {r['info']['src']}: db {i['dbA']:.2f}/{i['dbB']:.2f}")
            cells.append(cell(r["view"], f"{TITLES[m]}  ang={ang_txt(r['angle'])}", COLORS[m],
                              [(f"[{r['judge']}] read: {ascii_only(r['read']) or '(empty)'}", JCOL[r["judge"]]),
                               (f"truth: {ascii_only(truth)}{extra}", (60, 60, 60))]))
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    grid = np.hstack([grid, summary(res, truth, row["drug_N"], grid.shape[0])])
    bar = np.full((34, grid.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def make_sheets(files, out_dir, per_sheet=6, cols=2, width=1400):
    for s in range(0, len(files), per_sheet):
        ims = [base.imread_any(f) for f in files[s:s + per_sheet]]
        ims = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1])), interpolation=cv2.INTER_AREA) for i in ims]
        rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
        hs = [max(i.shape[0] for i in r) for r in rows]
        canvas = np.full((sum(hs), width * cols, 3), 255, np.uint8)
        y = 0
        for r, h in zip(rows, hs):
            for k, im in enumerate(r):
                canvas[y:y + im.shape[0], k * width:k * width + im.shape[1]] = im
            y += h
        cv2.imencode(".png", canvas)[1].tofile(os.path.join(out_dir, f"sheet_{s // per_sheet + 1}.png"))


# ----------------------------- 지표 -----------------------------
def scores(rows):
    """[정답률, 정확도, 정밀도, 민감도, F1] (0~1)"""
    acc, p, r, f = drug_metrics([x["truth_drug"] for x in rows], [x["pred_drug"] for x in rows])
    imp = float(np.mean([x["judge"] == "O" for x in rows]))
    return [imp, acc, p, r, f]


def align_accuracy(out_rows, m, sign, tol=20):
    """정렬 정답률 (둥근 알약). 같은 약·같은 면(정답 각인)끼리 r = 각도 - sign×회전각 (180° 주기) 을 구하고,
       나를 뺀 나머지의 원형 평균과 tol° 안쪽이면 '맞음'. 같은 약·면 사진이 3장 이상일 때만 셈.
       돌려주는 값: (맞은 비율, 센 사진 수). ④ 처럼 각도가 숫자가 아닌(펼친 띠) 사진은 뺌"""
    by = defaultdict(list)
    for x in out_rows:
        if x["method"] == m and x["shape"] == "둥근":
            try:
                a = float(x["angle"])
            except ValueError:
                continue
            by[(x["truth_drug"], x["truth_imprint"])].append((a - sign * float(x["rot_deg"] or 0)) % 180)
    ok = n = 0
    for v in by.values():
        if len(v) < 3:
            continue
        t = np.radians(np.array(v) * 2)
        for i in range(len(v)):
            rest = np.delete(t, i)
            ref = np.degrees(np.arctan2(np.sin(rest).mean(), np.cos(rest).mean())) / 2
            d = abs((v[i] - ref + 90) % 180 - 90)
            ok += d <= tol
            n += 1
    return (ok / n if n else float("nan")), n


# ----------------------------- 실행 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="viz_align2")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞에서 N장만")
    ap.add_argument("--no-png", action="store_true", help="그림 없이 CSV·지표만")
    ap.add_argument("--aligned-dir", default="", help="방식별 하위 폴더에 '고른 방향으로 세운 알약'만 저장 (원본 사진 포함, GitHub 금지)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, a.split)            # 정답이 '마크'뿐인 사진은 여기서 빠짐
    if a.limit:
        items = items[: a.limit]
    db = base.load_db(a.db)
    print(f"사진 {len(items)}장 ({a.split}), PARSeq + P0, GPU={base._gpu()}")
    rec = Recorder(base.PARSeq())
    prep = base.PREPS["P0"][1]
    from dbscore_exp300 import DBScorer, db_candidates
    scorer = DBScorer(rec.m, [c["text"] for c in db_candidates(a.db)])      # 두 길 합치기의 dbscore 기준용

    crops = {it["file_name"]: base.crop_pill(base.imread_any(it["image_path"])) for it in items}
    items.sort(key=lambda it: (base.shape_group(crops[it["file_name"]][1]) != "둥근", it["drug_N"], it["file_name"]))

    out_rows, files = [], []
    for k, it in enumerate(items, 1):
        gray, aspect, color, ok = crops[it["file_name"]]
        g0 = prep(gray)
        shape = base.shape_group(aspect)
        truth = base.norm(it["truth"])
        t0 = time.time()
        rec.log = []
        res = run_methods(g0, aspect, rec, a.step, db, scorer)
        for m in METHODS:
            r = res[m]
            r["view"], r["n_box"] = view_read(r, g0, COLORS[m])
            r["judge"] = judge(r["read"], truth)
            r["pred_drug"] = (r["drug"] if m in TWO_PATH else base.identify(r["read"], shape, color, db)[1]) or NONE_LABEL
            info = r["info"]
            out_rows.append({
                "file_name": it["file_name"], "split": it["split"], "shape": shape, "method": m,
                "truth_drug": it["drug_N"], "truth_imprint": truth, "read": r["read"], "conf": round(r["conf"], 1),
                "angle": r["angle"], "judge": r["judge"], "pred_drug": r["pred_drug"],
                "drug_ok": int(r["pred_drug"] == it["drug_N"]), "n_box_read": r["n_box"],
                "align_src": info.get("src", ""), "align_theta": info.get("theta", ""),
                "n_box_0deg": info.get("n_box", ""), "p180": info.get("p180", ""), "ms": round(r["ms"]),
                "dbA": info.get("dbA", ""), "dbB": info.get("dbB", ""), "confA": info.get("confA", ""),
                "confB": info.get("confB", ""),
                "candidates": "|".join(f"{x[0] or '-'}@{ang_txt(x[2])}:{x[1]:.0f}" for x in r["cands"]),
                "camera_la": it.get("camera_la", ""), "rot_deg": it.get("rot_deg", ""), "form": it.get("form", ""),
            })
        tag = "round" if shape == "둥근" else "long"
        if a.aligned_dir:                                  # 세운 알약만: 번호_약코드_rot회전각_cam_ang고른각도_각인판정_약판정
            for m in METHODS:
                r = res[m]
                src = r["view_src"][0] if r["view_src"] is not None else al.rotate_same(g0, r["angle"])
                sub = os.path.join(a.aligned_dir, m)
                os.makedirs(sub, exist_ok=True)
                jn = {"O": "O", "~": "part", "X": "X", "-": "none"}[r["judge"]]
                ok = "OK" if r["pred_drug"] == it["drug_N"] else "NG"
                ang = str(ang_txt(r["angle"])).replace(".", "p")
                cv2.imencode(".png", src)[1].tofile(os.path.join(
                    sub, f"{k:03d}_{it['drug_N']}_rot{it.get('rot_deg', '')}_cam{it.get('camera_la', '')}_ang{ang}_{jn}_{ok}.png"))
        if not a.no_png:
            cap = (f"{k:02d}  {ascii_only(it['drug_N'])}  {tag}  rot={it.get('rot_deg', '')}  "
                   f"camera={it.get('camera_la', '')}  truth={ascii_only(truth)}")
            path = os.path.join(a.out, f"align_{k:03d}_{tag}_{it['drug_N']}_rot{it.get('rot_deg', '')}"
                                       f"_cam{it.get('camera_la', '')}.png")
            cv2.imencode(".png", compose(g0, res, it, cap))[1].tofile(path)
            files.append(path)
        print(f"  {k:3d}/{len(items)} {tag:<5} {truth:<13} " +
              " ".join(f"{m}={res[m]['read'] or '-'}[{res[m]['judge']}]" for m in METHODS if m in ("vote", "conf", "pbox", "polar_conf", "cnn", "tps")) +
              f"  ({time.time() - t0:.1f}s)")

    if files:
        make_sheets(files, a.out)
    with open(os.path.join(a.out, "align_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    # ---- 지표
    groups = [("전체", lambda x: True), ("둥근", lambda x: x["shape"] == "둥근"), ("길쭉", lambda x: x["shape"] == "길쭉"),
              ("둥근·카메라60", lambda x: x["shape"] == "둥근" and x["camera_la"] == "60"),
              ("둥근·카메라90", lambda x: x["shape"] == "둥근" and x["camera_la"] == "90")]
    signs = {s: np.nanmean([align_accuracy(out_rows, m, s)[0] for m in METHODS]) for s in (1, -1)}
    sign = max(signs, key=lambda s: -1 if np.isnan(signs[s]) else signs[s])
    met_rows = []
    for gname, cond in groups:
        n = len({x["file_name"] for x in out_rows if cond(x)})
        if not n:
            continue
        print(f"\n[{gname} {n}장]  (1장 = {100 / n:.1f}%p)")
        print(f"  {'방식':<34} " + " ".join(f"{m:>8}" for m in METRICS) + "   O / ~ / X")
        for m in METHODS:
            sub = [x for x in out_rows if x["method"] == m and cond(x)]
            v = scores(sub)
            oxs = [sum(x["judge"] == j for x in sub) for j in "O~X"]
            print(f"  {LABELS[m]:<34} " + " ".join(f"{s * 100:7.1f}%" for s in v) + f"   {oxs[0]} / {oxs[1]} / {oxs[2]}")
            row = {"group": gname, "n": n, "method": m, **{k: round(s * 100, 1) for k, s in zip(METRICS, v)},
                   "O": oxs[0], "~": oxs[1], "X": oxs[2], "ms_mean": round(float(np.mean([x["ms"] for x in sub])))}
            if gname == "둥근":
                acc, na = align_accuracy(out_rows, m, sign)
                row.update({"align_acc": round(acc * 100, 1) if na else "", "align_n": na})
            met_rows.append(row)
    with open(os.path.join(a.out, "align_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in met_rows for k in r)))
        w.writeheader()
        w.writerows(met_rows)

    print(f"\n[둥근 알약: 정렬 정답률(같은 약·면 기준값과 20° 안, 180° 뒤집힘 허용) / 평균 시간]  회전각 부호 {'+' if sign > 0 else '-'}")
    for r in met_rows:
        if r["group"] == "둥근":
            al_txt = f"{r['align_acc']:5.1f}% ({r['align_n']}장)" if r.get("align_n") else "  계산 불가(같은 약·면 3장 미만)"
            print(f"  {LABELS[r['method']]:<34} 정렬 {al_txt}   평균 {r['ms_mean']} ms/장")
    src = defaultdict(lambda: defaultdict(int))
    for x in out_rows:
        if x["shape"] == "둥근" and x["method"] in ("box", "pbox"):
            src[x["method"]][x["align_src"]] += 1
    print(f"\n① 각도를 어디서 얻었나 (둥근): CRAFT {dict(src['box'])} / Paddle {dict(src['pbox'])}"
          f"   (box=상자, stroke=획 PCA, 4=방향 애매해서 4번 읽음)")
    print("※ 정답률 = 읽은 각인이 정답과 완전히 같음(O), 정확도 = 찾은 약이 정답 약 (edit 매칭), 정밀도·민감도·F1 = 약 종류별 평균(macro)")
    print(f"저장 완료: {a.out}  (사진별 PNG {len(files)}장, sheet_*.png, align_results.csv, align_metrics.csv)")


if __name__ == "__main__":
    main()
