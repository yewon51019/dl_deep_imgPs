# =====================================================================
#  parseq_labels_cns.py  —  PARSeq 파인튜닝용 글자 조각 라벨 (학습 계획 4번)
# =====================================================================
#
#  [읽는 것]  학습용 전체(trainall, 개발용 36종·Λ 5종 제외) 4,296장을 지금 방식으로 읽은 결과
#             ocr_v1\trainall\paddle_ft1_u2.0\orig_alignlong_line\pred.csv  (ft1 검출기 u2.0 + 축 정렬 + 줄 묶기)
#  [틀린 사진 원인 (자동 분류, 앞 칸부터 먼저 해당하는 것)]
#     order  순서만 다름  : 읽은 글자를 정렬하면 정답과 같음 → 상자별 글자가 정답 조각이라 사람 검수 없이 씀
#     other  다른 방향에 정답 : 12(또는 2)방향 중 정답과 완전일치한 방향이 있음 → 그 방향 상자를 씀 (검수 없음)
#     nobox  상자 0개 / big 큰 상자(넓이 ≥ 400×400 의 절반) / part 일부만 맞음(정답 글자의 절반 이상이 읽은 글자에 있음) / wrong 거의 틀림
#  [--step review]  사람 검수 폴더: nobox·big·part·wrong 중에서 약마다 최대 3장 (시드 42, 카메라·면이 섞이게 돌아가며 고름)
#             - 이미 사람이 상자를 본 학습용 500장(det_label_v1 train500)은 뺌 (중복 작업 방지)
#             → exp_cns\results\parseq_label_v1\review\  사진(긴 변 400 정사각형) + Label.txt·Label_auto.txt (PPOCRLabel 형식)
#               상자 = 고른 방향의 ft1 상자를 돌리기 전 400×400 좌표로 되돌린 네 꼭짓점, 글자 = 상자별 PARSeq 읽기
#             → exp_cns\results\parseq_label_v1\review_list.csv  (번호·파일명·정답 각인·읽은 글자·원인·고른 방향)
#  [--step crops]  PARSeq 학습용 글자 조각 → exp_cns\results\parseq_label_v1\crops\<train|val>\*.png + train.csv · val.csv
#             나누기 = attr_cnn_v1 labels.csv 의 part (약 단위 학습 258종 / 검증 65종 — 약 판별 w 고르기와 같은 검증 약)
#             조각 출처 (source 칸):
#               auto_ok    맞힌 사진: 고른 방향 상자 + 상자별 PARSeq 글자 (PARSeq 가 본 조각 그대로 = 돌린 사진에서 상자 잘라 냄)
#               auto_order 순서만 다름: 위와 같음, 상자 글자가 정답 안에 통째로 있는 것만
#               auto_other 다른 방향에 정답: 그 방향의 상자(boxes.json)를 사전학습 PARSeq 로 다시 읽어 정답 안에 통째로 있는 것만
#               human      사람 검수 review\Label_fixed.txt (Label.txt + 빠진 글자 채움, label_fixes.csv) — 네 꼭짓점을 펴서 자르고,
#                          0/90/180/270° 중 사전학습 PARSeq 읽기가 사람 글자와 가장 가까운 방향으로 세움 (같으면 확신도 높은 쪽)
#             글자: 자동은 norm 된 대문자, 사람은 적은 그대로 (소문자·'-'·'.' 포함 — PARSeq 글자표에 있음, 채점은 norm 으로 같게)
#             확인 그림 crops_check.png (출처별 무작위 조각 + 글자, 시드 42)
#  [실행]  python exp_cns\parseq_labels_cns.py --step review
#          python exp_cns\parseq_labels_cns.py --step crops
# =====================================================================

import argparse
import collections
import csv
import json
import os
import random

import cv2
import numpy as np

import common_cns as cc
import ocr_common_cns as oc
from make_det_labels_cns import unrotate_box

PRED = oc.OCR_OUT / "trainall" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"
OUT = cc.RESULTS / "parseq_label_v1"
PER_DRUG = 3
BIG_FRAC = 0.5


def cause(r):
    """틀린 사진 한 장 → 원인 (위 설명 순서)."""
    t, rd = r["truth"], r["read"]
    reads = [d.split(":")[1] for d in r["all_dirs"].split(" | ")]
    boxes = json.loads(r["best_boxes"] or "[]")
    if sorted(rd) == sorted(t):
        return "order"
    if t in reads:
        return "other"
    if not boxes:
        return "nobox"
    if any((b[2] - b[0]) * (b[3] - b[1]) >= BIG_FRAC * 400 * 400 for b in boxes):
        return "big"
    if sum(c in rd for c in t) >= len(t) * 0.5:
        return "part"
    return "wrong"


def step_review():
    rows = list(csv.DictReader(open(PRED, encoding="utf-8-sig")))
    done = {r["file_name"] for r in csv.DictReader(open(oc.TRAIN500_CSV, encoding="utf-8-sig"))}
    pool = collections.defaultdict(list)
    cnt = collections.Counter()
    for r in rows:
        if r["correct"] == "1":
            continue
        r["cause"] = cause(r)
        cnt[r["cause"]] += 1
        if r["cause"] in ("nobox", "big", "part", "wrong") and r["file_name"] not in done:
            pool[r["drug_N"]].append(r)
    print("틀린 사진 원인:", dict(cnt))
    rng = random.Random(42)
    pick = []
    for d in sorted(pool):
        groups = collections.defaultdict(list)                       # (카메라, 면) 묶음마다 섞은 뒤 돌아가며 하나씩
        for r in sorted(pool[d], key=lambda r: r["file_name"]):
            groups[(r["camera_la"], r["face"])].append(r)
        keys = sorted(groups)
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(groups[k])
        while len([p for p in pick if p["drug_N"] == d]) < PER_DRUG and any(groups[k] for k in keys):
            for k in keys:
                if groups[k] and len([p for p in pick if p["drug_N"] == d]) < PER_DRUG:
                    pick.append(groups[k].pop())
    items = {it["file_name"]: it for it in oc.load_split("trainall", 0)}
    d = OUT / "review"
    os.makedirs(d, exist_ok=True)
    lines = []
    for r in pick:
        fn = r["file_name"]
        cc.imwrite(str(d / fn), cc.fit_side(cc.imread(items[fn]["image_path"]), 400))   # 검출기가 본 사진과 같은 좌표
        ang = float(r["angle"])
        ann = [{"transcription": b[4], "points": unrotate_box(b[:4], ang), "difficult": False}
               for b in json.loads(r["best_boxes"] or "[]")]
        lines.append(f"review/{fn}\t{json.dumps(ann, ensure_ascii=False)}")
    for name in ("Label.txt", "Label_auto.txt"):                     # Label_auto.txt = PPOCRLabel 이 덮어써도 남는 원본
        with open(d / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(OUT / "review_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "file_name", "drug_N", "truth", "read", "cause", "angle", "camera_la", "face"])
        for i, r in enumerate(pick, 1):
            w.writerow([i, r["file_name"], r["drug_N"], r["truth"], r["read"], r["cause"], r["angle"], r["camera_la"], r["face"]])
    print(f"저장: {d} ({len(pick)}장, {len(pool)}종)  원인 {dict(collections.Counter(r['cause'] for r in pick))}")
    print(f"  카메라 {dict(collections.Counter(r['camera_la'] for r in pick))}, 면 {dict(collections.Counter(r['face'] for r in pick))}")
    print(f"  뺀 사진(이미 상자 본 500장 안): {sum(1 for r in rows if r.get('cause') in ('nobox', 'big', 'part', 'wrong') and r['file_name'] in done)}")


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def warp_quad(img, pts):
    """네 꼭짓점(첫 점 → 둘째 점 = 글자 윗변 방향) → 펴서 자른 조각."""
    p = np.asarray(pts, np.float32)
    if len(p) != 4:                                                  # 꼭짓점이 4개가 아닌 다각형 → 감싸는 최소 사각형 (방향은 뒤에서 네 방향 읽기로 정함)
        p = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    w = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
    h = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(p, dst), (max(w, 1), max(h, 1)),
                               borderMode=cv2.BORDER_REPLICATE)


def step_crops():
    import ocr_cns                                                   # torch·PARSeq (이 단계에서만)
    reader = ocr_cns.Reader()
    norm = ocr_cns.base.norm
    part_of = {r["file_name"]: r["part"] for r in csv.DictReader(open(cc.RESULTS / "attr_cnn_v1" / "labels.csv", encoding="utf-8-sig"))}
    items = {it["file_name"]: it for it in oc.load_split("trainall", 0)}
    rows = list(csv.DictReader(open(PRED, encoding="utf-8-sig")))
    saved = json.load(open(PRED.parent.parent / "orig_alignlong" / "boxes.json", encoding="utf-8"))
    out = []                                                         # (조각, 글자, 출처, 행)
    cnt = collections.Counter()

    def add(crop, text, src, r):
        if crop.size and min(crop.shape[:2]) >= 4 and text:
            out.append((crop, text, src, r))
            cnt[src] += 1
        else:
            cnt[src + "_skip"] += 1

    for k, r in enumerate(rows, 1):
        c = "ok" if r["correct"] == "1" else cause(r)
        if c not in ("ok", "order", "other"):
            continue
        img = cc.fit_side(cc.imread(items[r["file_name"]]["image_path"]), 400)   # 검출기·PARSeq 가 본 사진 (orig)
        if c in ("ok", "order"):
            rot = oc.rotate_color(img, float(r["angle"]) if "." in r["angle"] else int(r["angle"]))
            for x0, y0, x1, y1, t in json.loads(r["best_boxes"] or "[]"):
                if t and (c == "ok" or t in r["truth"]):
                    add(rot[y0:y1, x0:x1], t, "auto_" + c, r)
                elif t:
                    cnt["auto_order_skip"] += 1
        else:
            ang = next(d.split(":")[0] for d in r["all_dirs"].split(" | ") if d.split(":")[1] == r["truth"])
            rot = oc.rotate_color(img, float(ang) if "." in ang else int(ang))
            for x0, y0, x1, y1 in saved[r["file_name"]][ang]["boxes"]:
                crop = rot[y0:y1, x0:x1]
                t = norm(reader.read(crop)[0]) if crop.size else ""
                if t and t in r["truth"]:
                    add(crop, t, "auto_other", r)
                else:
                    cnt["auto_other_skip"] += 1
        if k % 500 == 0:
            print(f"  {k}/{len(rows)}  {dict(cnt)}")

    lst = {r["file_name"]: r for r in rows}
    n_turn = collections.Counter()
    for line in open(OUT / "review" / "Label_fixed.txt", encoding="utf-8"):
        key, v = line.rstrip("\n").split("\t", 1)
        fn = key.split("/")[-1]
        img = cc.imread(str(OUT / "review" / fn))
        for b in json.loads(v):
            t = b["transcription"]
            base_crop = warp_quad(img, b["points"])
            best = None
            for turn, code in ((0, None), (90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
                crop = base_crop if code is None else cv2.rotate(base_crop, code)
                rd, conf = reader.read(crop)
                key2 = (edit_distance(norm(rd), norm(t)), -conf)
                if best is None or key2 < best[0]:
                    best = (key2, turn, crop)
            n_turn[best[1]] += 1
            add(best[2], t, "human", lst[fn])
    print("사람 조각 세운 방향:", dict(n_turn))

    d = OUT / "crops"
    for p in ("train", "val"):
        os.makedirs(d / p, exist_ok=True)
    recs = {"train": [], "val": []}
    seq = collections.Counter()
    for crop, t, src, r in out:
        p = part_of[r["file_name"]]
        stem = r["file_name"][:-4]
        seq[stem] += 1
        name = f"{p}/{stem}_{src}_{seq[stem]}.png"
        cc.imwrite(str(d / name), crop)
        recs[p].append(dict(path=name, label=t, source=src, file_name=r["file_name"], drug_N=r["drug_N"]))
    for p, rr in recs.items():
        with open(d / f"{p}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
        print(f"{p}: 조각 {len(rr)}개 / 사진 {len({x['file_name'] for x in rr})}장 / 약 {len({x['drug_N'] for x in rr})}종  "
              f"출처 {dict(collections.Counter(x['source'] for x in rr))}")
    print("건너뜀:", {k: v for k, v in cnt.items() if k.endswith("_skip")})
    check_sheet(d, recs)


def check_sheet(d, recs, per=10):
    """출처마다 무작위 per 개 (시드 42) → crops_check.png  (한 줄 = 출처 하나, 조각 높이 64 + 글자)."""
    rng = random.Random(42)
    allr = recs["train"] + recs["val"]
    lines = []
    for src in ("auto_ok", "auto_order", "auto_other", "human"):
        pool = [x for x in allr if x["source"] == src]
        pick = rng.sample(pool, min(per, len(pool)))
        tiles = []
        for x in pick:
            im = cc.imread(str(d / x["path"]))
            im = cv2.resize(im, (min(200, max(20, int(im.shape[1] * 64 / im.shape[0]))), 64))
            tile = np.full((96, 210, 3), 255, np.uint8)
            tile[:64, :im.shape[1]] = im
            cv2.putText(tile, x["label"][:14], (2, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            tiles.append(tile)
        tiles += [np.full((96, 210, 3), 255, np.uint8)] * (per - len(tiles))
        head = np.full((24, 210 * per, 3), 230, np.uint8)
        cv2.putText(head, src, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        lines += [head, np.hstack(tiles)]
    cc.imwrite(str(OUT / "crops_check.png"), np.vstack(lines))
    print("저장:", OUT / "crops_check.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["review", "crops"])
    a = ap.parse_args()
    {"review": step_review, "crops": step_crops}[a.step]()


if __name__ == "__main__":
    main()
