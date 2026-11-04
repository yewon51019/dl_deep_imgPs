# =====================================================================
#  tps_cns.py  —  사람이 그린 다각형으로 TPS 펴기 학습 (휜 각인 읽기, 사후 보조 실험)
# =====================================================================
#
#  [목적]  휜 각인 (K-011307 SINGULAIR, 아리셉트에비스정 ARICEPT 등) 을 펴서 PARSeq 로 읽어 본다.
#          약 판별 방식을 고르는 실험이 아니라 읽기 정답률만 보는 사후 보조 지표.
#  [이 파일의 단계]
#     --step polycheck : 다각형을 위·아래 테두리로 나누고 펴 본 확인 그림 (학습 전에 사람이 확인)
#     --step example   : K-000576 한 장으로 학습·읽기 방식을 보여 주는 설명 그림
#     --step polar     : 방법 A, 학습 없는 극좌표 펴기로 시험 대상 36장 읽기 (기준 = base 읽기)
#  [다각형 라벨]  det_label_v1\fix\Label.txt, det_label_v3\fix2\Label.txt (사람이 PPOCRLabel 로 고친 것, 400×400 좌표)
#     쓰는 것 = 점 8개 이상 + 글자 2개 이상 + 시험 대상 약 아님.
#     점 5개는 대부분 네모에 가까워서 뺌. 시험 대상 K-011307 은 학습에 쓰면 시험이 공정하지 않아서 뺌.
#  [위·아래 나누기]  다각형의 변 두 개를 '양 끝 변' 으로 고르면 둘레가 두 줄로 나뉜다.
#     두 줄을 같은 간격으로 찍어 마주 보는 점 사이 거리(= 띠 두께)의 평균이 가장 작은 조합을 고른다 (긴 쪽을 가로지르면 커짐).
#  [글자 방향]  띠를 정방향·180° 두 가지로 펴서 PARSeq 로 읽고, 라벨 글자와 더 비슷한 쪽.
#     PARSeq 는 뒤집힌 글자도 읽으므로 틀릴 수 있다 → 비슷하면 flag 칸에 표시하고 그림으로 사람이 확인.
#  [저장]  exp_cns\results\tps_v1\  poly_check2.png (사진이 들어 있어 .gitignore), poly_check2.csv
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\tps_cns.py --step polycheck      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import argparse
import csv
import json
import os

import torch  # noqa: F401  (다른 패키지보다 먼저)
import cv2
import numpy as np

import common_cns as cc
import ocr_cns

base = ocr_cns.base                              # run_exp300: norm, PARSeq

OUT = cc.RESULTS / "tps_v1"
LABELS = [cc.RESULTS / "det_label_v1" / "fix" / "Label.txt",
          cc.RESULTS / "det_label_v3" / "fix2" / "Label.txt"]
TEST_DRUGS = {"K-011307", "K-023336", "K-023337"}   # 시험 대상 약: 학습 라벨에서 뺌
USER_DROP = {"K-016386"}                         # 사용자가 확인 그림 보고 뺌 (poly_check2 #25 #26, VAS, 펴도 안 읽힘)
MIN_POINTS = 8
MIN_CHARS = 2
# 라벨 글자 고치기: (사진, 원래 글자) → 고친 글자. 오타 1개, 글자를 안 채운 자리표시 1개
TEXT_FIX = {("K-001363_0_0_1_2_75_180_200.png", "TAGANET"): "TAGAMET",
            ("K-001362_0_0_1_2_75_180_200.png", "TEMPORARY"): "TAGAMET",
            ("K-015591_0_2_0_2_90_240_200.png", "XDP"): "ZDP"}       # 사진과 DB 뒷면 각인(ZDP5)이 ZDP
MIN_LEN_RATIO = 0.35                             # 위·아래 테두리 길이 비율 하한
N_FID = 10                                       # 한쪽 테두리 기준점 수 (exp300 TPS 와 같음, 위 10 + 아래 10)
STRIP_HW = (32, 128)                             # 펴진 띠 크기 (PARSeq 입력과 같음)


def load_polys():
    """조건에 맞는 다각형 목록. 사진 경로·글자·꼭짓점."""
    polys = []
    for path in LABELS:
        for line in open(path, encoding="utf-8"):
            if "\t" not in line:
                continue
            rel, js = line.rstrip("\n").split("\t", 1)
            fn = rel.replace("\\", "/").split("/")[-1]
            for b in json.loads(js):
                pts = np.array(b["points"], np.float32)
                text = TEXT_FIX.get((fn, b["transcription"]), b["transcription"])
                text = base.norm(text)
                if len(pts) < MIN_POINTS or len(text) < MIN_CHARS or fn[:8] in TEST_DRUGS or fn[:8] in USER_DROP:
                    continue
                polys.append({"file": fn, "path": path.parent / fn, "text": text, "pts": pts})
    return polys


def resample(chain, n):
    """꺾은선을 같은 길이 간격의 점 n 개로."""
    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, chain[:, k]) for k in range(2)], 1)


def split_band(pts):
    """다각형 → 두 테두리 (A: 왼→오, B: 왼→오, A 를 위로 보면 B 가 아래). 두께 평균이 가장 작은 양 끝 변을 고름."""
    # 신발끈 넓이 > 0 이면 (y 가 아래로 커지는 그림 좌표에서) 시계 방향. 시계 방향으로 맞춰 두면 뒤집힌 거울상이 안 생긴다
    x, y = pts[:, 0], pts[:, 1]
    if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() < 0:
        pts = pts[::-1]
    n = len(pts)
    best = None
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:                # 이웃한 변은 건너뜀
                continue
            # 변 i (i→i+1) 와 변 j (j→j+1) 를 양 끝으로: 줄 A = i+1..j, 줄 B = j+1..i
            a = pts[i + 1:j + 1]
            b = np.concatenate([pts[j + 1:], pts[:i + 1]])
            if len(a) < 2 or len(b) < 2:
                continue
            # 두 줄 길이가 너무 다르면 건너뜀: 안쪽 테두리는 바깥보다 짧지만, 한쪽이 아주 짧으면 끝을 잘못 고른 것 (1차 그림 #7 #9 #14 #26)
            la, lb = [np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in (a, b)]
            if min(la, lb) / max(la, lb) < MIN_LEN_RATIO:
                continue
            ra, rb = resample(a, N_FID), resample(b[::-1], N_FID)
            width = np.linalg.norm(ra - rb, axis=1)
            if best is None or width.mean() < best[0]:
                best = (width.mean(), ra, rb)
    return best[1], best[2]


def flatten(img, top, bot):
    """위·아래 테두리 사이를 곧은 띠로 펴기 (마주 보는 점을 이은 선을 세로줄로)."""
    h, w = STRIP_HW
    tt, bb = resample(top, w), resample(bot, w)
    v = np.linspace(0, 1, h)[:, None, None]
    grid = (tt[None] * (1 - v) + bb[None] * v).astype(np.float32)
    return cv2.remap(img, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def similarity(a, b):
    """편집거리 유사도 (L−ed)/(L+ed), 이 저장소의 매칭과 같은 식."""
    a, b = base.norm(a), base.norm(b)
    if not a or not b:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    L = max(len(a), len(b))
    return (L - d[-1]) / (L + d[-1])


def polycheck():
    OUT.mkdir(parents=True, exist_ok=True)
    png, table = OUT / "poly_check2.png", OUT / "poly_check2.csv"     # 1차(poly_check) 는 나누기 고치기 전 기록으로 남김
    if png.exists() or table.exists():
        raise SystemExit(f"이미 있음: {png}")
    polys = load_polys()
    print(f"다각형 {len(polys)}개 / 사진 {len({p['file'] for p in polys})}장 / 약 {len({p['file'][:8] for p in polys})}종")
    reader = base.PARSeq()
    rows, cells = [], []
    for k, p in enumerate(polys, 1):
        img = cc.imread(str(p["path"]))
        a, b = split_band(p["pts"])
        # 정방향 = (A 위, B 아래), 180° = 띠를 통째로 돌린 것 (B 를 뒤집어 위로, A 를 뒤집어 아래로)
        opts = [(a, b), (b[::-1], a[::-1])]
        strips = [flatten(img, t, u) for t, u in opts]
        reads = [reader.read_crop(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)) for s in strips]
        sims = [similarity(r[0], p["text"]) for r in reads]
        pick = int(sims[1] > sims[0] or (sims[1] == sims[0] and reads[1][1] > reads[0][1]))
        top, bot = opts[pick]
        rows.append({"no": k, "file": p["file"], "text": p["text"], "points": len(p["pts"]),
                     "read_pick": reads[pick][0], "sim_pick": round(sims[pick], 3), "conf_pick": round(reads[pick][1], 3),
                     "read_other": reads[1 - pick][0], "sim_other": round(sims[1 - pick], 3),
                     "conf_other": round(reads[1 - pick][1], 3),
                     "flag": "direction_unsure" if abs(sims[0] - sims[1]) < 0.1 else ""})
        # 칸 하나: 왼쪽 = 사진 + 위 테두리(초록)·아래 테두리(빨강)·글자 시작점(노랑), 오른쪽 = 고른 띠 / 반대 띠
        vis = img.copy()
        cv2.polylines(vis, [top.astype(np.int32)], False, (0, 220, 0), 2)
        cv2.polylines(vis, [bot.astype(np.int32)], False, (0, 0, 255), 2)
        cv2.circle(vis, tuple(int(v) for v in top[0]), 6, (0, 255, 255), -1)
        cell = np.full((300, 620, 3), 255, np.uint8)
        cell[:300, :300] = cv2.resize(vis, (300, 300))
        big = [cv2.resize(strips[pick], (304, 76)), cv2.resize(strips[1 - pick], (304, 76))]
        cell[40:116, 308:612] = big[0]
        cell[170:246, 308:612] = big[1]
        put = lambda s, y, c=(0, 0, 0): cv2.putText(cell, s, (308, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        put(f"#{k} {p['file'][:8]} truth {p['text']}", 18)
        put(f"pick read {reads[pick][0]}", 34, (0, 120, 0))
        put(f"other read {reads[1 - pick][0]}", 164, (120, 0, 0))
        put(rows[-1]["flag"], 266, (0, 0, 255))
        cells.append(cell)
    # 4칸씩 줄 맞춤
    per = 4
    while len(cells) % per:
        cells.append(np.full((300, 620, 3), 255, np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i + per]) for i in range(0, len(cells), per)])
    cc.imwrite(str(png), sheet)
    with open(table, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["no"], r["file"], r["text"], "| 고름", r["read_pick"], r["sim_pick"], "| 반대", r["read_other"], r["sim_other"], r["flag"])
    print(f"저장: {png} {sheet.shape}")


def rotate_up(img, pts):
    """다각형 중심이 알약 맨 위(12시)로 오도록 사진 중심 기준으로 돌림. 돌린 사진, 돌린 꼭짓점, 각도."""
    h, w = img.shape[:2]
    c = pts.mean(0) - [w / 2, h / 2]
    ang = np.degrees(np.arctan2(c[0], -c[1]))        # 12시에서 시계 방향으로 잰 위치 → 이만큼 반시계로 돌리면 12시
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rot, cv2.transform(pts[None], M)[0], ang


def band_outer_top(pts, center):
    """돌린 다각형 → 바깥 테두리를 위, 안쪽을 아래로 (둘 다 왼→오). 글자 방향은 보지 않고 모양만 본 띠."""
    a, b = split_band(pts)
    dist = lambda c: np.linalg.norm(c - center, axis=1).mean()
    top, bot = (a, b) if dist(a) > dist(b) else (b, a)
    fix = lambda c: c if c[0, 0] < c[-1, 0] else c[::-1]
    return fix(top), fix(bot)


def example():
    """설명용 그림 (학습 없음, 사람이 그린 다각형 사용): K-000576 한 장으로 학습 예제 2개와 12방향 읽기를 보여 줌."""
    fn = "K-000576_0_0_0_2_75_060_200.png"
    png = OUT / "example_k576.png"
    if png.exists():
        raise SystemExit(f"이미 있음: {png}")
    polys = [p for p in load_polys() if p["file"] == fn]
    img = cc.imread(str(polys[0]["path"]))
    center = np.array([200.0, 200.0])
    reader = base.PARSeq()
    read = lambda s: reader.read_crop(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY))[0]

    def label(canvas, s, x, y, c=(0, 0, 0), size=0.6):
        cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, c, 1, cv2.LINE_AA)

    rows, angs = [], {}
    for p in polys:
        rot, pts, ang = rotate_up(img, p["pts"])
        angs[p["text"]] = ang
        top, bot = band_outer_top(pts, center)
        strip = flatten(rot, top, bot)
        flip = cv2.rotate(strip, cv2.ROTATE_180)
        vis_src = img.copy()
        cv2.polylines(vis_src, [p["pts"].astype(np.int32)], True, (0, 220, 0), 2)
        vis = rot.copy()
        cv2.polylines(vis, [top.astype(np.int32)], False, (0, 220, 0), 3)
        cv2.polylines(vis, [bot.astype(np.int32)], False, (0, 0, 255), 3)
        row = np.full((340, 1500, 3), 255, np.uint8)
        row[40:340, 0:300] = cv2.resize(vis_src, (300, 300))
        row[40:340, 340:640] = cv2.resize(vis, (300, 300))
        row[60:176, 680:1144] = cv2.resize(strip, (464, 116))
        row[210:326, 680:1144] = cv2.resize(flip, (464, 116))
        label(row, f"photo: band = {p['text']}", 0, 28)
        label(row, "1. rotate band to top", 340, 28)
        label(row, "2. model target: unbend (green = top)", 680, 28)
        label(row, f"PARSeq reads: {read(strip)}", 1160, 120, (0, 120, 0))
        label(row, "3. also read turned 180", 680, 203)
        label(row, f"PARSeq reads: {read(flip)}", 1160, 270, (0, 120, 0))
        rows.append(row)
    # 12방향: 어느 방향에서 어느 띠가 맨 위에 오는지 (읽을 때 모델이 볼 사진)
    thumbs = []
    for k in range(12):
        a = k * 30
        M = cv2.getRotationMatrix2D((200, 200), a, 1.0)
        rot = cv2.warpAffine(img, M, (400, 400), borderMode=cv2.BORDER_REPLICATE)
        t = cv2.resize(rot, (125, 125))
        cell = np.full((165, 125, 3), 255, np.uint8)
        cell[:125] = t
        cv2.rectangle(cell, (40, 2), (85, 30), (255, 255, 255), 1)    # 맨 위 자리 표시
        hit = [txt for txt, v in angs.items() if min((a - v) % 360, (v - a) % 360) <= 15]
        label(cell, f"{a} deg", 2, 142, size=0.45)
        label(cell, hit[0] if hit else "-", 2, 160, (0, 120, 0) if hit else (150, 150, 150), 0.45)
        thumbs.append(cell)
    strip12 = np.full((215, 1500, 3), 255, np.uint8)
    label(strip12, "4. reading: try 12 directions, model unbends whatever is at the top, keep all reads as candidates", 0, 28)
    strip12[45:210, 0:1500] = np.hstack(thumbs)
    sheet = np.vstack(rows + [strip12])
    cc.imwrite(str(png), sheet)
    print(f"저장: {png} {sheet.shape}", {k: round(v, 1) for k, v in angs.items()})


def polar_targets():
    """시험 대상 36장: 싱귤레어 SINGULAIR 면 12장 (이미 잘린 사진) + 아리셉트에비스정 ARICEPT 면 24장 (crop_v2 로 자른 사진)."""
    out = []
    labels = {r["file_name"]: r["image_path"] for r in csv.DictReader(open(cc.RESULTS / "attr_cnn_v1" / "labels.csv", encoding="utf-8-sig"))}
    for r in csv.DictReader(open(cc.RESULTS / "match_v3" / "base" / "pred_val.csv", encoding="utf-8-sig")):
        if r["drug_N"] == "K-011307" and r["truth"] == "SINGULAIR":
            out.append({"file": r["file_name"], "drug": r["drug_N"], "truth": r["truth"], "base_read": r["read"], "path": labels[r["file_name"]]})
    crops = cc.RESULTS / "crop_v2" / "exp300test" / "images"
    for r in csv.DictReader(open(cc.RESULTS / "test_final_exp300" / "pred_exp300test.csv", encoding="utf-8-sig")):
        if r["drug_N"] in ("K-023336", "K-023337") and r["truth"] == "ARICEPT":
            out.append({"file": r["file_name"], "drug": r["drug_N"], "truth": r["truth"], "base_read": r["read"], "path": str(crops / r["file_name"])})
    return out


def polar_candidates(bgr, reader):
    """사진 한 장의 극좌표 후보 → ([(글자, 확신도, 창 이름)], 온바퀴 띠).
       띠 = exp300 P2 (crop_pill → 타원을 원으로 → polar_strip). 후보 = 온바퀴·1.5바퀴 + 각도 창 120/180/240/300°, 창 시작 30°씩, 각각 그대로·180°."""
    import curved_exp300 as old                      # exp300 의 ellipse_to_circle, align_exp300.polar_strip 재사용
    g, *_ = base.crop_pill(bgr)
    S, _ = old.al.polar_strip(old.ellipse_to_circle(g))
    n = S.shape[1]
    tile = np.ascontiguousarray(np.hstack([S, S]))
    wins = [("full", S), ("full1.5", tile[:, :n + n // 2])]
    for deg in (120, 180, 240, 300):
        w = n * deg // 360
        for a0 in range(0, 360, 30):
            x = n * a0 // 360
            wins.append((f"w{deg}_a{a0}", tile[:, x:x + w]))
    cands = []
    for name, win in wins:
        for flip in (0, 1):
            im = np.ascontiguousarray(win[::-1, ::-1]) if flip else win
            text, conf = reader.read_crop(im)
            cands.append((base.norm(text), conf, name + ("_r" if flip else "")))
    return cands, S


def polar():
    """방법 A: 학습 없는 극좌표 펴기로 시험 대상 36장 읽기. 기준 = base 의 읽기 (저장된 값).
       한 답 고르기 = 확신도 최고 (정답을 안 씀). '후보 안에 정답' 은 정답을 보고 찾은 상한 (실제 답이 아님)."""
    OUT.mkdir(parents=True, exist_ok=True)
    table, summary, png = OUT / "polar_reads.csv", OUT / "polar_summary.csv", OUT / "polar_sheet.png"
    if table.exists() or summary.exists() or png.exists():
        raise SystemExit(f"이미 있음: {table}")
    targets = polar_targets()
    print(f"대상 {len(targets)}장")
    reader = base.PARSeq()
    rows, cells = [], []
    for t in targets:
        cands, S = polar_candidates(cc.imread(t["path"]), reader)
        truth = t["truth"]
        pick = max(cands, key=lambda c: c[1])
        best = max(cands, key=lambda c: similarity(c[0], truth))
        hits = [c for c in cands if c[0] == truth]
        rows.append({"file": t["file"], "drug": t["drug"], "truth": truth, "base_read": t["base_read"],
                     "base_ok": int(base.norm(t["base_read"]) == truth), "base_sim": round(similarity(t["base_read"], truth), 3),
                     "pick_read": pick[0], "pick_conf": round(pick[1], 3), "pick_win": pick[2], "pick_ok": int(pick[0] == truth),
                     "pick_sim": round(similarity(pick[0], truth), 3), "best_read": best[0], "best_win": best[2],
                     "best_sim": round(similarity(best[0], truth), 3), "n_cands": len(cands), "hit": int(bool(hits)), "n_hit": len(hits)})
        # 칸: 온바퀴 띠 + 읽은 글자 (그림 글자는 영문만)
        cell = np.full((110, 1000, 3), 255, np.uint8)
        cell[30:90, 0:1000] = cv2.cvtColor(cv2.resize(S, (1000, 60)), cv2.COLOR_GRAY2BGR)
        r = rows[-1]
        cv2.putText(cell, f"{t['file'][:8]}_{t['file'][-14:-8]} truth {truth} | base {base.norm(t['base_read'])} | pick {pick[0]} | best {best[0]} {'HIT' if hits else ''}",
                    (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(cell)
    if len(cells) % 2:
        cells.append(np.full((110, 1000, 3), 255, np.uint8))
    cc.imwrite(str(png), np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]))
    with open(table, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = [("전체", rows), ("SINGULAIR 12", [r for r in rows if r["drug"] == "K-011307"]), ("ARICEPT 24", [r for r in rows if r["truth"] == "ARICEPT"])]
    summ = []
    for name, rs in groups:
        n = len(rs)
        summ.append({"group": name, "n": n, "base_exact": sum(r["base_ok"] for r in rs), "base_sim_mean": round(sum(r["base_sim"] for r in rs) / n, 3),
                     "polar_pick_exact": sum(r["pick_ok"] for r in rs), "polar_pick_sim_mean": round(sum(r["pick_sim"] for r in rs) / n, 3),
                     "polar_any_cand_exact": sum(r["hit"] for r in rs), "polar_best_sim_mean": round(sum(r["best_sim"] for r in rs) / n, 3)})
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)
    for r in rows:
        print(r["file"][:8], r["file"][-14:-8], "| base", r["base_read"], "| pick", r["pick_read"], r["pick_conf"], r["pick_win"], "| best", r["best_read"], r["best_win"], "| hit", r["n_hit"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["polycheck", "example", "polar"])
    a = ap.parse_args()
    if a.step == "polar":
        polar()
    if a.step == "polycheck":
        polycheck()
    elif a.step == "example":
        example()


if __name__ == "__main__":
    main()
