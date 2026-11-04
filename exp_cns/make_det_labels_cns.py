# =====================================================================
#  make_det_labels_cns.py  —  검출기 파인튜닝용 자동 박스 라벨 (다음 할 일 2-1)
# =====================================================================
#
#  [왜]  사전학습 PaddleOCR 검출기는 알약에 얕게 파인 각인을 글자로 잘 못 봄 (dbpp_diag: 확률 지도가 거의 0).
#        우리 알약 사진에 '글자는 여기' 상자를 그려 검출기를 더 학습시키려는데, 사람이 처음부터 다 그리면 힘드므로
#        지금 최고 방식(개발용 61.1%)으로 상자를 먼저 자동으로 만들고, 맞게 읽힌 사진은 '자동 통과', 나머지만 사람이 고침.
#  [단계]
#    --step select : 학습용(train)에서 500장 고르기 → det_label_v1\train500.csv
#                    - Λ(람다) 각인 약 5종 제외 (정신과 약에 없음), 개발용 36종·평가용은 애초에 안 들어감
#                    - 323종 모두 1장씩 + 나머지 177장은 다른 약에서 1장씩 더 (시드 42, 언제 돌려도 같은 500장)
#                    - 카메라 75°/90° 가 반반 섞이게: 약 순서대로 75·90 을 번갈아 고르고, 두 번째 사진은 첫 사진과 반대 카메라
#    (그다음 아래 3개 명령으로 상자·읽기 — 개발용과 같은 방식)
#      .\.venv_paddle_gpu\Scripts\python.exe exp_cns\paddle_boxes_cns.py --split train500 --prep orig --align long
#      .\.venv_paddle_gpu\Scripts\python.exe exp_cns\paddle_boxes_cns.py --split train500 --prep <K0L17R6 이름> --align long
#      .\.venv\Scripts\python.exe exp_cns\ocr_cns.py --split train500 --det paddle_t0 --prep orig --align long --join line
#          --suffix _line --retry-box-prep <K0L17R6 이름>
#    --step labels : 위 pred.csv → 자동 통과/확인 필요 판정 + PPOCRLabel 라벨
#                    - 상자는 '고른 방향으로 돌린 사진' 좌표 → 돌리기 전 400×400 사진 좌표의 네 꼭짓점으로 되돌림
#                      (ocr_common_cns.rotate_color 의 돌리기 행렬을 거꾸로 적용)
#                    - 자동 통과 = 읽은 글자가 정답과 완전히 같고, 상자가 1개 이상이고, '큰 상자'가 없음
#                      큰 상자 = 넓이가 400×400 사진의 절반 이상 (알약 전체를 덮은 상자. 우연히 맞혀도 학습에 쓰면
#                      검출기가 '알약 전체 = 글자'로 배우므로 확인 필요로 돌림)
#    --step try    : PPOCRLabel 시험용 5장 폴더 (자동 통과 2 + 잘못 읽음 1 + 박스 0개 1 + 큰 상자 1) → det_label_v1\try5\
#                    PPOCRLabel 은 '<폴더 이름>/<파일명>' 을 열쇠로 쓰므로 Label.txt 의 'images/' 를 'try5/' 로 바꿔 넣음
#    --step sheets : Claude 1차 분류용 그림 (다음 할 일 2-3). '확인 필요' 사진만, 한 그림에 5장 → det_label_v1\review\sheet_NNN.png
#                    한 줄 = [원본 + 자동 상자(초록) + 상자별 글자 | 같은 사진 대비 강화(CLAHE, 상자 없음 — 흐린 각인을 보려고)]
#                    줄 위 글씨: 번호, 파일명, 정답, 읽은 글자, 자동 판정 이유. 분류 결과는 Claude 가 claude_review.csv 로 씀
#    --step fixlist: 그림 없이 규칙으로 '확인 필요' 205장 분류 → fix_list.csv + 고칠 사진만 담은 PPOCRLabel 폴더 fix\
#                    (Claude 그림 분류는 2장 시험에서 사용량 +11%p 라 중단. claude_review.csv 에 있는 사진은 그 분류를 씀)
#                    규칙 (위에서부터 먼저 맞는 것):
#                      E 큰 상자                → 고칠 필요 (시험 5/5 고쳐야 했음)
#                      C 상자 0개               → 고칠 필요
#                      A 순서만 다름            → 고칠 필요 없음 (읽은 글자와 정답의 글자 구성이 같음, 예: 250CKD / CKD250)
#                      B 정답의 일부만 읽음      → 고칠 필요 (빠진 상자 추가 추정)
#                      G 정답과 거의 안 겹침     → 고칠 필요, 라벨 앞뒤 뒤바뀜 의심 (사진에 보이는 글자로 상자·글자)
#                      X 그 밖                  → 고칠 필요 (사람이 보고 판단)
#                    fix\ 의 Label.txt 는 자동 상자를 미리 채움 (열쇠 'fix/<파일명>'). 순서: 우선순위 → 파일명
#    --step spot   : 자동 통과 295장 중 20장 무작위 → 사람이 훑어보는 폴더 spot20\ (파인튜닝 1차 전 확인)
#                    - 카메라(75/90) × 면(앞/뒤) 칸마다 자동 통과 안의 비율대로 장수를 나눔 (나머지가 큰 칸부터 1장씩)
#                    - 칸 안에서는 시드 42 로 섞어 앞에서부터, 이미 뽑은 약은 건너뜀 (약 하나에 쏠리지 않게)
#                    - spot20_list.csv 에 사람이 채울 칸: ok(1/0), problem(넉넉/겹침/빠짐/기타), note
#    --step labels2: v1 상자가 넉넉한 문제(spot20 20장 모두) 고친 v2 라벨 → det_label_v2\images\Label.txt + compare\spot20_NN.png
#                    먼저 .\.venv_paddle_gpu\Scripts\python.exe exp_cns\det_polys_cns.py 로 Paddle 원래 네 꼭짓점을 구해 둠
#    --step shrink : v2 도 넉넉함(spot20 사람 확인) → 두께 0.5/0.6/0.7 × 길이 0.85 로 줄인 비교 그림 det_label_v2\shrink\
#                    (DBNet 넓히기를 거꾸로 하면 글자보다 작은 가운데 띠가 되므로 안 씀, 고정 비율로 줄이고 그림 보고 고름)
#    --step shrink_w: 두께 0.7 고정, 길이 0.85/0.9/1.0 비교 그림 det_label_v2\shrink_w\ (길이를 줄이면 첫·끝 글자가 잘릴 수 있음)
#    --step labels3 --h <두께> --w <길이> : 고른 비율로 줄인 v3 라벨 → det_label_v3\images\Label.txt
#    --step review3: 자동 통과 295장 v3 상자 모아 보기 (12장씩) → det_label_v3\review\sheet_NN.png + flags.csv (사람이 이상한 것만 표시)
#    --step ftdata : 파인튜닝 1차 데이터 → det_ft1\data\ (flags.csv 괜찮음 254장, 약 단위 val 10%, 학습 쪽만 90/180/270° 복사본)
#    --step ftdata2: 파인튜닝 2차 데이터 → det_ft2\data\ (1차 254장 + 사람이 고친 fix\ + fix2\ 의 확인한 사진, 나누기·돌리기는 1차와 같음)
#    --step fix3   : fix\ 의 미리 채운 상자를 v1 → v3 로 (v1 원본은 fix\Label_auto_v1.txt)
#    --step ftdata3: 파인튜닝 3차 데이터 → det_ft3\data\ (1차와 같은 254장·같은 상자·같은 나누기, 사진만 K0L17 전처리)
#  [주의]  PPOCRLabel 은 저장할 때 '확인 완료' 한 사진만 남기고 Label.txt 를 덮어씀 (savePPlabel)
#          → 자동 라벨 원본을 images\Label_auto.txt 로 따로 둠 (PPOCRLabel 이 건드리지 않는 이름)
#  [저장]  exp_cns\results\det_label_v1\
#            train500.csv            고른 500장 (약·카메라·면·정답)
#            auto.csv                사진마다 읽은 글자·정답 여부·상자 수·큰 상자 여부·상태(auto_pass / check)·이유
#            images\<파일명>.png     원본 400×400 (검출기가 실제로 보는 사진, .gitignore 로 안 올림)
#            images\Label.txt        PPOCRLabel 형식: 'images/<파일명>\t[{"transcription", "points"(네 꼭짓점), "difficult"}]'
#                                    500장 모두 (확인 필요 사진도 자동 상자를 미리 채워 둬서 고치기만 하면 되게)
# =====================================================================

import argparse
import csv
import json
import os
import random
from collections import Counter

import cv2
import numpy as np

import common_cns as cc
import ocr_common_cns as oc

OUT = cc.RESULTS / "det_label_v1"
LAMBDA_DRUGS = {"K-000027", "K-001061", "K-005676", "K-025922", "K-030177"}   # 라벨 각인에 'Λ' 가 있는 약 (학습용 전체 검색)
N_PICK = 500
BIG_FRAC = 0.5                                    # 상자 넓이 ≥ 400×400 의 이 비율이면 '큰 상자'
K0L17R6 = "K0L17R6-q2-Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12_auto_link17_low85_rim0.6"
PRED_DIR = oc.OCR_OUT / "train500" / "paddle_t0" / "orig_alignlong_line_retryK0L17R6"


def select():
    """학습용(train)에서 500장 고르기 → train500.csv."""
    allt = cc.load_items(cc.TRAIN_ROOT)
    sp = cc.split_train(allt)
    pool = [it for it in allt if sp[it["file_name"]] == "train" and it["drug_N"] not in LAMBDA_DRUGS
            and it["truth_kind"] == "보통"]
    by_drug = {}
    for it in pool:
        by_drug.setdefault(it["drug_N"], []).append(it)
    rng = random.Random(42)
    drugs = sorted(by_drug)
    rng.shuffle(drugs)

    def pick(cands, cam):
        """원하는 카메라 사진 중 하나 (없으면 아무거나)."""
        c = [it for it in cands if it["camera_la"] == cam] or cands
        return rng.choice(sorted(c, key=lambda it: it["file_name"]))

    chosen = []
    first_cam = {}
    for i, d in enumerate(drugs):                                       # 1차: 모든 약 1장씩, 카메라 75·90 번갈아
        it = pick(by_drug[d], 75 if i % 2 == 0 else 90)
        chosen.append(it)
        first_cam[d] = it["camera_la"]
    extra = drugs[:N_PICK - len(chosen)]                                # 2차: 섞인 순서 앞쪽 약에서 1장 더, 첫 사진과 반대 카메라
    for d in extra:
        rest = [it for it in by_drug[d] if it not in chosen]
        chosen.append(pick(rest, 90 if first_cam[d] == 75 else 75))
    chosen.sort(key=lambda it: it["file_name"])

    os.makedirs(OUT, exist_ok=True)
    keys = ["file_name", "drug_N", "dl_name", "face", "camera_la", "shape", "form", "truth_raw", "truth"]
    with open(oc.TRAIN500_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: it[k] for k in keys} for it in chosen)
    print(f"저장: {oc.TRAIN500_CSV}")
    print(f"  {len(chosen)}장, {len({it['drug_N'] for it in chosen})}종 (후보 {len(pool)}장 {len(by_drug)}종)")
    print(f"  카메라 {dict(Counter(it['camera_la'] for it in chosen))}, 면 {dict(Counter(it['face'] for it in chosen))}")
    print(f"  모양 {dict(Counter(it['shape'] for it in chosen).most_common())}")


def unrotate_box(box, ang, w=400, h=400):
    """돌린 사진 좌표의 사각형 (x0, y0, x1, y1) → 돌리기 전 사진 좌표의 네 꼭짓점 [[x, y] × 4].
       ocr_common_cns.rotate_color 와 같은 행렬(가운데 기준 ang 도, 캔버스 키움)을 만들고 거꾸로 적용."""
    x0, y0, x1, y1 = box
    return unrotate_pts([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], ang, w, h)


def unrotate_pts(pts, ang, w=400, h=400):
    """돌린 사진 좌표의 점들 [[x, y], ...] → 돌리기 전 사진 좌표 (unrotate_box 의 꼭짓점판, labels2 의 네 꼭짓점용)."""
    pts = np.asarray(pts, np.float64)
    if ang == 0:
        return pts.round().astype(int).tolist()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    Mi = cv2.invertAffineTransform(M)
    back = pts @ Mi[:, :2].T + Mi[:, 2]
    back[:, 0] = back[:, 0].clip(0, w - 1)                              # 돌리며 넓어진 캔버스 바깥은 사진 안으로 자름
    back[:, 1] = back[:, 1].clip(0, h - 1)
    return back.round().astype(int).tolist()


def labels():
    """pred.csv → auto.csv + images\\*.png + images\\Label.txt."""
    items = {it["file_name"]: it for it in oc.load_split("train500", 0)}
    rows = list(csv.DictReader(open(PRED_DIR / "pred.csv", encoding="utf-8-sig")))
    img_dir = OUT / "images"
    os.makedirs(img_dir, exist_ok=True)
    out_rows, lab_lines = [], []
    for r in rows:
        fn = r["file_name"]
        boxes = json.loads(r["best_boxes"])
        ang = float(r["angle"])                                         # 축 정렬된 긴 알약은 소수 각도 (예: 236.3)
        polys = [unrotate_box(b[:4], ang) for b in boxes]
        big = any(cv2.contourArea(np.array(p, np.float32)) >= BIG_FRAC * 400 * 400 for p in polys)
        ok = r["correct"] == "1"
        reasons = []
        if not ok:
            reasons.append("wrong_read")
        if not boxes:
            reasons.append("no_box")
        if big:
            reasons.append("big_box")
        status = "check" if reasons else "auto_pass"
        out_rows.append(dict(file_name=fn, drug_N=r["drug_N"], camera_la=r["camera_la"], face=r["face"],
                             truth=r["truth"], read=r["read"], correct=r["correct"], n_boxes=len(boxes),
                             big_box=int(big), retry=r.get("retry", ""), aligned=r["aligned"], angle=ang,
                             status=status, reason="+".join(reasons)))
        # 검출기가 보는 사진 = 원본을 긴 변 400 으로 맞춘 정사각형 (ocr_common_cns.Prep('orig') 와 같음)
        cc.imwrite(str(img_dir / fn), cc.fit_side(cc.imread(items[fn]["image_path"]), 400))
        ann = [{"transcription": b[4], "points": p, "difficult": False} for b, p in zip(boxes, polys)]
        lab_lines.append(f"images/{fn}\t{json.dumps(ann, ensure_ascii=False)}")

    with open(OUT / "auto.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for name in ("Label.txt", "Label_auto.txt"):                        # Label_auto.txt = PPOCRLabel 이 덮어쓰지 않는 원본 보관용
        with open(img_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lab_lines) + "\n")

    n = len(out_rows)
    st = Counter(r["status"] for r in out_rows)
    print(f"저장: {OUT / 'auto.csv'}, {img_dir / 'Label.txt'} ({n}장)")
    print(f"  읽기 정답률 {sum(r['correct'] == '1' for r in out_rows) / n * 100:.1f}%  "
          f"({sum(r['correct'] == '1' for r in out_rows)}/{n})")
    print(f"  자동 통과 {st['auto_pass']}장 / 확인 필요 {st['check']}장")
    print(f"  확인 필요 이유 (겹침 있음): " + ", ".join(
        f"{k} {sum(k in r['reason'].split('+') for r in out_rows)}" for k in ["wrong_read", "no_box", "big_box"]))
    print(f"  맞혔지만 큰 상자라 확인 필요: {sum(r['correct'] == '1' and r['big_box'] == 1 for r in out_rows)}장")


def make_try():
    """PPOCRLabel 시험용 5장 폴더 try5\\ (사진 + Label.txt). 원본 자동 라벨(Label_auto.txt)에서 복사."""
    rows = list(csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")))
    lab = dict(l.rstrip("\n").split("\t") for l in open(OUT / "images" / "Label_auto.txt", encoding="utf-8"))
    want = [("auto_pass", ""), ("auto_pass", ""), ("check", "wrong_read"), ("check", "no_box"), ("check", "big_box")]
    picked = []
    for st, rs in want:                                                 # 파일명 순서로 조건에 맞는 첫 사진 (다른 약으로)
        r = next(r for r in rows if r["status"] == st and (not rs or r["reason"] == rs)
                 and r["drug_N"] not in {p["drug_N"] for p in picked})
        picked.append(r)
    d = OUT / "try5"
    os.makedirs(d, exist_ok=True)
    lines = []
    for r in picked:
        fn = r["file_name"]
        cc.imwrite(str(d / fn), cc.imread(str(OUT / "images" / fn)))
        lines.append(f"try5/{fn}\t{lab['images/' + fn]}")
        print(f"  {fn}  {r['status']:9s} {r['reason']:10s} 정답 {r['truth']}  읽음 {r['read'] or '-'}  상자 {r['n_boxes']}")
    with open(d / "Label.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장: {d}")


OUT2 = cc.RESULTS / "det_label_v2"


def labels2():
    """det_label_v2\\polys.json (det_polys_cns.py) → v2 라벨 + spot20 비교 그림. 자동 통과/확인 필요 판정은 v1 auto.csv 그대로.
       v2 상자 = Paddle 이 낸 원래 네 꼭짓점을 돌리기 전 400×400 좌표로 되돌린 것 (v1 = 반듯한 사각형을 되돌린 것)."""
    rows = list(csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")))
    polys = json.load(open(OUT2 / "polys.json", encoding="utf-8"))
    lab1 = dict(l.rstrip("\n").split("\t") for l in open(OUT / "images" / "Label_auto.txt", encoding="utf-8"))
    img_dir = OUT2 / "images"
    os.makedirs(img_dir, exist_ok=True)
    lab2, ratio = {}, {"auto_pass": [], "check": []}                    # ratio = 상자마다 v2 넓이 / v1 넓이
    for r in rows:
        fn = r["file_name"]
        cc.imwrite(str(img_dir / fn), cc.imread(str(OUT / "images" / fn)))
        p = polys.get(fn, {"angle": 0, "boxes": []})
        ann = []
        for b, a1 in zip(p["boxes"], json.loads(lab1["images/" + fn])):
            pts = unrotate_pts(b["poly"], p["angle"])
            ann.append({"transcription": b["text"], "points": pts, "difficult": False})
            ratio[r["status"]].append(cv2.contourArea(np.array(pts, np.float32)) /
                                      max(1.0, cv2.contourArea(np.array(a1["points"], np.float32))))
        lab2[fn] = json.dumps(ann, ensure_ascii=False)
    for name in ("Label.txt", "Label_auto.txt"):                        # Label_auto.txt = PPOCRLabel 이 덮어쓰지 않는 원본 보관용
        with open(img_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(f"images/{fn}\t{lab2[r['file_name']]}" for r in rows for fn in [r["file_name"]]) + "\n")
    print(f"저장: {img_dir / 'Label.txt'} ({len(rows)}장)")
    for st, v in ratio.items():
        if v:
            print(f"  {st:9s} 상자 {len(v)}개: v2/v1 넓이 중앙값 {np.median(v):.2f} (25% {np.percentile(v, 25):.2f}, 75% {np.percentile(v, 75):.2f})")

    # spot20 비교 그림: 왼쪽 v1(빨강), 오른쪽 v2(초록), 위 글씨 = 번호·파일명·정답 (cv2 글씨라 영어)
    d = OUT2 / "compare"
    os.makedirs(d, exist_ok=True)
    spot = list(csv.DictReader(open(OUT / "spot20_list.csv", encoding="utf-8-sig")))
    for s in spot:
        fn = s["file_name"]
        img = cc.imread(str(OUT / "images" / fn))
        panes = []
        for lab, col in ((lab1["images/" + fn], (0, 0, 255)), (lab2[fn], (0, 200, 0))):
            im = img.copy()
            for a in json.loads(lab):
                cv2.polylines(im, [np.array(a["points"], np.int32)], True, col, 2)
            panes.append(cv2.resize(im, (500, 500), interpolation=cv2.INTER_CUBIC))
        bar = np.full((36, 1000, 3), 255, np.uint8)
        cv2.putText(bar, f"#{int(s['no']):02d} {fn}  T:{s['truth']}   v1 rect (red) | v2 quad (green)",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cc.imwrite(str(d / f"spot20_{int(s['no']):02d}.png"), np.vstack([bar, np.hstack(panes)]))
    print(f"저장: {d} (spot20 비교 그림 {len(spot)}장)")


N_SPOT = 20


def make_spot():
    """자동 통과 사진에서 N_SPOT 장을 칸(카메라 × 면) 비율대로 뽑아 spot20\\ (사진 + Label.txt + spot20_list.csv)."""
    rows = [r for r in csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")) if r["status"] == "auto_pass"]
    lab = dict(l.rstrip("\n").split("\t") for l in open(OUT / "images" / "Label_auto.txt", encoding="utf-8"))
    groups = {}
    for r in sorted(rows, key=lambda r: r["file_name"]):               # 파일명 순으로 정렬한 뒤 섞어야 언제 돌려도 같음
        groups.setdefault((r["camera_la"], r["face"]), []).append(r)
    # 칸별 장수: 비율대로 내림 → 남은 장수는 소수점 아래가 큰 칸부터 1장씩
    exact = {g: N_SPOT * len(v) / len(rows) for g, v in groups.items()}
    quota = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: exact[g] - quota[g], reverse=True)[:N_SPOT - sum(quota.values())]:
        quota[g] += 1
    rng = random.Random(42)
    picked, drugs = [], set()
    for g in sorted(groups):
        cands = groups[g][:]
        rng.shuffle(cands)
        took = [r for r in cands if r["drug_N"] not in drugs][:quota[g]]
        took += [r for r in cands if r not in took][:quota[g] - len(took)]   # 약이 모자라면 겹쳐도 채움
        picked += took
        drugs |= {r["drug_N"] for r in took}
    picked.sort(key=lambda r: r["file_name"])
    d = OUT / "spot20"
    os.makedirs(d, exist_ok=True)
    lines, out = [], []
    for i, r in enumerate(picked, 1):
        fn = r["file_name"]
        cc.imwrite(str(d / fn), cc.imread(str(OUT / "images" / fn)))
        lines.append(f"spot20/{fn}\t{lab['images/' + fn]}")
        out.append(dict(no=i, file_name=fn, drug_N=r["drug_N"], camera_la=r["camera_la"], face=r["face"],
                        truth=r["truth"], read=r["read"], n_boxes=r["n_boxes"], ok="", problem="", note=""))
    with open(d / "Label.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(OUT / "spot20_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"자동 통과 {len(rows)}장 칸별: " + ", ".join(f"{g[0]}/{g[1]} {len(groups[g])}→{quota[g]}" for g in sorted(groups)))
    print(f"저장: {d} ({len(picked)}장, 약 {len({r['drug_N'] for r in picked})}종), {OUT / 'spot20_list.csv'}")


def clahe_color(img):
    """밝기(Lab 의 L)에만 CLAHE (clip 3, 8×8 칸) → 얕은 각인의 명암 차이를 키움. 그림 확인용일 뿐 학습에는 안 씀."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_sheets(per=5, tile=320, cap=34):
    """'확인 필요' 사진을 파일명 순서로 per 장씩 묶어 그림 한 장으로."""
    rows = [r for r in csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")) if r["status"] == "check"]
    lab = dict(l.rstrip("\n").split("\t") for l in open(OUT / "images" / "Label_auto.txt", encoding="utf-8"))
    d = OUT / "review"
    os.makedirs(d, exist_ok=True)
    n_sheet = (len(rows) + per - 1) // per
    s = tile / 400                                                      # 400 좌표 → 그림 칸 크기
    for k in range(n_sheet):
        strips = []
        for j, r in enumerate(rows[k * per:(k + 1) * per]):
            fn = r["file_name"]
            img = cc.imread(str(OUT / "images" / fn))
            left = img.copy()
            for a in json.loads(lab["images/" + fn]):
                p = np.array(a["points"], np.int32)
                cv2.polylines(left, [p], True, (0, 220, 0), 2)
                cv2.putText(left, a["transcription"] or "-", (int(p[:, 0].min()), max(16, int(p[:, 1].min()) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            pair = np.hstack([cv2.resize(left, (tile, tile), interpolation=cv2.INTER_AREA),
                              cv2.resize(clahe_color(img), (tile, tile), interpolation=cv2.INTER_AREA)])
            bar = np.full((cap, 2 * tile, 3), 255, np.uint8)
            txt = f"#{k * per + j + 1} {fn[:8]}{fn[8:-4].replace('_0_', '_')[-12:]}  T:{r['truth']}  R:{r['read'] or '-'}  {r['reason']}"
            cv2.putText(bar, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            strips.append(np.vstack([bar, pair, np.full((6, 2 * tile, 3), 128, np.uint8)]))
        cc.imwrite(str(d / f"sheet_{k + 1:03d}.png"), np.vstack(strips))
    with open(d / "index.csv", "w", newline="", encoding="utf-8-sig") as f:   # 그림 번호 ↔ 사진 (분류 결과를 맞출 때 씀)
        w = csv.writer(f)
        w.writerow(["sheet", "no", "file_name", "truth", "read", "reason"])
        for i, r in enumerate(rows):
            w.writerow([i // per + 1, i + 1, r["file_name"], r["truth"], r["read"], r["reason"]])
    print(f"저장: {d}  그림 {n_sheet}장 (사진 {len(rows)}장), 번호표 index.csv")


def edit_distance(a, b):
    """편집거리 (ocr_cns.edit_distance 와 같음 — 그쪽은 torch 를 불러와서 여기 따로 둠)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


FIX_RULES = {  # 칸 → (고칠 필요, 우선순위(작을수록 먼저), 설명)
    "E": (1, 1, "큰 상자: 지우고 글자 둘레로 다시"),
    "C": (1, 2, "상자 없음: 새로 그리기"),
    "B": (1, 3, "일부 빠짐: 빠진 글자 상자 추가"),
    "G": (1, 4, "정답 글자가 안 보일 수 있음(라벨 뒤바뀜 의심): 사진에 보이는 글자로"),
    "D": (1, 4, "엉뚱한 곳: 틀린 상자 지우기"),
    "X": (1, 5, "규칙으로 못 정함: 보고 판단"),
    "F": (1, 5, "판단 불가: 보고 결정"),
    "A": (0, 9, "상자 맞음: 고칠 필요 없음"),
}


def rule_class(r):
    """그림 없이 auto.csv 한 줄만 보고 칸 정하기 (fixlist 주석의 규칙)."""
    rd, t, why = r["read"], r["truth"], r["reason"].split("+")
    if "big_box" in why:
        return "E"
    if "no_box" in why or r["n_boxes"] == "0":
        return "C"
    if rd and sorted(rd) == sorted(t):
        return "A"
    if rd and rd in t:
        return "B"
    if 1 - edit_distance(rd, t) / max(len(rd), len(t), 1) < 0.34:          # 비슷한 정도 1/3 미만 = 거의 안 겹침
        return "G"
    return "X"


def fix_list():
    rows = [r for r in csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")) if r["status"] == "check"]
    seen = {}
    if (OUT / "claude_review.csv").exists():
        seen = {r["file_name"]: r for r in csv.DictReader(open(OUT / "claude_review.csv", encoding="utf-8-sig"))}
    lab = dict(l.rstrip("\n").split("\t") for l in open(OUT / "images" / "Label_auto.txt", encoding="utf-8"))
    out = []
    for r in rows:
        c = seen.get(r["file_name"])
        cls = c["cls"] if c else rule_class(r)
        need, pri, how = FIX_RULES[cls]
        if c:                                                           # Claude 가 그림으로 본 사진은 그 판단 (G 인데 글자 없음 = 고칠 필요 없음)
            need = int(c["need_fix"])
        out.append(dict(priority=pri, file_name=r["file_name"], drug_N=r["drug_N"], truth=r["truth"], read=r["read"],
                        n_boxes=r["n_boxes"], auto_reason=r["reason"], cls=cls, by="claude" if c else "rule",
                        need_fix=need, how=how, visible_text=c["visible_text"] if c else "",
                        note=c["note"] if c else ""))
    out.sort(key=lambda r: (r["priority"], r["file_name"]))
    with open(OUT / "fix_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    d = OUT / "fix"
    os.makedirs(d, exist_ok=True)
    lines = []
    for r in out:
        if not r["need_fix"]:
            continue
        fn = r["file_name"]
        cc.imwrite(str(d / fn), cc.imread(str(OUT / "images" / fn)))
        lines.append(f"fix/{fn}\t{lab['images/' + fn]}")
    with open(d / "Label.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장: {OUT / 'fix_list.csv'} ({len(out)}장), PPOCRLabel 폴더 {d} ({len(lines)}장)")
    for cls in "ECBGDXFA":
        n = [r for r in out if r["cls"] == cls]
        if n:
            print(f"  {cls} {len(n):3d}장  고칠 필요 {sum(r['need_fix'] for r in n):3d}  ({FIX_RULES[cls][2]})")


OUT3 = cc.RESULTS / "det_label_v3"
SHRINK_H = (0.5, 0.6, 0.7)                        # --step shrink 그림의 두께 후보
SHRINK_W = 0.85                                   # 길이는 후보 모두 같은 비율
SHRINK_W_CANDS = (0.85, 0.9, 1.0)                 # --step shrink_w 의 길이 후보 (두께 0.7 고정)


def shrink_quad(pts, hf, wf):
    """네 꼭짓점 상자를 가운데·기울기는 그대로 두고 짧은 변(두께) × hf, 긴 변(길이) × wf 로 줄임.
       꼭짓점마다 (점 − 가운데)를 상자의 두 축(0→1 변, 1→2 변 방향)으로 나눠 축별로 줄이므로 꼭짓점 순서가 그대로 유지됨
       (PPOCRLabel·Paddle 학습은 꼭짓점 순서를 씀)."""
    p = np.asarray(pts, np.float64)
    c = p.mean(0)
    e1 = (p[1] - p[0] + p[2] - p[3]) / 2                                # 0→1 변 방향 (마주 보는 두 변 평균)
    e2 = (p[2] - p[1] + p[3] - p[0]) / 2                                # 1→2 변 방향
    u, v = e1 / max(np.linalg.norm(e1), 1e-6), e2 / max(np.linalg.norm(e2), 1e-6)
    fu, fv = (wf, hf) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (hf, wf)   # 긴 쪽 = 길이
    d = p - c
    q = c + np.outer(d @ u * fu, u) + np.outer(d @ v * fv, v)
    q[:, 0] = q[:, 0].clip(0, 399)
    q[:, 1] = q[:, 1].clip(0, 399)
    return q.round().astype(int).tolist()


def make_shrink(by_w=False):
    """spot20 비교 그림: [v2 그대로 | 두께 0.5 | 0.6 | 0.7] (길이는 모두 SHRINK_W) → det_label_v2\\shrink\\spot20_NN.png.
       by_w=True (--step shrink_w): 두께 0.7 고정, [v2 그대로 | 길이 0.85 | 0.9 | 1.0] → det_label_v2\\shrink_w\\
       (사람이 두께 0.7 을 고른 뒤, 길이를 줄이면 첫·끝 글자가 잘릴 수 있어서 길이만 따로 비교)"""
    lab2 = dict(l.rstrip("\n").split("\t") for l in open(OUT2 / "images" / "Label_auto.txt", encoding="utf-8"))
    cands = [(0.7, w) for w in SHRINK_W_CANDS] if by_w else [(h, SHRINK_W) for h in SHRINK_H]
    d = OUT2 / ("shrink_w" if by_w else "shrink")
    os.makedirs(d, exist_ok=True)
    spot = list(csv.DictReader(open(OUT / "spot20_list.csv", encoding="utf-8-sig")))
    for s in spot:
        fn = s["file_name"]
        img = cc.imread(str(OUT / "images" / fn))
        ann = json.loads(lab2["images/" + fn])
        panes = []
        for c in [None] + cands:                                        # None = v2 그대로
            im = img.copy()
            for a in ann:
                pts = a["points"] if c is None else shrink_quad(a["points"], *c)
                cv2.polylines(im, [np.array(pts, np.int32)], True, (0, 200, 0), 2)
            panes.append(cv2.resize(im, (400, 400), interpolation=cv2.INTER_CUBIC))
        bar = np.full((36, 1600, 3), 255, np.uint8)
        head = " | ".join(f"h{h}w{w}" for h, w in cands)
        cv2.putText(bar, f"#{int(s['no']):02d} {fn}  T:{s['truth']}   v2 | {head}",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cc.imwrite(str(d / f"spot20_{int(s['no']):02d}.png"), np.vstack([bar, np.hstack(panes)]))
    print(f"저장: {d} (그림 {len(spot)}장)")


def labels3(hf, wf):
    """v2 라벨의 상자를 shrink_quad(hf, wf) 로 줄인 v3 라벨 → det_label_v3\\images\\Label.txt (+ Label_auto.txt). 판정은 v1 auto.csv 그대로."""
    rows = list(csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")))
    lab2 = dict(l.rstrip("\n").split("\t") for l in open(OUT2 / "images" / "Label_auto.txt", encoding="utf-8"))
    img_dir = OUT3 / "images"
    os.makedirs(img_dir, exist_ok=True)
    lines = []
    for r in rows:
        fn = r["file_name"]
        cc.imwrite(str(img_dir / fn), cc.imread(str(OUT2 / "images" / fn)))
        ann = [dict(a, points=shrink_quad(a["points"], hf, wf)) for a in json.loads(lab2["images/" + fn])]
        lines.append(f"images/{fn}\t{json.dumps(ann, ensure_ascii=False)}")
    for name in ("Label.txt", "Label_auto.txt"):
        with open(img_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"저장: {img_dir / 'Label.txt'} ({len(rows)}장, 두께 × {hf}, 길이 × {wf})")


def review3(per=12, cols=4, tile=300, cap=28):
    """자동 통과 사진의 v3 상자를 한 그림에 per 장씩 (cols 칸 × 줄) → det_label_v3\\review\\sheet_NN.png + flags.csv.
       사람이 그림을 넘기며 이상한 사진만 flags.csv 의 problem(잘림/큼/빠짐/기타)·note 칸에 적음 (나머지 칸은 비워 둠)."""
    rows = [r for r in csv.DictReader(open(OUT / "auto.csv", encoding="utf-8-sig")) if r["status"] == "auto_pass"]
    lab3 = dict(l.rstrip("\n").split("\t") for l in open(OUT3 / "images" / "Label_auto.txt", encoding="utf-8"))
    d = OUT3 / "review"
    os.makedirs(d, exist_ok=True)
    s = tile / 400
    n_sheet = (len(rows) + per - 1) // per
    for k in range(n_sheet):
        cells = []
        for j, r in enumerate(rows[k * per:(k + 1) * per]):
            fn = r["file_name"]
            im = cv2.resize(cc.imread(str(OUT3 / "images" / fn)), (tile, tile), interpolation=cv2.INTER_AREA)
            for a in json.loads(lab3["images/" + fn]):
                cv2.polylines(im, [(np.array(a["points"], np.float64) * s).round().astype(np.int32)], True, (0, 220, 0), 2)
            bar = np.full((cap, tile, 3), 255, np.uint8)
            cv2.putText(bar, f"#{k * per + j + 1}  T:{r['truth']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cells.append(np.vstack([bar, im]))
        while len(cells) % cols:                                        # 마지막 줄 빈칸
            cells.append(np.full_like(cells[0], 255))
        grid = [np.hstack([np.pad(c, ((3, 3), (3, 3), (0, 0)), constant_values=128) for c in cells[i:i + cols]])
                for i in range(0, len(cells), cols)]
        title = np.full((34, grid[0].shape[1], 3), 255, np.uint8)
        cv2.putText(title, f"v3 h0.7 w0.9 - sheet {k + 1:02d}/{n_sheet}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cc.imwrite(str(d / f"sheet_{k + 1:02d}.png"), np.vstack([title] + grid))
    with open(d / "flags.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "sheet", "file_name", "truth", "problem", "note"])
        for i, r in enumerate(rows):
            w.writerow([i + 1, i // per + 1, r["file_name"], r["truth"], "", ""])
    print(f"저장: {d}  그림 {n_sheet}장 (사진 {len(rows)}장), 적는 칸 flags.csv")


OUT_FT1 = cc.RESULTS / "det_ft1" / "data"
OUT_FT2 = cc.RESULTS / "det_ft2" / "data"
OUT_FT3 = cc.RESULTS / "det_ft3" / "data"
K0L17 = "K0L17-c3-Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12_auto_link17_low85"
FT_VAL_FRAC = 0.1                                 # 약 단위로 이 비율을 검증용(val)으로
ROTS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_pts90(pts, ang, w=400, h=400):
    """cv2.rotate 로 돌린 사진에 맞게 꼭짓점도 옮김 (화소 번호 기준이라 W-1, H-1). 꼭짓점 순서는 그대로 둠.
       90 = 시계 방향: (x, y) → (H-1-y, x) / 180: (W-1-x, H-1-y) / 270 = 반시계: (y, W-1-x)"""
    out = []
    for x, y in pts:
        if ang == 90:
            out.append([h - 1 - y, x])
        elif ang == 180:
            out.append([w - 1 - x, h - 1 - y])
        else:
            out.append([y, w - 1 - x])
    return out


def write_ftdata(entries, out, tag):
    """(파일명, 원본 사진 경로, 상자 목록) 들 → out\\ (images\\ + train.txt + val.txt, PaddleOCR 검출 학습 형식) + check_rot.png.
       - 약 단위로 val ≈10% (시드 42) — 같은 약이 학습·검증에 같이 들어가지 않게
       - 학습 쪽만 90/180/270° 복사본 추가 (각인이 여러 방향으로 찍히므로). val 은 원본만 (점수가 부풀지 않게)
       - check_rot.png: 학습 사진 하나의 0/90/180/270 + 상자 (꼭짓점 옮기기가 맞는지 눈으로 확인)"""
    use = [fn for fn, _, _ in entries]
    drugs = sorted({fn[:8] for fn in use})                              # 파일명 앞 8글자 = 약 코드 (K-000030)
    rng = random.Random(42)
    shuffled = drugs[:]
    rng.shuffle(shuffled)
    val_drugs = set(shuffled[:round(len(drugs) * FT_VAL_FRAC)])
    img_dir = out / "images"
    os.makedirs(img_dir, exist_ok=True)
    tr_lines, va_lines = [], []
    for fn, src, ann in entries:
        img = cc.imread(str(src))
        cc.imwrite(str(img_dir / fn), img)
        line = f"images/{fn}\t{json.dumps(ann, ensure_ascii=False)}"
        if fn[:8] in val_drugs:
            va_lines.append(line)
            continue
        tr_lines.append(line)
        h, w = img.shape[:2]
        stem = os.path.splitext(fn)[0]
        for ang, code in ROTS.items():
            rfn = f"{stem}_r{ang:03d}.png"
            cc.imwrite(str(img_dir / rfn), cv2.rotate(img, code))
            rann = [dict(a, points=rotate_pts90(a["points"], ang, w, h)) for a in ann]
            tr_lines.append(f"images/{rfn}\t{json.dumps(rann, ensure_ascii=False)}")
    for name, lines in (("train.txt", tr_lines), ("val.txt", va_lines)):
        with open(out / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    # 확인 그림: 학습 쪽 첫 사진의 4방향
    fn = next(fn for fn in use if fn[:8] not in val_drugs)
    stem = os.path.splitext(fn)[0]
    tr = dict(l.split("\t") for l in tr_lines)
    cells = []
    for ang in (0, 90, 180, 270):
        key = f"images/{fn}" if ang == 0 else f"images/{stem}_r{ang:03d}.png"
        im = cc.imread(str(out / key))
        for a in json.loads(tr[key]):
            p = np.array(a["points"], np.int32)
            cv2.polylines(im, [p], True, (0, 220, 0), 2)
            cv2.circle(im, tuple(int(v) for v in p[0]), 5, (0, 0, 255), -1)    # 빨간 점 = 첫 꼭짓점 (순서 유지 확인)
        cv2.putText(im, f"{ang} deg", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cells.append(np.pad(im, ((3, 3), (3, 3), (0, 0)), constant_values=128))
    title = np.full((34, sum(c.shape[1] for c in cells), 3), 255, np.uint8)
    cv2.putText(title, f"{tag} rotation check - {fn}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cc.imwrite(str(out / "check_rot.png"), np.vstack([title, np.hstack(cells)]))
    n_va = len(va_lines)
    print(f"사용 {len(use)}장 / 약 {len(drugs)}종 → val 약 {len(val_drugs)}종 {n_va}장 (원본만), "
          f"train 약 {len(drugs) - len(val_drugs)}종 {len(use) - n_va}장 × 4방향 = {len(tr_lines)}줄")
    print(f"val 약: {', '.join(sorted(val_drugs))}")
    print(f"저장: {out}  (train.txt, val.txt, images\\, check_rot.png)")


def load_label(path):
    """PPOCRLabel 형식 Label.txt → {파일명: 상자 목록} (열쇠의 '<폴더>/' 는 뗌)."""
    return {l.split("\t")[0].split("/", 1)[1]: json.loads(l.rstrip("\n").split("\t")[1])
            for l in open(path, encoding="utf-8") if l.strip()}


def v3_ok_names():
    """v3 검수(flags.csv)에서 problem 이 '4'(괜찮음) 또는 빈칸인 254장 (1차 학습 사진)."""
    flags = list(csv.DictReader(open(OUT3 / "review" / "flags.csv", encoding="utf-8-sig")))
    return [r["file_name"] for r in flags if r["problem"].strip() in ("4", "")]


def ftdata():
    """검출기 파인튜닝 1차 데이터 → det_ft1\\data\\. 사진 = v3 검수 괜찮음 254장, 라벨 = v3 Label_auto.txt"""
    lab3 = load_label(OUT3 / "images" / "Label_auto.txt")
    write_ftdata([(fn, OUT3 / "images" / fn, lab3[fn]) for fn in v3_ok_names()], OUT_FT1, "det_ft1")


def ftdata2():
    """검출기 파인튜닝 2차 데이터 → det_ft2\\data\\. 1차 254장 (v3) + 사람이 고친 fix\\ 168장 + fix2\\ 63장.
       fix·fix2 는 PPOCRLabel 로 '확인(Check)'한 사진만 Label.txt 에 있음 (건너뛴 사진 = 글자 위치 확신 없음 → 뺌).
       세 묶음이 겹치면 멈춤. 상자가 0개인 사진도 뺌 (검출 학습에서 쓸 게 없음)."""
    lab3 = load_label(OUT3 / "images" / "Label_auto.txt")
    groups = [("v3", [(fn, OUT3 / "images" / fn, lab3[fn]) for fn in v3_ok_names()])]
    for tag, d in (("fix", OUT / "fix"), ("fix2", OUT3 / "fix2")):
        groups.append((tag, [(fn, d / fn, ann) for fn, ann in load_label(d / "Label.txt").items()]))
    seen, entries = set(), []
    for tag, ents in groups:
        dup = seen & {fn for fn, _, _ in ents}
        assert not dup, f"{tag} 가 앞 묶음과 겹침: {sorted(dup)[:5]}"
        ents = [e for e in ents if e[2]]
        seen |= {fn for fn, _, _ in ents}
        entries += ents
        print(f"  {tag}: {len(ents)}장")
    write_ftdata(entries, OUT_FT2, "det_ft2")


def ftdata3():
    """검출기 파인튜닝 3차 데이터 → det_ft3\\data\\. 1차(det_ft1)와 사진 고르기·상자·약 나누기가 같고 사진만 K0L17 전처리.
       - 전처리 결과는 prep_v2\\final_train500\\<K0L17>\\ 에 저장됨 (oc.Prep 이 없을 때만 만듦, 있으면 읽기만)
       - K0L17 은 원본과 똑같이 긴 변 400 맞춤이라 v3 상자 좌표를 그대로 씀 → check_rot.png 로 눈으로 확인
       - 1차 폴더(det_ft1)는 읽기만 하고 쓰지 않음. 이미 있으면 멈춤 (덮어쓰기 방지)"""
    assert not OUT_FT3.exists(), f"이미 있음, 덮어쓰지 않음: {OUT_FT3}"
    lab3 = load_label(OUT3 / "images" / "Label_auto.txt")
    names = v3_ok_names()
    items = {it["file_name"]: it for it in oc.load_split("train500", 0)}
    prep = oc.Prep(K0L17, "train500")
    for k, fn in enumerate(names, 1):
        out, _ = prep(items[fn], cc.imread(items[fn]["image_path"]))     # 저장 폴더에 없으면 만들어 저장
        assert out.shape[:2] == cc.imread(str(OUT3 / "images" / fn)).shape[:2], fn   # 상자 좌표를 그대로 쓰려면 크기가 같아야 함
        if k % 50 == 0:
            print(f"  전처리 {k}/{len(names)}")
    write_ftdata([(fn, prep.cache / fn, lab3[fn]) for fn in names], OUT_FT3, "det_ft3")
    # 1차와 같은 조건인지 확인: 학습·검증 줄(사진 이름 + 상자)이 글자 하나까지 같아야 함
    for name in ("train.txt", "val.txt"):
        a = open(OUT_FT1 / name, encoding="utf-8").read()
        b = open(OUT_FT3 / name, encoding="utf-8").read()
        assert a == b, f"{name} 가 1차와 다름"
    print("확인: train.txt·val.txt 가 1차와 글자 하나까지 같음 (사진 내용만 K0L17)")


def fix3():
    """fix\\ 폴더(사람이 고칠 182장)의 미리 채운 상자를 v1 → v3 로 바꿈 (1차 학습 det_ft1 과 상자 크기 기준을 맞추려고).
       v1 원본은 fix\\Label_auto_v1.txt 로 보관, Label.txt·Label_auto.txt 는 v3 로 새로 씀. 사진 순서는 그대로."""
    d = OUT / "fix"
    if not (d / "Label_auto_v1.txt").exists():
        os.replace(d / "Label_auto.txt", d / "Label_auto_v1.txt")
    keys = [l.split("\t")[0] for l in open(d / "Label_auto_v1.txt", encoding="utf-8") if l.strip()]
    lab3 = dict(l.rstrip("\n").split("\t") for l in open(OUT3 / "images" / "Label_auto.txt", encoding="utf-8"))
    lines = [f"{k}\t{lab3['images/' + k.split('/', 1)[1]]}" for k in keys]       # 열쇠 'fix/<파일명>' 은 그대로
    for name in ("Label.txt", "Label_auto.txt"):
        with open(d / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"저장: {d / 'Label.txt'} ({len(lines)}장, v3 상자)  v1 원본: Label_auto_v1.txt")


def fix2():
    """사람이 아직 상자를 안 본 63장 → det_label_v3\\fix2\\ (PPOCRLabel 폴더, v3 상자 미리 채움, 열쇠 'fix2/<파일명>').
       - 41장: review3 flags.csv 에서 이상 표시(잘림·큼·기울기·기타)라 1차 학습(det_ft1)에서 뺀 사진
       - 22장: fix_list.csv 의 A(순서만 다름, 고칠 필요 없음) — v3 상자는 사람이 본 적 없음
       fix2_list.csv: 순서·파일명·정답·출처(flag/A)·flags 표시·메모 (고칠 때 참고)."""
    flags = list(csv.DictReader(open(OUT3 / "review" / "flags.csv", encoding="utf-8-sig")))
    rows = [(r["file_name"], r["truth"], "flag", r["problem"].strip(), (r[list(r)[5]] or "").strip())   # 6번째 칸 = 메모 (머리줄에 쉼표가 있어 열 이름이 길게 나뉨)
            for r in flags if r["problem"].strip() not in ("4", "")]
    rows += [(r["file_name"], r["truth"], "A", "", r["how"])
             for r in csv.DictReader(open(OUT / "fix_list.csv", encoding="utf-8-sig")) if r["cls"] == "A"]
    lab3 = dict(l.rstrip("\n").split("\t") for l in open(OUT3 / "images" / "Label_auto.txt", encoding="utf-8"))
    d = OUT3 / "fix2"
    os.makedirs(d, exist_ok=True)
    lines = []
    for fn, *_ in rows:
        cc.imwrite(str(d / fn), cc.imread(str(OUT3 / "images" / fn)))
        lines.append(f"fix2/{fn}\t{lab3['images/' + fn]}")
    for name in ("Label.txt", "Label_auto.txt"):                   # Label_auto.txt = PPOCRLabel 이 덮어써도 남는 원본
        with open(d / name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(OUT3 / "fix2_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "file_name", "truth", "source", "flag", "note"])
        for i, r in enumerate(rows):
            w.writerow([i + 1, *r])
    print(f"저장: {d} ({len(rows)}장: flag {sum(r[2] == 'flag' for r in rows)} + A {sum(r[2] == 'A' for r in rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["select", "labels", "try", "sheets", "fixlist", "spot", "labels2", "shrink", "shrink_w", "review3",
                             "labels3", "ftdata", "fix3", "fix2", "ftdata2", "ftdata3"])
    ap.add_argument("--h", type=float, default=0.6, help="labels3: 상자 두께(짧은 변) 비율")
    ap.add_argument("--w", type=float, default=SHRINK_W, help="labels3: 상자 길이(긴 변) 비율")
    a = ap.parse_args()
    if a.step == "labels3":
        return labels3(a.h, a.w)
    {"select": select, "labels": labels, "try": make_try, "sheets": make_sheets, "fixlist": fix_list, "spot": make_spot,
     "labels2": labels2, "shrink": make_shrink, "shrink_w": lambda: make_shrink(by_w=True), "review3": review3,
     "ftdata": ftdata, "fix3": fix3, "fix2": fix2, "ftdata2": ftdata2, "ftdata3": ftdata3}[a.step]()


if __name__ == "__main__":
    main()
