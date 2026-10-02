"""
알약 각인 인식 3차 실험 (전처리 조합 비교)
==========================================

3차에서 바뀐 점 (2차 코드 기반)
------------------------------
- 전처리 조합 비교:  A 배경제거만 / B A+조명평탄화 / C A+반사광제거 / D A+반사광제거+조명평탄화
- AUTO: 인쇄/음각을 자동으로 구분해서 인쇄->A, 음각->D 적용
- 인쇄/음각 자동 구분 규칙과 그 정확도도 함께 기록

(아래는 2차 설명)

1차 실험에서 바뀐 점
--------------------
- 전처리: 4가지 -> 원본(M0), CLAHE(M1) 2가지 (이진화는 1차에서 손해로 확인돼 제외)
- 채점: 글자 인식만 -> 글자 인식 + 최종 약 판별(Top-1)
- 매칭 3단계 비교: ① 매칭 없음 ② 편집거리 매칭 ③ 편집거리 + 색·모양 필터
- 결과를 각인 유형별(인쇄/음각)·오류 원인별로 나눠서 분석

실행 순서
---------
  (1) python imprint_experiment_v3.py crop --input images/multi_scene.jpg --out crops
  (2) labels.csv  : 이미지경로,정답각인          (crop_overview.png 보고 작성)
      pill_db.csv : 약이름,각인,모양,색           (판별 후보 목록 = 식약처/AI Hub 정보로 작성)
  (3) python imprint_experiment_v3.py run --engine easyocr      (또는 --engine tesseract)

결과 (results_<엔진>/ 폴더)
---------------------------
  detail.csv          이미지 x 전처리별: 인식 결과, 글자 점수, 매칭 3종 판별 결과, 오류 원인
  summary.csv         전처리 x 매칭 조합별 평균
  by_type.csv         각인 유형(인쇄/음각)별 평균
  errors.csv          오류 원인별 개수
  chart_summary.png   조합별 정확도·속도 그래프
"""
import argparse
import csv
import os
import time
from collections import Counter

import cv2
import numpy as np


# ======================================================================
# STEP 1. 알약 잘라내기 (여러 알약 사진 -> 알약 1개짜리 사진들)
# ======================================================================
def crop_pills(input_path, out_dir, min_area_ratio=0.0005):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(input_path)

    # 빠르게 찾기 위해 긴 변 1000px로 줄인 사진에서 위치를 찾음
    scale = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Otsu: 밝은 알약 / 어두운 배경을 나누는 기준값을 자동으로 골라 두 색으로 나눔
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # 작은 먼지 제거
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 알약 속 구멍 메우기

    # 흰 덩어리(=알약)마다 기울어진 상자를 씌움
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    rects = [cv2.minAreaRect(c) for c in cnts if cv2.contourArea(c) >= min_area]
    rects.sort(key=lambda r: (round(r[0][1] / 50), r[0][0]))   # 위->아래, 왼->오른 순서

    overview = small.copy()
    saved = []
    for i, ((cx, cy), (w, h), ang) in enumerate(rects):
        cx, cy, w, h = cx / scale, cy / scale, w / scale * 1.15, h / scale * 1.15   # 원본 크기 + 여백 15%
        if w < h:                       # 긴 쪽이 가로가 되도록
            w, h, ang = h, w, ang + 90
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (cx, cy))
        path = os.path.join(out_dir, f"crop_{i:02d}.png")
        cv2.imwrite(path, crop)
        saved.append(path)
        box = np.int32(cv2.boxPoints(((cx * scale, cy * scale), (w * scale, h * scale), ang)))
        cv2.drawContours(overview, [box], 0, (0, 0, 255), 2)
        cv2.putText(overview, str(i), (int(cx * scale) - 10, int(cy * scale) - int(h * scale / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(out_dir, "crop_overview.png"), overview)
    print(f"[STEP 1] 알약 {len(saved)}개 검출 -> {out_dir}/ (번호 확인: crop_overview.png)")
    return saved


# ======================================================================
# STEP 2. 배경 지우기 + 모양·색 특징 뽑기
# ======================================================================
def pill_mask(bgr):
    """알약 영역 마스크(흰색=알약)와 장축/단축 비율을 반환"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 0.3 * g.size:
        # 알약이 화면을 꽉 채운 사진 등: 전체를 알약으로 간주
        return np.full_like(g, 255), max(g.shape) / min(g.shape)
    c = max(cnts, key=cv2.contourArea)
    (_, (w, h), _) = cv2.minAreaRect(c)
    m = np.zeros_like(g)
    cv2.drawContours(m, [c], -1, 255, -1)
    k = max(3, int(min(g.shape) * 0.04))
    m = cv2.erode(m, np.ones((k, k), np.uint8))   # 테두리 그림자 제외
    return m, max(w, h) / max(1.0, min(w, h))


def isolate_pill(bgr):
    """배경을 알약 평균 밝기로 칠해 지운 그레이 이미지 + 모양비"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m, aspect = pill_mask(bgr)
    x, y, w, h = cv2.boundingRect(m)
    out = np.full_like(g, int(np.median(g[m > 0])))
    out[m > 0] = g[m > 0]
    return out[y:y + h, x:x + w], aspect


def pill_features(bgr):
    """판별 후보를 좁힐 때 쓰는 모양·색 (일부러 거칠게 분류)
    - 모양: 장축/단축 비율 1.3 초과면 장방형, 아니면 원형
    - 색  : 채도가 뚜렷하면 색 이름, 아니면 '무채색'(하양·회색 묶음)
            -> 조명·화이트밸런스에 따라 하양/회색은 사진마다 뒤바뀌어서 하나로 묶음"""
    m, aspect = pill_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S = int(np.median(hsv[..., 0][m > 0])), int(np.median(hsv[..., 1][m > 0]))
    shape = "장방형" if aspect > 1.3 else "원형"
    if S < 50:
        color = "무채색"
    elif H < 8 or H >= 160:
        color = "빨강"
    elif H < 15:
        color = "주황"
    elif H < 35:
        color = "노랑"
    elif H < 85:
        color = "초록"
    elif H < 130:
        color = "파랑"
    else:
        color = "보라"
    return shape, color


# DB의 색 표기(식약처식)를 위 색 묶음으로 변환
COLOR_GROUP = {"하양": "무채색", "흰색": "무채색", "회색": "무채색", "검정": "무채색",
               "노랑": "노랑", "주황": "주황", "빨강": "빨강", "분홍": "빨강",
               "초록": "초록", "연두": "초록", "파랑": "파랑", "남색": "파랑", "보라": "보라"}


# ======================================================================
# STEP 3. 전처리 (M0 원본 / M1 CLAHE)
# ======================================================================
def resize_long(g, long_side=400):
    s = long_side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def flatten_light(g):
    """조명 평탄화: 크게 흐린 사진(=조명 지도)으로 나눠 한쪽이 어둡게 찍힌 그늘을 없앰"""
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, min(g.shape) * 0.08)).astype(np.float32)
    out = g.astype(np.float32) / (bg + 1) * float(np.median(g))
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_glare(g):
    """반사광 제거: 하얗게 날아간(거의 255) 작은 부분만 주변 값으로 메움"""
    med = float(np.median(g))
    m = ((g >= 240) & (g > med + 25)).astype(np.uint8) * 255
    if m.sum() == 0:
        return g
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.inpaint(g, m, 5, cv2.INPAINT_TELEA)


def glare_ratio(g):
    med = float(np.median(g))
    return float(((g >= 240) & (g > med + 25)).mean())


def detect_imprint_type(gray):
    """인쇄/음각 자동 구분: 조명 평탄화 후 주변 밝기의 65% 미만인 '진한 잉크' 픽셀이
    0.5% 이상이면 인쇄, 아니면 음각(눌러 새긴 각인)으로 판단"""
    bg = cv2.GaussianBlur(gray, (0, 0), max(3.0, min(gray.shape) * 0.08)).astype(np.float32)
    dark = (gray.astype(np.float32) / (bg + 1) < 0.65).mean()
    return ("인쇄" if dark >= 0.005 else "음각"), float(dark)


def pA(g):
    return g


def pB(g):
    return flatten_light(g)


def pC(g):
    return remove_glare(g)


def pD(g):
    return flatten_light(remove_glare(g))


METHODS = {"A_배경제거만": pA, "B_+조명평탄화": pB, "C_+반사광제거": pC, "D_+반사광+평탄화": pD, "AUTO_유형별": None}
AUTO_RULE = {"인쇄": pA, "음각": pD}


# ======================================================================
# STEP 4. 돌려가며 읽기 (OCR)
# ======================================================================
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class TesseractOCR:
    def __init__(self):
        import pytesseract
        self.t = pytesseract
        self.cfg = f"--psm 7 -c tessedit_char_whitelist={ALLOW}"

    def __call__(self, img):
        img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_REPLICATE)
        d = self.t.image_to_data(img, config=self.cfg, output_type=self.t.Output.DICT)
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return "".join(w for w, _ in words), float(np.mean([c for _, c in words]))


class EasyOCR:
    def __init__(self, gpu=False):
        import easyocr
        self.r = easyocr.Reader(["en"], gpu=gpu)

    def __call__(self, img):
        res = self.r.readtext(img, allowlist=ALLOW)
        if not res:
            return "", 0.0
        return "".join(t for _, t, _ in res), float(np.mean([c for _, _, c in res]) * 100)


def rotate(g, ang):
    if ang == 0:
        return g
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def norm(s):
    """대소문자·공백·기호 무시: 'alza 18' -> 'ALZA18'"""
    return "".join(ch for ch in s.upper() if ch.isalnum())


# ======================================================================
# STEP 5. 다수결로 답 고르기
# ======================================================================
def read_imprint(gray, aspect, method_fn, ocr):
    """원형은 30도씩 12방향, 장방형은 0/180도로 돌려 읽고 다수결로 1개 선택"""
    g = method_fn(resize_long(gray))
    angles = [0, 180] if aspect > 1.3 else list(range(0, 360, 30))
    cands = []
    for a in angles:
        text, conf = ocr(rotate(g, a))
        cands.append((norm(text), conf))
    votes = Counter(t for t, _ in cands if 1 <= len(t) <= 10)
    pick = max(cands, key=lambda c: (votes.get(c[0], 0) if c[0] else -1, c[1]))
    return pick[0], [t for t, _ in cands]


# ======================================================================
# STEP 6. 약 판별 (매칭 3종)
# ======================================================================
def levenshtein(a, b):
    """편집거리: a를 b로 바꾸는 데 필요한 최소 글자 수정 횟수"""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def similarity(a, b):
    """각인 유사도 (Heo et al., 2023 방식): (길이 - 편집거리) / (길이 + 편집거리)"""
    if not a or not b:
        return 0.0
    L, ed = max(len(a), len(b)), levenshtein(a, b)
    return (L - ed) / (L + ed)


MIN_SIM = 0.3   # 유사도 하한: 6글자 각인이면 3글자까지 틀려도 인정, 4글자 이상 틀리면 판별 불가


def identify(pred, shape, color, db, mode):
    """반환: 판별된 약 이름 (판별 불가면 '')
    mode = 'none'  : 인식 결과가 DB 각인과 글자까지 똑같을 때만 인정
           'edit'  : 가장 비슷한 각인을 고름
           'edit+attr' : 모양·색이 맞는 후보로 먼저 좁힌 뒤 가장 비슷한 각인"""
    cands = db
    if mode == "edit+attr":
        cands = [d for d in db if d["shape"] == shape and COLOR_GROUP.get(d["color"], d["color"]) == color] or db
    if mode == "none":
        hit = [d for d in cands if d["imprint"] == pred]
        return hit[0]["name"] if len(hit) == 1 else ""
    if mode == "edit+attr" and len(cands) == 1:
        # 모양·색만으로 후보가 1개로 좁혀지면 글자를 못 읽어도 그 약으로 판별
        return cands[0]["name"]
    if not pred:
        return ""
    scores = sorted(((similarity(pred, d["imprint"]), d["name"]) for d in cands), reverse=True)
    if scores[0][0] < MIN_SIM or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        return ""    # 충분히 비슷한 각인이 없거나 1등이 동점이면 판별 불가 (우연히 맞는 것 방지)
    return scores[0][1]


MATCH_MODES = {"none": "① 매칭 없음", "edit": "② 편집거리", "edit+attr": "③ 편집거리+색·모양"}


def char_acc(pred, gt):
    """문자 단위 정확도 = 1 - (틀린 글자 수 / 정답 글자 수)"""
    return max(0.0, 1 - levenshtein(pred, gt) / max(len(gt), 1))


def error_type(pred, gt, all_texts):
    if pred == gt:
        return "정답"
    if not pred:
        return "빈 결과(아무것도 못 읽음)"
    if gt in all_texts:
        return "방향/선택 실패(후보엔 정답 있었음)"
    if char_acc(pred, gt) >= 0.5:
        return "부분 인식(헷갈리는 글자)"
    return "오인식"


# ======================================================================
# 실험 실행
# ======================================================================
def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]


def run(labels_path, db_path, engine, out_dir, gpu=False):
    out_dir = out_dir or f"results_{engine}"
    os.makedirs(out_dir, exist_ok=True)
    items = [(r[0].strip(), r[1].strip(), r[2].strip() if len(r) > 2 else "") for r in load_csv(labels_path)]
    db = [{"name": r[0].strip(), "imprint": norm(r[1]), "shape": r[2].strip(), "color": r[3].strip()}
          for r in load_csv(db_path)[1:]]
    name_of = {d["imprint"]: d["name"] for d in db}
    ocr = TesseractOCR() if engine == "tesseract" else EasyOCR(gpu=gpu)

    rows = []
    for path, gt_raw, ptype in items:
        bgr = cv2.imread(path)
        if bgr is None:
            print("이미지 없음:", path)
            continue
        gt = norm(gt_raw)
        gray, aspect = isolate_pill(bgr)
        shape, color = pill_features(bgr)
        auto_type, dark = detect_imprint_type(gray)
        for mname, fn in METHODS.items():
            t0 = time.perf_counter()
            if fn is None:      # AUTO: 유형을 먼저 판단하고 그에 맞는 전처리 적용
                fn = AUTO_RULE[detect_imprint_type(gray)[0]]
            pred, all_texts = read_imprint(gray, aspect, fn, ocr)
            ms = (time.perf_counter() - t0) * 1000
            row = {"image": path, "type": ptype, "auto_type": auto_type, "dark_ink_ratio": round(dark, 4),
                   "glare_ratio": round(glare_ratio(gray), 4), "gt": gt, "method": mname, "pred": pred,
                   "exact": int(pred == gt), "char_acc": round(char_acc(pred, gt), 3),
                   "shape": shape, "color": color, "ms": round(ms, 1),
                   "error": error_type(pred, gt, all_texts)}
            for mode in MATCH_MODES:
                got = identify(pred, shape, color, db, mode)
                row[f"id_{mode}"] = got
                row[f"ok_{mode}"] = int(got != "" and got == name_of.get(gt))
            rows.append(row)
            print(f"{os.path.basename(path):20s} {mname:14s} 정답={gt:7s} 인식={pred or '-':9s} "
                  f"{shape}/{color}  판별: " + " ".join(str(row[f'ok_{m}']) for m in MATCH_MODES) + f"  {ms:5.0f}ms")

    def write(name, data):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    write("detail.csv", rows)

    summary = []
    for mname in METHODS:
        r = [x for x in rows if x["method"] == mname]
        for mode, label in MATCH_MODES.items():
            summary.append({
                "전처리": mname, "매칭": label, "n": len(r),
                "글자_완전일치_%": round(100 * np.mean([x["exact"] for x in r]), 1),
                "글자_문자정확도_%": round(100 * np.mean([x["char_acc"] for x in r]), 1),
                "약판별_Top1_%": round(100 * np.mean([x[f"ok_{mode}"] for x in r]), 1),
                "평균_ms": round(float(np.mean([x["ms"] for x in r])), 1),
            })
    write("summary.csv", summary)

    by_type = []
    for ptype in sorted({x["type"] for x in rows}):
        for mname in METHODS:
            r = [x for x in rows if x["type"] == ptype and x["method"] == mname]
            by_type.append({"유형": ptype or "-", "전처리": mname, "n": len(r),
                            "글자_문자정확도_%": round(100 * np.mean([x["char_acc"] for x in r]), 1),
                            **{f"Top1_{MATCH_MODES[m]}": round(100 * np.mean([x[f'ok_{m}'] for x in r]), 1)
                               for m in MATCH_MODES}})
    write("by_type.csv", by_type)

    errs = []
    for mname in METHODS:
        c = Counter(x["error"] for x in rows if x["method"] == mname)
        for k, v in c.most_common():
            errs.append({"전처리": mname, "오류 원인": k, "개수": v})
    write("errors.csv", errs)

    labeled = [x for x in rows if x["method"] == list(METHODS)[0] and x["type"]]
    if labeled:
        ok = sum(x["type"] == x["auto_type"] for x in labeled)
        print(f"\n[인쇄/음각 자동 구분] {ok}/{len(labeled)} 정확")
    print(f"\n=== 요약 (엔진: {engine}, 알약 {len(items)}개) ===")
    for s in summary:
        print(f"{s['전처리']:14s} {s['매칭']:14s} 글자 완전일치 {s['글자_완전일치_%']:5.1f}%  "
              f"문자정확도 {s['글자_문자정확도_%']:5.1f}%  약판별 Top-1 {s['약판별_Top1_%']:5.1f}%  {s['평균_ms']:.0f}ms")
    plot_summary(summary, os.path.join(out_dir, "chart_summary.png"), engine, len(items))
    return rows, summary


# ======================================================================
# 그래프
# ======================================================================
def _setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    else:
        path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_summary(summary, path, engine, n):
    plt = _setup_font()
    methods = list(METHODS)
    modes = list(MATCH_MODES.values())
    colors = ["#c9d8ee", "#7aa6e0", "#2a78d6"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.6, 1]})

    ax = axes[0]
    x = np.arange(len(methods))
    bw = 0.27
    for i, mode in enumerate(modes):
        vals = [next(s["약판별_Top1_%"] for s in summary if s["전처리"] == m and s["매칭"] == mode) for m in methods]
        bars = ax.bar(x + (i - 1) * bw, vals, bw - 0.02, color=colors[i], label=mode)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n", 1) for m in methods], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("약 판별 Top-1 정확도 (%)", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[1]
    ms = [next(s["평균_ms"] for s in summary if s["전처리"] == m) for m in methods]
    ax.barh([m.split("_")[0] for m in methods][::-1], ms[::-1], height=0.45, color="#2a78d6")
    for i, v in enumerate(ms[::-1]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(ms) * 1.3)
    ax.set_title("알약 1개당 처리시간 (ms)", fontsize=11)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"3차 실험 결과 (OCR 엔진: {engine}, 알약 {n}개)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="white")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop")
    c.add_argument("--input", required=True)
    c.add_argument("--out", default="crops")
    r = sub.add_parser("run")
    r.add_argument("--labels", default="labels.csv")
    r.add_argument("--db", default="pill_db.csv")
    r.add_argument("--engine", choices=["tesseract", "easyocr"], default="easyocr")
    r.add_argument("--out", default=None)
    r.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    if a.cmd == "crop":
        crop_pills(a.input, a.out)
    else:
        run(a.labels, a.db, a.engine, a.out, a.gpu)
