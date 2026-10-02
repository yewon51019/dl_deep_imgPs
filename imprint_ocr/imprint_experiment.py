"""
알약 각인 추출 비교 실험 (전처리 방식별 OCR 정확도·속도 비교)
==============================================================

사용법
------
1) 여러 알약이 찍힌 사진에서 알약을 하나씩 잘라내기
     python imprint_experiment.py crop --input images/multi_scene.jpg --out crops

   -> crops/ 폴더에 crop_00.png, crop_01.png ... 저장
   -> crops/crop_overview.png 에 어떤 알약이 몇 번인지 번호가 표시됨

2) labels.csv 에 정답 각인 적기 (파일명,정답)
     crops/crop_00.png,alza18
     images/single_YJ.jpg,YJ
     ...
   * 정답은 대소문자·공백 무시하고 비교함 (alza 18 == ALZA18)

3) 실험 실행
     python imprint_experiment.py run --labels labels.csv --engine tesseract
     python imprint_experiment.py run --labels labels.csv --engine easyocr --out results_easyocr
     (EasyOCR: pip install easyocr, GPU 있으면 --gpu 추가)

   -> results/results_detail.csv  : 이미지 x 방식별 인식 결과·정확도·시간
   -> results/results_summary.csv : 방식별 평균
   -> results/summary_chart.png   : 방식별 정확도/속도 그래프
   -> results/preprocess_examples.png : 방식별 전처리 결과 예시

비교하는 전처리 방식 (모두 크롭된 알약 1개 단위로 적용)
------------------------------------------------------
  M0 원본       : 그레이스케일만
  M1 CLAHE      : 지역 대비 향상
  M2 CLAHE+적응형 threshold : 지역 블록 단위 이진화
  M3 Top/Black-hat+Otsu     : 각인(국소 명암 변화)만 뽑은 뒤 이진화

공통 처리: 크롭 안의 배경을 지우고(isolate_pill), 원형 알약은 각인 방향을 몰라
30도 간격 12방향, 장방형 알약은 0/180도로 돌려가며 OCR한 뒤
"여러 방향에서 같은 결과가 많이 나온 것(다수결) > 신뢰도" 순으로 최종 결과를 고른다.
(정답을 보고 고르는 것이 아니므로 공정한 비교)

oracle_exact = "후보들 중에 정답이 하나라도 있었는가" -> 방향 선택만 완벽했다면 얻을 수 있는
상한선. 실제 성능이 아니라 '글자 인식 능력'과 '방향 선택 문제'를 분리해서 보기 위한 참고 지표.
"""
import argparse
import csv
from collections import Counter
import os
import time

import cv2
import numpy as np


# ------------------------------------------------------------------
# 1. 다중 알약 사진 -> 알약별 크롭
# ------------------------------------------------------------------
def crop_pills(input_path, out_dir, min_area_ratio=0.0005):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(input_path)

    # 처리 속도를 위해 축소본에서 위치를 찾고, 원본 해상도에서 잘라냄
    scale = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 배경(어두운 천)과 알약(밝음)을 Otsu로 자동 분리
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    rects = [cv2.minAreaRect(c) for c in cnts if cv2.contourArea(c) >= min_area]
    rects.sort(key=lambda r: (round(r[0][1] / 50), r[0][0]))  # 위->아래, 왼->오른

    overview = small.copy()
    saved = []
    for i, ((cx, cy), (w, h), ang) in enumerate(rects):
        # 원본 좌표로 환산, 여백 15%
        cx, cy, w, h = cx / scale, cy / scale, w / scale * 1.15, h / scale * 1.15
        # 회전된 사각형을 똑바로 펴서 잘라냄 (긴 축이 가로가 되도록)
        if w < h:
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
    print(f"알약 {len(saved)}개 검출 -> {out_dir}/ 저장 (번호 확인: crop_overview.png)")
    return saved


# ------------------------------------------------------------------
# 2. 공통 준비: 알약 바깥(배경) 지우기 + 크기 맞추기
# ------------------------------------------------------------------
def isolate_pill(bgr):
    """크롭 안에서 알약 영역만 남기고 배경은 알약 평균 밝기로 채움.
    반환: (그레이 이미지, 장축/단축 비율)  -> 비율로 원형/장방형 판단"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g, g.shape[1] / g.shape[0]
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * g.size:          # 알약이 화면을 꽉 채운 경우 등: 마스킹 생략
        return g, max(g.shape) / min(g.shape)
    (_, (w, h), _) = cv2.minAreaRect(c)
    m = np.zeros_like(g)
    cv2.drawContours(m, [c], -1, 255, -1)
    k = max(3, int(min(g.shape) * 0.04))           # 테두리 그림자 제거용으로 살짝 안쪽만 사용
    m = cv2.erode(m, np.ones((k, k), np.uint8))
    x, y, ww, hh = cv2.boundingRect(m)
    out = np.full_like(g, int(np.median(g[m > 0])))
    out[m > 0] = g[m > 0]
    return out[y:y + hh, x:x + ww], max(w, h) / max(1.0, min(w, h))


def resize_long(g, long_side=400):
    s = long_side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


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


# ------------------------------------------------------------------
# 3. 비교할 전처리 방식 (입력: 그레이, 출력: OCR에 넣을 후보 이미지 리스트)
# ------------------------------------------------------------------
def _clahe(g):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def m0_raw(g):
    return [g]


def m1_clahe(g):
    return [_clahe(g)]


def m2_adaptive(g):
    b = cv2.adaptiveThreshold(_clahe(cv2.GaussianBlur(g, (3, 3), 0)), 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 8)
    return [b, 255 - b]  # 글자가 어두운 경우/밝은 경우 둘 다


def m3_tophat(g):
    c = _clahe(g)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tb = cv2.add(cv2.morphologyEx(c, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(c, cv2.MORPH_BLACKHAT, k))
    _, b = cv2.threshold(cv2.GaussianBlur(tb, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return [255 - b]  # 흰 바탕에 검은 글자


METHODS = {
    "M0_원본": m0_raw,
    "M1_CLAHE": m1_clahe,
    "M2_CLAHE+적응형threshold": m2_adaptive,
    "M3_Tophat/Blackhat+Otsu": m3_tophat,
}

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


# ------------------------------------------------------------------
# 4. OCR 엔진 (텍스트, 신뢰도 0~100 반환)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 5. 평가 지표 + 인식 절차
# ------------------------------------------------------------------
def norm(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def char_acc(pred, gt):
    """문자 단위 정확도 = 1 - CER (0 미만은 0)"""
    return max(0.0, 1 - levenshtein(pred, gt) / max(len(gt), 1))


def recognize(bgr, method_fn, ocr):
    """알약 1개 -> 후보 회전각마다 OCR -> 최종 1개 선택.
    원형 알약은 각인 방향을 모르므로 30도 간격 12방향, 장방형은 0/180도만 시도.
    선택 규칙(정답을 보지 않음): 여러 회전에서 같은 결과가 많이 나온 것(다수결) > 신뢰도."""
    g, aspect = isolate_pill(bgr)
    g = resize_long(g)
    angles = [0, 180] if aspect > 1.3 else list(range(0, 360, 30))
    cands = []
    for a in angles:
        for im in method_fn(rotate(g, a)):
            text, conf = ocr(im)
            cands.append((norm(text), conf))
    votes = Counter(t for t, _ in cands if 1 <= len(t) <= 10)
    pick = max(cands, key=lambda c: (votes.get(c[0], 0) if c[0] else -1, c[1]))
    return pick[0], pick[1], [t for t, _ in cands]


# ------------------------------------------------------------------
# 6. 실험 실행
# ------------------------------------------------------------------
def run(labels_path, engine, out_dir, gpu=False):
    os.makedirs(out_dir, exist_ok=True)
    with open(labels_path, encoding="utf-8-sig") as f:
        items = [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if r and not r[0].startswith("#")]

    ocr = TesseractOCR() if engine == "tesseract" else EasyOCR(gpu=gpu)
    rows = []
    for path, gt in items:
        bgr = cv2.imread(path)
        if bgr is None:
            print("이미지 없음:", path)
            continue
        g = norm(gt)
        for mname, fn in METHODS.items():
            t0 = time.perf_counter()
            pred, conf, all_texts = recognize(bgr, fn, ocr)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "image": path, "gt": g, "method": mname, "pred": pred,
                "exact": int(pred == g), "char_acc": round(char_acc(pred, g), 3),
                # 방향만 맞혔다면 맞았을까? (후보 중 정답이 있었는지 = 상한선, 참고용)
                "oracle_exact": int(g in all_texts),
                "conf": round(conf, 1), "ms": round(ms, 1),
            })
            print(f"{os.path.basename(path):22s} {mname:26s} 정답={g:8s} 인식={pred:10s} "
                  f"{'O' if pred == g else 'X'}  {ms:6.0f}ms")

    with open(os.path.join(out_dir, "results_detail.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for mname in METHODS:
        r = [x for x in rows if x["method"] == mname]
        summary.append({
            "method": mname, "n": len(r),
            "exact_acc_%": round(100 * np.mean([x["exact"] for x in r]), 1),
            "char_acc_%": round(100 * np.mean([x["char_acc"] for x in r]), 1),
            "oracle_exact_%": round(100 * np.mean([x["oracle_exact"] for x in r]), 1),
            "avg_ms": round(float(np.mean([x["ms"] for x in r])), 1),
        })
    with open(os.path.join(out_dir, "results_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n=== 방식별 요약 (엔진: {engine}) ===")
    for s in summary:
        print(f"{s['method']:26s} 완전일치 {s['exact_acc_%']:5.1f}%  문자정확도 {s['char_acc_%']:5.1f}%  "
              f"(방향 정답 가정 시 {s['oracle_exact_%']:5.1f}%)  평균 {s['avg_ms']:.0f}ms")

    plot_summary(summary, os.path.join(out_dir, "summary_chart.png"), engine)
    plot_examples([p for p, _ in items], os.path.join(out_dir, "preprocess_examples.png"))
    return rows, summary


# ------------------------------------------------------------------
# 7. 시각화
# ------------------------------------------------------------------
def _setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if any(cand == f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    else:
        for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_summary(summary, path, engine=""):
    plt = _setup_font()
    names = [s["method"].replace("_", "\n", 1) for s in summary]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    specs = [("exact_acc_%", "완전일치 정확도 (%)", 100), ("char_acc_%", "문자 단위 정확도 (%)", 100),
             ("avg_ms", "평균 처리시간 (ms/알약)", None)]
    for ax, (key, title, xmax) in zip(axes, specs):
        vals = [s[key] for s in summary]
        ax.barh(y, vals, height=0.5, color="#2a78d6")
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:g}", va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        if xmax:
            ax.set_xlim(0, xmax * 1.15)
        else:
            ax.set_xlim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"전처리 방식별 각인 인식 결과 (OCR 엔진: {engine}, 알약 {summary[0]['n']}개)", fontsize=11)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="white")
    plt.close()


def plot_examples(paths, path, n=4):
    plt = _setup_font()
    picks = paths[:n]
    fig, axes = plt.subplots(len(picks), len(METHODS) + 1, figsize=(2.4 * (len(METHODS) + 1), 1.9 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, p in enumerate(picks):
        bgr = cv2.imread(p)
        g = resize_long(isolate_pill(bgr)[0])
        axes[i, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if i == 0:
            axes[i, 0].set_title("입력(크롭)", fontsize=9)
        for j, (mname, fn) in enumerate(METHODS.items(), 1):
            axes[i, j].imshow(fn(g)[0], cmap="gray")
            if i == 0:
                axes[i, j].set_title(mname, fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor="white")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop")
    c.add_argument("--input", required=True)
    c.add_argument("--out", default="crops")
    r = sub.add_parser("run")
    r.add_argument("--labels", default="labels.csv")
    r.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    r.add_argument("--out", default="results")
    r.add_argument("--gpu", action="store_true", help="EasyOCR에서 GPU 사용")
    a = ap.parse_args()
    if a.cmd == "crop":
        crop_pills(a.input, a.out)
    else:
        run(a.labels, a.engine, a.out, a.gpu)
