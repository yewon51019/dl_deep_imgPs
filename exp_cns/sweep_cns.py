# =====================================================================
#  sweep_cns.py  —  전처리 단계별 비교 시트 만들기 + 사람이 채택한 3개를 평가용 432장에 적용
# =====================================================================
#
#  [진행 방식 (사용자와 정함)]
#     단계마다 파라미터를 조금씩 바꾼 후보를 한 시트에 모음 → 사람이 보고 3개 채택(번호로) →
#     채택한 3개 각각에 다음 단계 후보를 붙여 다시 시트 → … 반복
#     '괜찮다' 기준 = 5항목 체크리스트 (① 획 대비 ② 글자 모양 ③ 표면 잡음 ④ 가짜 무늬 ⑤ 고르게 좋은가),
#     동점이면 더 약한 처리(작은 커널·'안 함').
#  [지금 만든 단계]
#     1  가우시안 (k 3·5·7·9 × σ 0.5·1·1.5·2·3 + 안 함 = 21개, 번호 G00~G20) — 입력은 gray
#     1b 색 채널 (gray·L·R·G·B·min·auto 7가지 × 1단계에서 채택한 가우시안 3개 = 21개, 번호 C0-G13 …)
#        채널은 원래 가우시안 앞 단계라 처리 순서는 컬러 → 채널 → 400 맞춤 → 가우시안. auto 가 고른 채널은 따로 집계.
#        사용자 요청으로 결과는 모두 컬러: 그 채널의 가우시안 결과를 밝기(L)로, 색(a·b)은 원본 것을 입힘 (ps.recolor).
#        (흑백 7줄 시트는 *_7ch.png 로 보관)
#     2  평균 필터 (1b에서 채택한 것들 × 안 함·k3·k5·k7·k9, 번호 M0-C6-G12 …) — 밝기에만 적용 후 원본 색 입힘
#     3a 히스토그램 ① clip 고르기 (2에서 채택한 것들 × 안 함·HE·CLAHE clip 1~8(tile 8 고정)·비교용 R·G·B 각각, 번호 H0-M2-C6-G08 …)
#     3b 히스토그램 ② tile 고르기 (3a 의 CLAHE 채택 × tile 2·4·8·12·16, 번호 T0-H6-M2-C5-G12 …). HE 채택은 tile 이 없어 4단계로 그대로 넘김
#     4a 샤프닝 ① 세기 (3b 채택 + 3a 의 HE 채택 × 세기 0·0.25·0.5·1·1.5·2·3, 반경 2·문턱 0 고정, 번호 A0-… )
#     4b 샤프닝 ② 반경 (4a 채택 × 반경 0.5·1·1.5·2·3·4, 세기 고정, 번호 R0-…)
#     4c 샤프닝 ③ 문턱 (4b 채택 × 문턱 0·2·5·10·20, 세기·반경 고정, 번호 D0-…)
#     5a 점 제거 ① 위치 (4c 채택 × 없음·끝에서·처음에서 + 마스크 확인 2칸, 기본값, 번호 P0-…). 색 a·b 도 같이 메움
#     5b 점 제거 ② 마무리 (5a 채택 × 점 제거 안 함·fh0·fh2·fh6·fh6+획 보호 마무리 + 마스크, 획 보호에 밝은 선 추가, 번호 F0-…)
#     5c 점 제거 ③ 점 찾기 기준 (5b 채택 중 점 제거하는 것 × (dark, color) 8가지, 번호 Q0-…) + 마스크 시트
#     크기 기준(점 최대 넓이·획 최소 길이)은 5c 채택 후 필요하면 사용자 확인을 받고 추가한다.
#  [샘플 8장]  어려운 경우를 일부러 넣어 규칙으로 고정 (모든 단계에서 같은 8장). 카메라 75°·앞면·회전각 가장 작은 사진.
#  [저장]  exp_cns\results\prep_v2\
#     samples8.csv                     : 샘플 8장과 고른 이유
#     step1_gauss\sheet_step1_gauss.png      : 8블록 × 21칸 (블록 = 샘플 한 장, 행 = k, 열 = σ)
#     step1_gauss\sheet_step1_gauss_zoom.png : 같은 구성, 가운데 200×200 을 2배 확대 (흐림 차이를 보기 쉽게)
#     step1_gauss\candidates.csv       : 번호 → 설정, 한 장당 처리 시간(432장, 첫 장 워밍업 제외)
#     step1_gauss\<번호>_<설정>\        : --pick 으로 채택한 것만 432장 저장 (+ params.json)
#     selection_log.csv                : 단계마다 채택한 번호·설정·이유
#  [실행 예]  python exp_cns\sweep_cns.py --step 1                                   ← 시트 만들기
#             python exp_cns\sweep_cns.py --step 1 --pick G07,G12,G15 --why "…"      ← 채택 기록 + 432장 적용
# =====================================================================

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

import common_cns as cc
import prep_steps as ps

OUT = cc.RESULTS / "prep_v2"
CELL = 220                       # 시트 한 칸 크기 (px)
BAR = 24                         # 칸 위 제목 줄 높이

# ---------------------------------------------------------------------
# 샘플 8장: (영문 조건 — 시트에 쓰는 글씨, 한글 이유 — CSV 에 기록, 약 코드 또는 조건 함수)
# ---------------------------------------------------------------------
SAMPLE_RULES = [
    ("white round tablet", "흰 원형 정제, 파인 각인", "K-000209"),
    ("white soft capsule", "흰 연질캡슐 (가장 희미)", "K-014249"),
    ("2-color hard capsule", "두 색 경질캡슐, 진한 절반", "K-021931"),
    ("colored tablet", "색 있는 정제", "K-000441"),
    ("spotted tablet", "점 많은 정제", "K-006037"),
    ("printed capsule", "인쇄 글자 캡슐", "K-000251"),
    ("white tablet, short text", "흰 정제, 짧은 각인", "K-023405"),
    ("white on light-gray bg", "연회색 배경 + 흰 알약",
     lambda it: it["back_color"] == "연회색 배경" and it["color1"] == "하양"),
]


def pick_samples8(items):
    """규칙대로 8장을 고름. 약마다 카메라 75° → 앞면 우선 → 회전각 가장 작은 사진 (75° 가 없으면 다른 각도)."""
    rank = lambda it: (it["camera_la"] != 75, it["face"] != "앞면", it["rot"], it["file_name"])
    rows, used = [], set()
    for en, ko, rule in SAMPLE_RULES:
        if callable(rule):
            cand = [it for it in items if rule(it) and it["drug_N"] not in used]
        else:
            cand = [it for it in items if it["drug_N"] == rule]
        if not cand:
            print(f"[주의] 샘플 조건에 맞는 사진 없음: {ko}")
            continue
        it = min(cand, key=rank)
        rows.append((it, en, ko))
        used.add(it["drug_N"])
    return rows


# ---------------------------------------------------------------------
# 단계 정의: 후보 목록(번호, 칸 제목, 설정)과 시트 배치(행·열)
# ---------------------------------------------------------------------
def step1_candidates():
    """1단계 가우시안 후보: G00 = 안 함, G01~G20 = k(행) × σ(열). → (번호, 칸 제목, 설정, 행 이름, 열 이름)"""
    cands = [("G00", "G00 none", {"k": 0, "s": 0}, "none", None)]
    n = 1
    for k in (3, 5, 7, 9):
        for s in (0.5, 1.0, 1.5, 2.0, 3.0):
            cands.append((f"G{n:02d}", f"G{n:02d} k{k} s{s:g}", {"k": k, "s": s}, f"k={k}", f"s={s:g}"))
            n += 1
    return cands


def picked(step):
    """selection_log.csv 에서 그 단계에 채택한 (번호, 설정) 목록 (기록된 순서). 같은 번호가 여러 번이면 한 번만."""
    log = OUT / "selection_log.csv"
    out = {}
    if log.exists():
        for r in csv.DictReader(open(log, encoding="utf-8-sig")):
            if r["step"] == step:
                out.setdefault(r["code"], json.loads(r["params"]))
    if not out:
        raise SystemExit(f"{step}단계 채택 기록이 없습니다. 먼저 --step {step} --pick … 을 실행하세요")
    return list(out.items())


def step1b_candidates():
    """1b단계 색 채널 후보: 채널 7가지(행) × 1단계에서 채택한 가우시안 3개(열). 번호 예: C1-G13.
       처리 순서: 컬러 → 채널 → 400 맞춤 → 가우시안 (채널은 원래 가우시안 앞에 오는 단계)."""
    cands = []
    for i, ch in enumerate(ps.CHANNELS):                               # C0~C6: 모두 '그 채널을 밝기로 + 원본 색 입히기' 컬러 결과
        for gcode, gp in picked("1"):
            p = {"ch": ch, "k": gp["k"], "s": gp["s"], "gauss": gcode}
            glab = "none" if gp["k"] == 0 else f"k{gp['k']} s{gp['s']:g}"
            cands.append((f"C{i}-{gcode}", f"C{i}-{gcode} {ch} {glab}", p, f"C{i} {ch}", f"{gcode} {glab}"))
    return cands


def bright_1b(bgr, p):
    """1b단계까지의 '밝기' 사진 (색 입히기 전): 채널 → 400 맞춤 → 가우시안. 다음 단계들은 이 밝기에만 처리하고 마지막에 색을 입힘."""
    return ps.gauss(cc.fit_side(ps.channel_p(bgr, p), ps.SIDE), p["k"], p["s"])


MEAN_KS = [1, 3, 5, 7, 9]          # 2단계 평균 필터 후보 (1 = 안 함)


def step2_candidates():
    """2단계 평균 필터 후보: 1b단계에서 채택한 것들(행) × 평균 필터 5가지(열). 번호 예: M1-C6-G12 (= C6-G12 + 평균 k3)."""
    cands = []
    for pcode, pp in picked("1b"):
        glab = f"{pp['ch']} k{pp['k']} s{pp['s']:g}"
        for i, k in enumerate(MEAN_KS):
            klab = "none" if k == 1 else f"k{k}"
            cands.append((f"M{i}-{pcode}", f"M{i}-{pcode} {klab}", {**pp, "prev": pcode, "mean_k": k},
                          f"{pcode}|{glab}", f"M{i} {klab}"))
    return cands


def bright_2(bgr, p):
    """2단계까지의 밝기 사진 (색 입히기 전): 1b 밝기 → 평균 필터."""
    return ps.mean_f(bright_1b(bgr, p), p["mean_k"])


# 3단계-① 후보 열: (번호 접미, 열 이름, 설정). tile 8 고정으로 clip 을 촘촘히 + 안 함 + HE + 비교용(R·G·B 각각)
HIST_A = ([("H0", "H0 none", {"hist": "none"}), ("H1", "H1 HE", {"hist": "he"})] +
          [(f"H{i + 2}", f"H{i + 2} c{c:g}", {"hist": "clahe", "clip": c, "tile": 8})
           for i, c in enumerate([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])] +
          [("H9", "H9 RGB c2 bad", {"hist": "rgb", "clip": 2.0, "tile": 8})])


def step3a_candidates():
    """3단계-① 히스토그램(clip 고르기) 후보: 2단계에서 채택한 것들(행) × HIST_A 10가지(열). 번호 예: H4-M2-C6-G08."""
    cands = []
    for pcode, pp in picked("2"):
        rlab = f"{pp['ch']} k{pp['k']} s{pp['s']:g} m{pp['mean_k']}"
        for hc, col, hp in HIST_A:
            p = {**pp, "prev": pcode, "clip": 2.0, "tile": 8, **hp}
            cands.append((f"{hc}-{pcode}", f"{hc}-{pcode} {col.split(' ', 1)[1]}"[:26], p, f"{pcode}|{rlab}", col))
    return cands


TILES = [2, 4, 8, 12, 16]          # 3단계-② tile 후보 (처리 크기 400 에서 한 칸 = 400 / tile px)


def step3b_candidates():
    """3단계-② tile 고르기 후보: 3a 에서 채택한 CLAHE 들(행, clip 그대로) × tile 5가지(열). 번호 예: T3-H6-M2-C5-G12.
       HE 는 칸을 나누지 않아 tile 이 없으므로 여기서 빼고, 다음 단계(샤프닝)에 그대로 넘긴다."""
    cands = []
    for pcode, pp in picked("3a"):
        if pp["hist"] != "clahe":
            continue
        rlab = f"{pp['ch']} m{pp['mean_k']} c{pp['clip']:g}"
        for i, t in enumerate(TILES):
            cands.append((f"T{i}-{pcode}", f"T{i}-{pcode} t{t}", {**pp, "prev": pcode, "tile": t},
                          f"{pcode}|{rlab}", f"T{i} t{t} ({400 // t}px)"))
    return cands


def apply_hist(bgr, p):
    """3단계 적용: 비교용(rgb)은 색 입힌 사진의 B·G·R 를 각각 CLAHE, 나머지는 밝기에 HE/CLAHE 후 원본 색 입히기."""
    base = cc.fit_side(bgr, ps.SIDE)
    if p["hist"] == "rgb":
        return ps.clahe_rgb_each(ps.recolor(base, bright_2(bgr, p)), p["clip"], p["tile"])
    return ps.recolor(base, ps.hist(bright_2(bgr, p), p["hist"], p["clip"], p["tile"]))


def bright_3(bgr, p):
    """3단계까지의 밝기 사진 (색 입히기 전): 2단계 밝기 → HE 또는 CLAHE."""
    return ps.hist(bright_2(bgr, p), p["hist"], p["clip"], p["tile"])


AMOUNTS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]      # 4단계-① 샤프닝 세기 후보 (0 = 안 함), 반경 2·문턱 0 고정


def step4_inputs():
    """4단계 입력: 3b 에서 채택한 CLAHE 들 + 3a 에서 채택한 HE (tile 이 없어 3b 를 거치지 않고 그대로 넘어옴)."""
    return picked("3b") + [(c, p) for c, p in picked("3a") if p["hist"] == "he"]


def step4a_candidates():
    """4단계-① 샤프닝 세기 후보: 4단계 입력 5개(행) × 세기 7가지(열). 번호 예: A3-T3-H8-M2-C6-G12 (= 세기 1.0)."""
    cands = []
    for pcode, pp in step4_inputs():
        rlab = f"{pp['ch']} m{pp['mean_k']} " + ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}")
        for i, a in enumerate(AMOUNTS):
            alab = "none" if a == 0 else f"a{a:g}"
            cands.append((f"A{i}-{pcode}", f"A{i}-{pcode} {alab}", {**pp, "prev": pcode, "amount": a, "sigma": 2.0, "thr": 0},
                          f"{pcode}|{rlab}", f"A{i} {alab}"))
    return cands


SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]          # 4단계-② 샤프닝 반경 후보 (세기는 4a 채택값 고정, 문턱 0)


def step4b_candidates():
    """4단계-② 샤프닝 반경 후보: 4a 에서 채택한 것들(행, 세기 고정) × 반경 6가지(열). 번호 예: R1-A5-T2-H8-M2-C6-G12."""
    cands = []
    for pcode, pp in picked("4a"):
        rlab = ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}") + f" a{pp['amount']:g}"
        for i, s in enumerate(SIGMAS):
            cands.append((f"R{i}-{pcode}", f"R{i}-{pcode} s{s:g}", {**pp, "prev": pcode, "sigma": s},
                          f"{pcode}|{rlab}", f"R{i} s{s:g}"))
    return cands


THRS = [0, 2, 5, 10, 20]          # 4단계-③ 샤프닝 문턱 후보 (세기·반경은 4b 채택값 고정)


def step4c_candidates():
    """4단계-③ 샤프닝 문턱 후보: 4b 에서 채택한 것들(행, 세기·반경 고정) × 문턱 5가지(열). 번호 예: D2-R2-A5-T2-H8-M2-C6-G12."""
    cands = []
    for pcode, pp in picked("4b"):
        rlab = ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}") + f" a{pp['amount']:g} s{pp['sigma']:g}"
        for i, t in enumerate(THRS):
            cands.append((f"D{i}-{pcode}", f"D{i}-{pcode} t{t}", {**pp, "prev": pcode, "thr": t},
                          f"{pcode}|{rlab}", f"D{i} thr{t}"))
    return cands


def apply_sharpen(bgr, p):
    """4단계 적용: 3단계 밝기 → 샤프닝 → 원본 색 입히기."""
    return ps.recolor(cc.fit_side(bgr, ps.SIDE), ps.sharpen(bright_3(bgr, p), p["amount"], p["sigma"], p["thr"]))


def chain(b0, p):
    """밝기 b0(채널 → 400 맞춤)에 2~4단계 채택 처리를 차례로: 가우시안 → 평균 → HE/CLAHE → 샤프닝.
       (bright_1b → bright_2 → bright_3 → sharpen 과 같은 계산을, 점 제거를 '처음'에 넣을 때도 쓰려고 분리)"""
    g = ps.gauss(b0, p["k"], p["s"])
    g = ps.mean_f(g, p["mean_k"])
    g = ps.hist(g, p["hist"], p["clip"], p["tile"])
    return ps.sharpen(g, p["amount"], p["sigma"], p["thr"])


SPOT_POS = [("P0", "P0 none", "none"), ("P1", "P1 spot@end", "end"), ("P2", "P2 spot@start", "start"),
            ("P3", "P3 mask@end", "mask_end"), ("P4", "P4 mask@start", "mask_start")]


def step5a_candidates():
    """5단계-① 점 제거 위치 후보: 4c 에서 채택한 것들(행) × 없음 / 끝에서 / 처음에서 / 마스크 확인 2칸(열).
       점 제거 파라미터는 기본값(ps.SPOT_DEFAULT). 마스크 칸(P3·P4)은 확인용이라 채택 대상이 아님."""
    cands = []
    for pcode, pp in picked("4c"):
        rlab = ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}") + f" a{pp['amount']:g} s{pp['sigma']:g} thr{pp['thr']}"
        for code, col, pos in SPOT_POS:
            cands.append((f"{code}-{pcode}", f"{code}-{pcode}", {**pp, "prev": pcode, "spot_pos": pos, **ps.SPOT_DEFAULT},
                          f"{pcode}|{rlab}", col))
    return cands


def apply_spot(bgr, p):
    """5단계 적용.
       none  : 4단계까지 결과 (밝기 + 원본 색)
       end   : 4단계까지 끝난 밝기에서 점을 찾아 지움 (색 a·b 도 같이 메움)
       start : 채널 → 400 맞춤 직후의 원본 밝기에서 점부터 지운 뒤, 가우시안~샤프닝을 똑같이 다시 함
       mask_*: 같은 처리의 획 보호(초록)·점(빨강) 마스크를 4단계 결과 위에 칠한 확인 그림"""
    base = cc.fit_side(bgr, ps.SIDE)
    _, a, b = cv2.split(cv2.cvtColor(base, cv2.COLOR_BGR2LAB))
    b0 = cc.fit_side(ps.channel_p(bgr, p), ps.SIDE)                         # 채널 (+6단계 색 특징 더하기)
    sp = {k: p[k] for k in ps.SPOT_DEFAULT}
    pos = p["spot_pos"]
    if pos == "none":
        return ps.recolor(base, chain(b0, p))
    if pos in ("end", "mask_end"):
        L4 = chain(b0, p)
        L2, a2, b2, stroke, spot = ps.remove_spots(L4, a, b, **sp)
        if pos == "mask_end":
            return ps.overlay_masks(ps.recolor(base, L4), stroke, spot)
        return cv2.cvtColor(cv2.merge([L2, a2, b2]), cv2.COLOR_LAB2BGR)
    L2, a2, b2, stroke, spot = ps.remove_spots(b0, a, b, **sp)                # start / mask_start
    if pos == "mask_start":
        return ps.overlay_masks(ps.recolor(base, chain(b0, p)), stroke, spot)
    return cv2.cvtColor(cv2.merge([chain(L2, p), a2, b2]), cv2.COLOR_LAB2BGR)


# 5단계-② 마무리 방식 후보 열: (번호 접미, 열 이름, 바꿀 값). 모두 획 보호에 밝은 선 추가(bright=True)
FINISH = [("F0", "F0 no spot", {"spot_pos": "none"}),
          ("F1", "F1 fh0", {"final_h": 0}),
          ("F2", "F2 fh2", {"final_h": 2}),
          ("F3", "F3 fh6", {"final_h": 6}),
          ("F4", "F4 fh6 protect", {"final_h": 6, "protect": True}),
          ("F5", "F5 mask", {"spot_pos": "mask"})]


def step5b_candidates():
    """5단계-② 마무리 방식 후보: 5a 에서 채택한 것들(행, 위치 끝/처음 그대로) × FINISH 6칸(열). 번호 예: F1-P1-D3-R2-A5-T2-H8-M2-C6-G12.
       5a 에서 '마무리 흐림(fh6)이 글씨를 흐리게 만든 주원인'으로 확인 → fh 0·2·6 + '획 보호 마무리'를 비교."""
    cands = []
    for pcode, pp in picked("5a"):
        rlab = f"spot@{pp['spot_pos']} " + ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}")
        for code, col, ch in FINISH:
            p = {**pp, "prev": pcode, **ps.SPOT_DEFAULT, "bright": True, **ch}
            if ch.get("spot_pos") == "mask":                          # 마스크 칸은 그 행의 위치(끝/처음)에 맞춰
                p["spot_pos"] = "mask_" + pp["spot_pos"]
            cands.append((f"{code}-{pcode}", f"{code}-{pcode}", p, f"{pcode}|{rlab}", col))
    return cands


DETECT = [(2, 4), (2, 6), (4, 4), (4, 6), (6, 4), (6, 6), (8, 4), (8, 6)]   # 5단계-③ (dark, color) 후보, (6, 6) = 지금 값


def step5c_candidates():
    """5단계-③ 점 찾기 기준 후보: 5b 에서 채택한 것 중 '점 제거를 하는' 것들(행) × (dark, color) 8가지(열). 번호 예: Q2-F4-P1-….
       5b 에서 '점 제거 안 함'으로 고른 것은 바꿀 기준이 없어 여기서 빼고 그대로 최종 후보로 넘긴다."""
    cands = []
    for pcode, pp in picked("5b"):
        if pp["spot_pos"] not in ("end", "start"):
            continue
        rlab = f"spot@{pp['spot_pos']} " + ("HE" if pp["hist"] == "he" else f"c{pp['clip']:g} t{pp['tile']}") + \
               f" fh{pp['final_h']}" + (" protect" if pp["protect"] else "")
        for i, (d, c) in enumerate(DETECT):
            cands.append((f"Q{i}-{pcode}", f"Q{i}-{pcode}", {**pp, "prev": pcode, "dark": d, "color": c},
                          f"{pcode}|{rlab}", f"Q{i} dark{d} col{c}"))
    return cands


def mask_of(bgr, p):
    """같은 설정의 마스크 확인 그림 (초록 = 획 보호, 빨강 = 지울 점). 위치(끝/처음)는 설정 그대로."""
    return apply_spot(bgr, {**p, "spot_pos": "mask_" + p["spot_pos"]})


# 6단계 색 특징 더하기
#   기준 경로: 5c 채택 Q4 를 CLAHE clip 3 으로 낮춘 것 (clip 시트에서 사용자가 고름: 흐린 사진 노이즈가 줄어듦)
#   후보 (사용자가 임시 시트 sheet_addcolor4 에서 고름): K0 = 더하기 없음(기존 auto), K1 = 양방향 α16 t1.5
#   기준 경로 q2: 5c 채택 Q2 그대로 (clip 8 tile 16). 사용자 요청으로 획 잇기 + 테두리 빼기를 얹어 봄
STEP6_BASE = [("Q4-F4-P1-D3-R3-A5-T3-H8-M2-C6-G12", {"clip": 3.0}, "c3"),
              ("Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12", {}, "q2")]
#   K0L17   = K0 + 마지막에 획 잇기(선 필터 길이 17 → 히스테리시스 low85) (임시 시트 sheet_k2_link 에서 사용자가 고름)
#   K0L17R6 = K0L17 + 테두리 빼기 (사진 긴 변의 60% 이상 긴 조각은 진하게 하지 않음, 임시 시트 sheet_rim_cut 에서 정함)
ADDCOLOR = [("K0", "K0 auto only", {"add_alpha": 0}),
            ("K1", "K1 both a16 t1.5", {"add_alpha": 16, "add_thr": 1.5, "add_both": True}),
            ("K0L17", "K0 + link len17", {"add_alpha": 0, "link_len": 17, "link_low": 85}),
            ("K0L17R6", "K0 + link len17 rim0.6", {"add_alpha": 0, "link_len": 17, "link_low": 85, "link_rim": 0.6})]


def apply6(bgr, p):
    """6단계 적용: 5단계 경로(apply_spot, 채널에서 색 특징 더하기 포함) → (link_len 이 있으면) 마지막에 획 잇기
       (link_rim 이 있으면 테두리 조각은 빼고)."""
    out = apply_spot(bgr, p)
    if p.get("link_len"):
        out = ps.link_strokes(out, p["link_len"], p["link_low"], rim_ext=p.get("link_rim"))
    return out


def step6_candidates():
    """6단계 색 특징 더하기 후보: 기준 경로(행) × ADDCOLOR(열). 번호 예: K1-c3-Q4-F4-P1-…"""
    pk = dict(picked("5c"))
    cands = []
    for pcode, over, tag in STEP6_BASE:
        pp = {**pk[pcode], **over}
        rlab = f"{tag} (clip{pp['clip']:g} tile{pp['tile']})"
        for code, col, ch in ADDCOLOR:
            cands.append((f"{code}-{tag}-{pcode}", f"{code}-{tag}", {**pp, "prev": f"{tag}-{pcode}", **ch},
                          f"{tag}-{pcode}|{rlab}", col))
    return cands


# 단계 정의: prepare(컬러 원본) → 그 단계의 공통 입력, apply(입력, 설정) → 결과 400 (1단계 흑백, 1b부터 컬러)
# 시간(ms)은 apply 만 잰다 — 1단계는 가우시안만, 1b단계는 채널 + 400 맞춤 + 가우시안 전체
STEPS = {
    "1": {"folder": "step1_gauss", "name": "gauss", "candidates": step1_candidates,
          "prepare": ps.to_gray400,
          "apply": lambda g, p: ps.gauss(g, p["k"], p["s"]),
          "label": lambda p: "none" if p["k"] == 0 else f"k{p['k']}_s{p['s']:g}"},
    "1b": {"folder": "step1b_channel", "name": "channel", "candidates": step1b_candidates,
           "prepare": lambda bgr: bgr,
           # 사용자 요청(전부 컬러로): 채널 → 400 맞춤 → 가우시안 결과를 밝기로 쓰고, 원본(400 맞춤) 색을 입힘
           "apply": lambda bgr, p: ps.recolor(cc.fit_side(bgr, ps.SIDE),
                                              ps.gauss(cc.fit_side(ps.channel(bgr, p["ch"]), ps.SIDE), p["k"], p["s"])),
           "label": lambda p: f"{p['ch']}_{p['gauss']}"},
    # 2단계: 1b 채택 → 밝기에 평균 필터 → 원본 색 입히기 (시간은 채널부터 색 입히기까지 전체)
    "2": {"folder": "step2_mean", "name": "mean", "candidates": step2_candidates,
          "prepare": lambda bgr: bgr,
          "apply": lambda bgr, p: ps.recolor(cc.fit_side(bgr, ps.SIDE), ps.mean_f(bright_1b(bgr, p), p["mean_k"])),
          "label": lambda p: f"{p['prev']}_mean{'none' if p['mean_k'] == 1 else p['mean_k']}"},
    # 3단계-①: 2단계 채택 → 밝기에 HE/CLAHE(tile 8, clip 여러 개) → 원본 색 입히기 (+ 비교용 R·G·B 각각)
    "3a": {"folder": "step3a_clip", "name": "clip", "candidates": step3a_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_hist,
           "label": lambda p: f"{p['prev']}_{p['hist']}" + (f"_c{p['clip']:g}_t{p['tile']}" if p["hist"] in ("clahe", "rgb") else "")},
    # 3단계-②: 3a 의 CLAHE 채택(clip 고정) → tile 바꾸기. 적용 방식은 3a 와 같음 (밝기에 CLAHE 후 원본 색 입히기)
    "3b": {"folder": "step3b_tile", "name": "tile", "candidates": step3b_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_hist,
           "label": lambda p: f"{p['prev']}_t{p['tile']}"},
    # 4단계-①: 3단계 채택(3b CLAHE + 3a HE) → 밝기에 샤프닝(세기 여러 개, 반경 2, 문턱 0) → 원본 색 입히기
    "4a": {"folder": "step4a_amount", "name": "amount", "candidates": step4a_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_sharpen,
           "label": lambda p: f"{p['prev']}_a{p['amount']:g}"},
    # 4단계-②: 4a 채택(세기 고정) → 반경만 바꿔 샤프닝. 적용 방식은 4a 와 같음
    "4b": {"folder": "step4b_radius", "name": "radius", "candidates": step4b_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_sharpen,
           "label": lambda p: f"{p['prev']}_s{p['sigma']:g}"},
    # 4단계-③: 4b 채택(세기·반경 고정) → 문턱만 바꿔 샤프닝. 적용 방식은 4a 와 같음
    "4c": {"folder": "step4c_threshold", "name": "threshold", "candidates": step4c_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_sharpen,
           "label": lambda p: f"{p['prev']}_thr{p['thr']}"},
    # 5단계-①: 4c 채택 → 점 제거 위치(없음/끝/처음) + 마스크 확인. 점 제거가 한 장 약 0.4초라 시간은 36장(약마다 1장)으로만 잼
    "5a": {"folder": "step5a_spotpos", "name": "spotpos", "candidates": step5a_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_spot, "time_n": 36,
           "label": lambda p: f"{p['prev']}_spot{p['spot_pos']}"},
    # 5단계-②: 5a 채택 → 마무리 방식(fh 0·2·6, 획 보호 마무리) + 마스크 확인. 획 보호에 밝은 선 추가
    "5b": {"folder": "step5b_finish", "name": "finish", "candidates": step5b_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_spot, "time_n": 36,
           "label": lambda p: f"{p['prev']}_fh{p['final_h']}" + ("_protect" if p["protect"] else "") +
                              ("_nospot" if p["spot_pos"] == "none" else "")},
    # 5단계-③: 5b 채택(점 제거 하는 것) → dark·color 기준만 바꿈. 결과 시트 + 같은 배치의 마스크 시트(mask_apply)
    "5c": {"folder": "step5c_spotdetect", "name": "spotdetect", "candidates": step5c_candidates,
           "prepare": lambda bgr: bgr, "apply": apply_spot, "mask_apply": mask_of, "time_n": 36,
           "label": lambda p: f"{p['prev']}_dark{p['dark']}_col{p['color']}"},
    # 6단계: Q4(clip 3) → 채널에 색 특징 더하기 (없음 / 양방향). 적용할 때 원본|결과 나란히 그림도 '_view' 폴더에 저장
    "6": {"folder": "step6_addcolor", "name": "addcolor", "candidates": step6_candidates,
          "prepare": lambda bgr: bgr, "apply": apply6, "time_n": 36, "view": True,
          "label": lambda p: ("auto" if not p.get("add_alpha") else
                             f"{'both' if p['add_both'] else 'one'}_a{p['add_alpha']:g}_t{p['add_thr']:g}") + (f"_link{p['link_len']}_low{p['link_low']}" if p.get("link_len") else "") +
                             (f"_rim{p['link_rim']:g}" if p.get("link_rim") else "")},
}


def step_input(bgr, step):
    """그 단계의 공통 입력 (후보마다 다시 계산하지 않도록 사진당 한 번)."""
    return STEPS[step]["prepare"](bgr)


# ---------------------------------------------------------------------
# 시트 그리기 (OpenCV 글씨는 영문만 가능 → 칸 제목은 번호·설정 영문, 한글 설명은 CSV 에)
# ---------------------------------------------------------------------
def _cell(img, title):
    """그림 한 칸: CELL×CELL 로 맞추고 위에 제목 줄. 흑백은 3채널로 바꾸고, 컬러(C7 등)는 그대로."""
    im = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    bar = np.full((BAR, CELL, 3), 245, np.uint8)
    cv2.putText(bar, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def _text_box(w, h, lines, scale=0.5, bold=1, bg=255):
    """글씨만 있는 칸 (행·열 이름, 블록 제목)."""
    c = np.full((h, w, 3), bg, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(c, t, (6, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), bold, cv2.LINE_AA)
    return c


def _zoom(img):
    """가운데 절반(400 기준 200×200)을 잘라냄 → 칸에 그리면 2배 확대가 됨. 각인은 대부분 알약 가운데에 있음."""
    h, w = img.shape[:2]
    return img[h // 4: h * 3 // 4, w // 4: w * 3 // 4]


def build_sheet(samples, results, cands, zoom):
    """블록(샘플 한 장)마다: 제목 줄 → 'none' 행 → k 행들(열 = σ). 맨 위에 열 이름 줄."""
    row_names = []
    for c in cands:                                   # 행 이름 순서 유지 (none, k=3, k=5 …)
        if c[3] not in row_names:
            row_names.append(c[3])
    col_names = [c[4] for c in cands if c[4] is not None]
    col_names = list(dict.fromkeys(col_names))        # 중복 제거, 순서 유지
    lab_w = 300                                       # 행 이름 칸 폭 ('|' 로 두 줄: 번호 / 설정). 단계가 쌓여 번호가 길어져 300 으로
    width = lab_w + CELL * len(col_names)
    blocks = [np.hstack([_text_box(lab_w, 30, [""]), *[_text_box(CELL, 30, [n], 0.6, 2, 235) for n in col_names]])]
    for it, en, _ in samples:
        title = f"{it['drug_N']}  truth={it['truth']}  rot{it['rot']} cam{it['camera_la']}  [{en}]"
        blocks.append(_text_box(width, 34, [title], 0.65, 2, 225))
        for rname in row_names:
            cells = [_text_box(lab_w, CELL + BAR, ["", "", "", *rname.split("|")], 0.45, 1)]
            row_c = [c for c in cands if c[3] == rname]
            if row_c and row_c[0][4] is None:          # 'none' 행: 첫 칸에만 그림
                img = results[(it["file_name"], row_c[0][0])]
                cells.append(_cell(_zoom(img) if zoom else img, row_c[0][1]))
                cells += [np.full((CELL + BAR, CELL, 3), 255, np.uint8)] * (len(col_names) - 1)
            else:
                for cn in col_names:
                    c = next(x for x in row_c if x[4] == cn)
                    img = results[(it["file_name"], c[0])]
                    fits = cv2.getTextSize(c[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] <= CELL - 8
                    title = c[1] if fits else cn                # 칸 폭을 넘으면 열 이름만 (행 이름에 앞부분 번호가 있음)
                    cells.append(_cell(_zoom(img) if zoom else img, title))
            blocks.append(np.hstack(cells))
        blocks.append(np.full((14, width, 3), 255, np.uint8))   # 블록 사이 여백
    return np.vstack(blocks)


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
def make_sheet(step, items):
    """후보를 샘플 8장에 적용해 시트 2장(전체·확대) + 432장 기준 처리 시간 표."""
    S = STEPS[step]
    d = OUT / S["folder"]
    os.makedirs(d, exist_ok=True)
    samples = pick_samples8(items)
    with open(OUT / "samples8.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "file_name", "drug_N", "dl_name", "truth", "back_color", "condition"])
        for i, (it, en, ko) in enumerate(samples, 1):
            w.writerow([i, it["file_name"], it["drug_N"], it["dl_name"], it["truth"], it["back_color"], ko])
    cands = S["candidates"]()
    results = {}
    for it, _, _ in samples:
        g = step_input(cc.imread(it["image_path"]), step)
        for code, _, p, _, _ in cands:
            results[(it["file_name"], code)] = S["apply"](g, p)
    for zoom, name in ((False, f"sheet_step{step}_{S['name']}.png"), (True, f"sheet_step{step}_{S['name']}_zoom.png")):
        cc.imwrite(str(d / name), build_sheet(samples, results, cands, zoom))
    if S.get("mask_apply"):                                             # 같은 배치의 마스크 시트 (가운데 2배 확대, 빨강·초록이 잘 보이게)
        masks = {}
        for it, _, _ in samples:
            g = step_input(cc.imread(it["image_path"]), step)
            for code, _, p, _, _ in cands:
                masks[(it["file_name"], code)] = S["mask_apply"](g, p)
        cc.imwrite(str(d / f"sheet_step{step}_{S['name']}_mask.png"), build_sheet(samples, masks, cands, True))

    # 처리 시간: 평가용 432장 전부에 적용해 재되 저장은 안 함 (그 단계 연산만, 첫 장 = 워밍업 제외)
    # 느린 단계(time_n 지정)는 약마다 1장꼴로 time_n 장만 잼 (점 제거 432장 × 20칸이면 약 45분이라)
    times = {c[0]: [] for c in cands}
    t_items = items if not S.get("time_n") else items[::max(1, len(items) // S["time_n"])][:S["time_n"]]
    for it in t_items:
        g = step_input(cc.imread(it["image_path"]), step)
        for code, _, p, _, _ in cands:
            t0 = time.perf_counter()
            S["apply"](g, p)
            times[code].append((time.perf_counter() - t0) * 1000)
    with open(d / "candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "params", "ms_mean", "ms_median", "ms_p95", "ms_warmup", "n"])
        for code, _, p, _, _ in cands:
            v = np.array(times[code][1:])
            w.writerow([code, json.dumps(p), round(float(v.mean()), 4), round(float(np.median(v)), 4),
                        round(float(np.percentile(v, 95)), 4), round(times[code][0], 4), len(times[code])])
    if step == "1b":                                                    # 'auto' 채널이 사진마다 무엇을 골랐는지 집계
        chosen = {}
        for it in items:
            n = ps.channel_auto(cc.imread(it["image_path"]))[1]
            chosen[n] = chosen.get(n, 0) + 1
        with open(d / "auto_channel_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["channel", "n_images", "desc"])
            for n, c in sorted(chosen.items(), key=lambda x: -x[1]):
                w.writerow([n, c, ps.CHANNEL_DESC[n]])
        print("  auto 채널이 고른 것 (432장):", chosen)
    print(f"[{step}단계] 시트 저장: {d}")
    print("  샘플 8장:", [f"{it['drug_N']}({ko})" for it, _, ko in samples])
    print("  한 장당 평균 처리 시간(ms):", {c[0]: round(float(np.mean(times[c[0]][1:])), 3) for c in cands})


def apply_picks(step, items, picks, why, force, dest=None):
    """채택한 번호들을 432장에 적용해 폴더에 저장하고 selection_log.csv 에 기록.
       dest: 저장할 폴더(prep_v2 기준 상대 경로, 예: final). 없으면 그 단계 폴더."""
    S = STEPS[step]
    cands = {c[0]: c for c in S["candidates"]()}
    for code in picks:
        if code not in cands:
            raise SystemExit(f"없는 번호: {code} (가능: {', '.join(cands)})")
    for code in picks:
        p = cands[code][2]
        d = OUT / (dest or S["folder"]) / f"{code}_{S['label'](p)}"
        if (d / "params.json").exists() and not force:
            print(f"[건너뜀] {d.name} (이미 있음, 다시 만들려면 --force)")
            continue
        for it in items:
            bgr = cc.imread(it["image_path"])
            out = S["apply"](step_input(bgr, step), p)
            cc.imwrite(str(d / it["file_name"]), out)                # OCR 입력용 (결과만)
            if S.get("view"):                                         # 눈으로 보기용: 왼쪽 원본(400 맞춤) | 오른쪽 결과
                cc.imwrite(str(d.parent / (d.name + "_view") / it["file_name"]),
                           np.hstack([cc.fit_side(bgr, ps.SIDE), out]))
        json.dump({"step": step, "code": code, "params": p, "n_images": len(items), "date": time.strftime("%Y-%m-%d %H:%M")},
                  open(d / "params.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  저장: {d} ({len(items)}장)")
    log = OUT / "selection_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "step", "code", "params", "why"])
        for code in picks:
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), step, code, json.dumps(cands[code][2]), why])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=list(STEPS), help="단계: 1 = 가우시안, 1b = 색 채널(+채택한 가우시안), 2 = 평균 필터, 3a = 히스토그램 clip, 3b = 히스토그램 tile, 4a = 샤프닝 세기, 4b = 샤프닝 반경, 4c = 샤프닝 문턱, 5a = 점 제거 위치, 5b = 점 제거 마무리, 5c = 점 찾기 기준, 6 = 색 특징 더하기")
    ap.add_argument("--pick", default="", help="채택한 번호, 예: G07,G12,G15")
    ap.add_argument("--why", default="", help="채택 이유 (selection_log.csv 에 기록)")
    ap.add_argument("--force", action="store_true", help="이미 있는 채택 폴더도 다시 만듦")
    ap.add_argument("--dest", default=None, help="채택 결과를 저장할 폴더 (prep_v2 기준, 예: final). 없으면 그 단계 폴더")
    a = ap.parse_args()
    items = cc.load_items(cc.EVAL_ROOT)                                  # 평가용 432장 (그림 확인용, 점수 없음)
    if a.pick:
        apply_picks(a.step, items, [x.strip() for x in a.pick.split(",") if x.strip()], a.why, a.force, a.dest)
    else:
        make_sheet(a.step, items)


if __name__ == "__main__":
    main()
