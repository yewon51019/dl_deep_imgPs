"""6단계: 결과 표시 — 사용자에게 약 몇 개를 보여 줄지 정한다 (정밀도 우선: 확실할 때만 약 1개).

  확실   답 있음, 1등 점수 ≥ 0.95, 1등 − 2등 ≥ 0.1          → 약 1개
  동점   1등 동점 (판정 없음)                              → 1등과 동점인 약 전부
  불확실 그 밖                                             → 1등 + 2등 + 2등과 동점인 약
  보여 줄 후보가 5개 넘으면 '다른 면·다시 찍기 요청'
(exp_cns/display_cns.py·final_wrap_cns.py 와 같은 규칙)
"""
from . import config as C

EPS = 1e-9


def shown(pred, ranked, top_min=C.SURE_TOP, gap_min=C.SURE_GAP):
    """→ (종류 'sure'/'tie'/'unsure', 보여 줄 약 번호 목록)."""
    s1, s2 = ranked[0][1], ranked[1][1]
    if not pred:
        return "tie", [d for d, v in ranked if v >= s1 - EPS]
    if s1 >= top_min and s1 - s2 >= gap_min:
        return "sure", [pred]
    return "unsure", [d for d, v in ranked if v >= s2 - EPS]


def flow(kind, shown_list, cap=C.CAP):
    """표시 문구."""
    if kind == "sure":
        return "약 1개 (확실)"
    if len(shown_list) <= cap:
        return f"후보 {len(shown_list)}개 표시"
    return f"후보 {len(shown_list)}개 → 다른 면·다시 찍기 요청"
