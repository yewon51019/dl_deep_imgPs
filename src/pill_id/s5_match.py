"""5단계: 약 판별 — 읽은 글자와 겉모습 확률을 검색 DB(1,211종)와 비교해 약 순위를 만든다.

  각인 후보   고른 방향: 읽은 글자 + 고른 방향 상자 글자(2~4개)의 모든 순서 / 다른 방향: 나머지 방향의 읽은 글자 (1~30글자)
  각인 점수 T 약마다 앞·뒷면 각인 중 최고 유사도. 유사도 = (L − 편집거리) / (L + 편집거리), 헷갈리는 글자 통일(0OQD, 1IL, 5S, 8B,
              2Z, 6G, UV) 후 계산, 읽은 글자(2자 이상)가 DB 각인 안에 통째로 있으면 max(유사도, 0.5 + 0.5 × 길이 비율).
              T = max(고른 방향 점수, 0.9 × 다른 방향 점수), 0.3 미만은 0
  겉모습 점수 A log P(모양) + log P(제형) + Σ 색 12칸 log(p 또는 1−p) (확률은 0.02~0.98 로 자름)
  최종 점수  T + 0.02 × A. 1등이 동점이면 '판정 없음'
(exp_cns/match_cns.py·match2_cns.py, exp300/correct_exp300.py V3 와 같은 계산)
"""
import csv
import itertools
import math
from functools import lru_cache

from . import config as C
from .s4_appearance import COLORS

CONFUSABLE = ["0OQD", "1IL", "5S", "8B", "2Z", "6G", "UV"]
FOLD = {c: g[0] for g in CONFUSABLE for c in g}
EPS = 1e-9


def load_db(path=C.DB_CSV):
    """DB CSV → 약 dict 목록 (imprints = 비어 있지 않은 앞·뒷면 각인, cols = 색 집합)."""
    db = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    for r in db:
        r["imprints"] = [x for x in (r["front"], r["back"]) if x]
        r["cols"] = set(x for x in r["colors"].split("|") if x)
    return db


def fold(s):
    return "".join(FOLD.get(c, c) for c in s)


@lru_cache(maxsize=None)
def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def similarity(read, imprint):
    """읽은 글자 ↔ DB 각인 유사도 (0~1)."""
    a, b = fold(read), fold(imprint)
    if not a or not b:
        return 0.0
    L, ed = max(len(a), len(b)), levenshtein(a, b)
    s = (L - ed) / (L + ed)
    if 2 <= len(a) < len(b) and a in b:
        s = max(s, 0.5 + 0.5 * len(a) / len(b))
    return s


def candidates(dirs, best, max_len=C.MAX_LEN):
    """(고른 방향 후보, 다른 방향 후보)."""
    main = {best["text"]}
    pieces = [] if best["fallback"] else [t for t in best["texts"] if t]
    if 2 <= len(pieces) <= 4:
        main |= {"".join(p) for p in itertools.permutations(pieces)}
    other = {d["text"] for d in dirs} - main
    keep = lambda c: [x for x in c if 1 <= len(x) <= max_len]
    return keep(main), keep(other)


def text_scores(cands, db):
    out = {}
    for d in db:
        s = max((similarity(c, imp) for c in cands for imp in d["imprints"]), default=0.0)
        if s >= C.MIN_SIM:
            out[d["drug_N"]] = s
    return out


def combine(tm, to, lam=C.LAM):
    """T = max(고른 방향, λ × 다른 방향)."""
    T = dict(tm)
    for d, v in to.items():
        if lam * v >= C.MIN_SIM and lam * v > T.get(d, 0.0):
            T[d] = lam * v
    return T


def attr_score(prob, d):
    cl = lambda x: min(max(x, 0.02), 0.98)
    a = math.log(cl(prob[f"s_{d['shape']}"])) + math.log(cl(prob[f"f_{d['form']}"]))
    for c in COLORS:
        a += math.log(cl(prob[f"c_{c}"]) if c in d["cols"] else 1 - cl(prob[f"c_{c}"]))
    return a


def rank(scores):
    return sorted(scores.items(), key=lambda kv: -kv[1])


def decide(ranked):
    """1등 약 번호 (동점·비어 있음 = '' 판정 없음)."""
    if not ranked or (len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < EPS):
        return ""
    return ranked[0][0]


def match(dirs, best, prob, db):
    """→ (1등 약 번호 또는 '', 점수순 [(약 번호, 점수)] 전체)."""
    cm, co = candidates(dirs, best)
    T = combine(text_scores(cm, db), text_scores(co, db))
    s = {d["drug_N"]: T.get(d["drug_N"], 0.0) + C.W_ATTR * attr_score(prob, d) for d in db}
    ranked = rank(s)
    return decide(ranked), ranked
