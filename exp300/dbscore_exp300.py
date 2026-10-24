# =====================================================================
#  dbscore_exp300.py  —  DB 제약 점수 매기기 (--match dbscore): '자유롭게 읽기' 대신 'DB 각인 중 무엇이 가장 그럴듯한가'
# =====================================================================
#
#  [아이디어]
#   지금은 PARSeq 가 글자를 자유롭게 읽은 뒤(예: 'ARCED') DB 와 편집거리로 비교해요.
#   dbscore 는 거꾸로, DB 에 있는 각인 후보(약 27종 × 앞·뒷면)를 하나씩 PARSeq 에 '정답이라면'하고 넣어서(teacher forcing)
#   그 글자들이 나올 확률(로그확률)을 계산해요. → 모든 방향·모든 후보 중 가장 높은 후보의 약을 예측으로.
#   · 후보 글자 확률은 대문자·소문자를 합쳐서 계산 (PARSeq 가 'a' 로 봐도 'A' 와 같은 글자로)
#   · PARSeq 가 모르는 글자(한글)는 빼고 계산 (예: '명인PR500' → 'PR500' 으로 점수, 기록은 원래 각인)
#   · 점수 2가지: 합계(sum) / 글자 수(+끝 기호)로 나눈 평균(mean). 합계는 짧은 후보가 유리해지는 경향이 있어요.
#   · 방향: 둥근 알약 12방향(30°씩), 길쭉한 알약 0°/180° (지금 방식과 같음)
#   · 박스: CRAFT 박스가 있으면 박스별로, 없으면 알약 전체를 한 장으로
#       max    : 박스마다 점수를 매기고 후보별로 가장 높은 박스 점수
#       concat : 박스들을 읽는 순서(위→아래, 왼→오른)대로 옆으로 이어 붙인 이미지 한 장의 점수
#   → 규칙 4가지 (max_sum, max_mean, concat_sum, concat_mean) 를 개발용에서 비교해요.
#
#  [비교 대상]  같은 개발용 51장의 vote(12방향 다수결), conf(② 신뢰도 최고), 보정 V3(후보 전체 + 부분일치·헷갈리는 글자)
#   vote·conf·V3 는 run_exp300.py --align vote/conf 결과 CSV 를 그대로 씀 (OCR 다시 안 돌림)
#
#  [저장]  --out 폴더 (기존 결과는 덮어쓰지 않도록 새 폴더)
#   dbscore_dev.csv     : 사진별·규칙별 예측 약, 고른 후보 각인·방향·박스 수, 1·2등 후보(서로 다른 약)와 점수 차이
#   dbscore_metrics.csv : 방식별 정답률·정확도·정밀도·민감도·F1 (전체 / 둥근 / 길쭉)
#
#  [실행 예]  (저장소 맨 위 폴더에서, 개발용만)
#     python exp300\dbscore_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db.csv ^
#            --vote exp300\results\pred_dev_parseq_P0_align_vote.csv --conf exp300\results\pred_dev_parseq_P0_align_conf.csv ^
#            --out exp300\results\dbscore
# =====================================================================

import argparse
import csv
import os
import time
from collections import defaultdict

import torch

import cv2
import numpy as np

import correct_exp300 as corr
import run_exp300 as base
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

RULES = ["max_sum", "max_mean", "concat_sum", "concat_mean"]
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def db_candidates(path):
    """search_db.csv → 후보 목록 [{drug_N, face, text}] (norm 적용, 빈 각인 제외)"""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for face, col in (("front", "print_front"), ("back", "print_back")):
                t = base.norm(r.get(col))
                if t:
                    out.append({"drug_N": r["drug_N"], "face": face, "text": t})
    return out


class DBScorer:
    """PARSeq 에 후보 글자를 teacher forcing 으로 넣어 로그확률 계산"""
    def __init__(self, parseq, texts):
        self.p, self.m, self.dev = parseq, parseq.m, parseq.dev
        tok = self.m.tokenizer
        self.texts = texts
        # PARSeq 가 모르는 글자(한글 등, 예: '명인PR500')는 빼고 점수를 매김 → 'PR500'
        enc = ["".join(ch for ch in t if ch in tok._stoi) or t[:0] for t in texts]
        tgt = tok.encode(enc, self.dev)                                 # (K, L+2): [B] 글자들 [E] [P]...
        self.tgt_in, self.tgt_out = tgt[:, :-1], tgt[:, 1:]
        self.pad_id, self.eos_id = tok.pad_id, tok.eos_id
        self.valid = (self.tgt_out != self.pad_id)                       # 글자 + [E]
        self.n_tok = self.valid.sum(1).float()
        # 대문자 ↔ 소문자 짝 (없으면 자기 자신 → 아래에서 한 번만 더함)
        stoi = tok._stoi
        alt = torch.arange(len(tok), device=self.dev)
        for ch, i in stoi.items():
            if len(ch) == 1 and ch.isalpha():
                o = ch.lower() if ch.isupper() else ch.upper()
                if o in stoi:
                    alt[i] = stoi[o]
        self.alt = alt
        L = self.tgt_in.shape[1]
        self.mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=self.dev), 1)
        self.pad_mask = (self.tgt_in == self.pad_id) | (self.tgt_in == self.eos_id)

    def images(self, crops):
        from PIL import Image
        return torch.stack([self.p.tf(Image.fromarray(c).convert("RGB")) for c in crops]).to(self.dev)

    @torch.no_grad()
    def score(self, crops):
        """흑백 이미지 B장 → (합계 로그확률 (B,K), 평균 로그확률 (B,K))"""
        x = self.images(crops)
        mem = self.m.model.encode(x)                                    # (B, S, D)
        B, K, L = mem.shape[0], len(self.texts), self.tgt_in.shape[1]
        memK = mem.repeat_interleave(K, 0)
        tin = self.tgt_in.repeat(B, 1)
        out = self.m.model.decode(tin, memK, self.mask, self.pad_mask.repeat(B, 1),
                                  tgt_query=self.m.model.pos_queries[:, :L].expand(B * K, -1, -1), tgt_query_mask=self.mask)
        lp = self.m.model.head(out).log_softmax(-1)                      # (B*K, L, C)
        tout = self.tgt_out.repeat(B, 1)
        tout = torch.where(tout == self.pad_id, torch.full_like(tout, self.eos_id), tout)   # [P] 는 출력 칸에 없음 (어차피 아래에서 0 처리)
        a = lp.gather(-1, tout[..., None])[..., 0]
        alt_id = self.alt[tout]
        b = lp.gather(-1, alt_id[..., None])[..., 0]
        tok_lp = torch.where(alt_id != tout, torch.logaddexp(a, b), a)  # 대·소문자 합치기
        tok_lp = tok_lp * self.valid.repeat(B, 1)
        s = tok_lp.sum(1).view(B, K)
        return s.cpu().numpy(), (s / self.n_tok[None]).cpu().numpy()


def ordered_boxes(boxes):
    """join_boxes 와 같은 읽는 순서 (위→아래 줄, 줄 안에서 왼→오른)"""
    if not boxes:
        return []
    ys = [b[1] for b in boxes]
    gap = max(10, (max(ys) - min(ys)) / 3 if len(ys) > 1 else 10)
    return sorted(boxes, key=lambda b: (round(b[1] / gap), b[0]))


def hconcat(crops, h=32, gap=6):
    """잘라낸 박스들을 높이 h 로 맞춰 옆으로 이어 붙임 (사이 빈칸은 각 박스의 중간 밝기)"""
    parts = []
    for c in crops:
        s = h / c.shape[0]
        r = cv2.resize(c, (max(4, int(round(c.shape[1] * s))), h), interpolation=cv2.INTER_AREA)
        parts += [r, np.full((h, gap), int(np.median(r)), np.uint8)]
    return np.hstack(parts[:-1])


def score_image(scorer, g0, aspect, step=30):
    """한 사진의 모든 방향 × 규칙 점수 → {규칙: (K,) 최고 점수, 그 방향, 박스 수}"""
    angles = [0, 180] if aspect > 1.3 else list(range(0, 360, step))
    best = {r: (np.full(len(scorer.texts), -np.inf), np.zeros(len(scorer.texts)), np.zeros(len(scorer.texts), int))
            for r in RULES}
    for a in angles:
        r = base.rotate(g0, a)
        boxes = ordered_boxes(base.craft_boxes(r))
        crops = [r[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes] or [r]
        imgs = crops + ([hconcat(crops)] if len(crops) > 1 else [])
        s_sum, s_mean = scorer.score(imgs)
        n = len(crops)
        vals = {"max_sum": s_sum[:n].max(0), "max_mean": s_mean[:n].max(0),
                "concat_sum": s_sum[-1], "concat_mean": s_mean[-1]}           # 박스 1개(또는 없음)면 concat = 그 1장
        for rule, v in vals.items():
            cur, ang, nb = best[rule]
            upd = v > cur
            cur[upd], ang[upd], nb[upd] = v[upd], a, len(boxes)
    return best


def top2(cands, scores):
    """1등 후보 + 1등과 '다른 약'인 2등 후보"""
    order = np.argsort(-scores)
    i1 = order[0]
    i2 = next((i for i in order[1:] if cands[i]["drug_N"] != cands[i1]["drug_N"]), None)
    return i1, i2


def load_pred(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


def scores(rows, imprint=True):
    acc, p, r, f = drug_metrics([x["truth_drug"] for x in rows], [x["pred_drug"] for x in rows])
    imp = float(np.mean([x["judge"] == "O" for x in rows])) if imprint else float("nan")
    return [imp, acc, p, r, f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--vote", required=True, help="run_exp300.py --align vote 결과 (개발용)")
    ap.add_argument("--conf", required=True, help="run_exp300.py --align conf 결과 (개발용)")
    ap.add_argument("--out", default="dbscore")
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                     # 개발용만 (평가용은 돌리지 않음)
    if a.limit:
        items = items[: a.limit]
    cands = db_candidates(a.db)
    parseq = base.PARSeq()
    scorer = DBScorer(parseq, [c["text"] for c in cands])
    print(f"사진 {len(items)}장 (dev), DB 후보 {len(cands)}개 (약 {len({c['drug_N'] for c in cands})}종 × 앞·뒷면)")

    vote, conf = load_pred(a.vote), load_pred(a.conf)
    cdb = corr.load_db(a.db)
    rows = []
    for k, it in enumerate(items, 1):
        fn = it["file_name"]
        gray, aspect, color, ok = base.crop_pill(base.imread_any(it["image_path"]))
        g0 = base.PREPS["P0"][1](gray)
        shape = base.shape_group(aspect)
        truth = base.norm(it["truth"])
        common = {"file_name": fn, "shape": shape, "camera_la": it.get("camera_la", ""), "truth_drug": it["drug_N"],
                  "truth_imprint": truth}
        t0 = time.time()
        best = score_image(scorer, g0, aspect, a.step)
        ms = (time.time() - t0) * 1000
        for rule in RULES:
            sc, ang, nb = best[rule]
            i1, i2 = top2(cands, sc)
            c1 = cands[i1]
            rows.append({**common, "method": f"dbscore_{rule}", "pred_drug": c1["drug_N"], "pred_imprint": c1["text"],
                         "pred_face": c1["face"], "judge": judge(c1["text"], truth), "angle": int(ang[i1]),
                         "n_boxes": int(nb[i1]), "score1": round(float(sc[i1]), 3),
                         "cand2_drug": cands[i2]["drug_N"] if i2 is not None else "",
                         "cand2_imprint": cands[i2]["text"] if i2 is not None else "",
                         "score2": round(float(sc[i2]), 3) if i2 is not None else "",
                         "gap": round(float(sc[i1] - sc[i2]), 3) if i2 is not None else "", "ms": round(ms)})
        for name, src in (("vote", vote), ("conf", conf)):
            r = src[fn]
            rows.append({**common, "method": name, "pred_drug": r["pred_edit"] or NONE_LABEL,
                         "pred_imprint": r["pred_imprint"], "judge": judge(r["pred_imprint"], truth),
                         "angle": r.get("angle", "")})
        v3 = corr.predict([vote[fn]], cdb, "V3", 1.0)
        rows.append({**common, "method": "V3", "pred_drug": v3 or NONE_LABEL, "pred_imprint": "", "judge": "-"})
        d = {x["method"]: x for x in rows[-(len(RULES) + 3):]}
        print(f"  {k:2d}/{len(items)} {shape} {truth:<14} vote={d['vote']['pred_imprint'] or '-'}"
              f"({'OK' if d['vote']['pred_drug'] == it['drug_N'] else 'NG'})  " +
              "  ".join(f"{r.split('_', 1)[1] if '_' in r else r}={d['dbscore_' + r]['pred_imprint']}"
                        f"({'OK' if d['dbscore_' + r]['pred_drug'] == it['drug_N'] else 'NG'})" for r in RULES) +
              f"  ({ms / 1000:.1f}s)")

    with open(os.path.join(a.out, "dbscore_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

    methods = [("vote", "12방향 다수결 (지금, 보정 전)"), ("conf", "② 신뢰도 최고 (보정 전)"),
               ("V3", "보정 V3 (후보 전체+부분일치)")] + [(f"dbscore_{r}", f"dbscore {r}") for r in RULES]
    met = []
    for gname, cond in (("전체", lambda x: True), ("둥근", lambda x: x["shape"] == "둥근"),
                        ("길쭉", lambda x: x["shape"] == "길쭉")):
        n = len({x["file_name"] for x in rows if cond(x)})
        if not n:
            continue
        print(f"\n[{gname} {n}장]  (1장 = {100 / n:.1f}%p)")
        print(f"  {'방식':<30} " + " ".join(f"{m:>8}" for m in METRICS))
        for m, label in methods:
            sub = [x for x in rows if x["method"] == m and cond(x)]
            v = scores(sub, imprint=(m != "V3"))
            print(f"  {label:<30} " + " ".join("     -  " if np.isnan(s) else f"{s * 100:7.1f}%" for s in v))
            met.append({"group": gname, "n": n, "method": m, **{k: ("" if np.isnan(s) else round(s * 100, 1))
                                                                 for k, s in zip(METRICS, v)}})
    with open(os.path.join(a.out, "dbscore_metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(met[0]))
        w.writeheader()
        w.writerows(met)
    print("\n※ dbscore 는 언제나 DB 안의 약 하나를 고르므로 '판정 없음'이 없어요 (vote·V3 는 유사도 0.3 미만·동점이면 판정 없음).")
    print("※ V3 는 약만 고르는 방식이라 각인 정답률이 없어요 (-).")
    print(f"저장 완료: {a.out}  (dbscore_dev.csv, dbscore_metrics.csv)")


if __name__ == "__main__":
    main()
