# =====================================================================
#  rerank_tf_cns.py  —  불확실한 사진을 teacher forcing 점수로 다시 고르기 (exp300 dbscore 의 concat_mean 방식, 읽기 pft1)
# =====================================================================
#
#  [왜]    틀린 사진 대부분은 어느 방향에서도 정답 각인을 못 읽음 → 읽은 글자 대신 사진을 다시 보고 'DB 각인이 나올 확률' 을 잼
#  [점수]  --step score (GPU): DB 의 모든 각인 × 모든 방향. 방향마다 저장된 상자(ft1_u2.0)를 읽는 순서대로 이어 붙인 조각 한 장
#          (상자 없으면 사진 전체), 후보 글자들의 글자당 평균 로그확률 (대·소문자 합침), 각인별로 방향 중 최고 점수
#          PARSeq 가 모르는 글자는 빼고 계산, 남는 글자가 없거나 25자 넘는 각인은 후보에서 뺌
#  [규칙]  --step rules (실행 전 고정): 기준 = pft1 (λ 0.9, w 0.02)
#     불확실 = display_cns 규칙 (1등 < 0.95 또는 1·2등 차이 < 0.1, 동점 포함). 확실한 사진의 답은 그대로
#     TF 답 = 약별 max(각인 점수) 1등. 같은 점수면 겉모습 점수 A, 그래도 같으면 판정 없음
#     TF1 = 판정 없음 사진에만 TF 답 / TF2 = 불확실한 사진 전체에 TF 답
#  [채택 규칙, 실행 전 고정]  기준 대비 (좋아짐 − 나빠짐) 이 검증 +5장 이상, 개발 0장 이상. 여럿이면 검증 맞은 수 최고
#  [대상]  검증 828장 + 개발 468장 중 불확실한 사진. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\rerank_tf_v1\
#            tf_scores_<part>.npz (사진 × 각인 점수, 50장마다 저장해 끊겨도 이어 감, .gitignore), summary.csv, changed.csv, pred_<part>.csv
#          --limit 시험은 try\ 아래에
#  [실행]  python exp_cns\rerank_tf_cns.py --step score --limit 3
#          python exp_cns\rerank_tf_cns.py --step score
#          python exp_cns\rerank_tf_cns.py --step rules
# =====================================================================

import argparse
import csv
import time

import torch
import cv2
import numpy as np

import align_long_cns as al
import common_cns as cc
import display_cns as dc
import match7_cns as m7
import match_cns as m1
import ocr_cns as ocr
import ocr_common_cns as oc
from dbscore_exp300 import DBScorer, ordered_boxes

RES = cc.RESULTS
OUT = RES / "rerank_tf_v1"
WEIGHTS = next((RES / "parseq_ft1_sure").rglob("best.pt"))
DET = "paddle_ft1_u2.0"
SPLIT = {"val": "trainval", "dev": "dev"}
BOX_SPLIT = {"val": "trainall", "dev": "dev"}                        # trainval 상자는 trainall 에 저장돼 있음 (ocr_cns 와 같음)
CHUNK, MAX_LABEL, SAVE_EVERY = 256, 25, 50
BETTER_MIN, DEV_MIN = 5, 0
EPS = 1e-6


class Tf:
    """DB 각인 전체의 teacher forcing 점수. 글자 수가 비슷한 후보끼리 CHUNK 개씩 묶어 패딩 계산을 줄임."""

    def __init__(self, ps, texts):
        self.ps = ps
        known = ps.m.tokenizer._stoi
        enc = {t: "".join(c for c in t if c in known) for t in texts}
        self.texts = sorted((t for t in texts if 1 <= len(enc[t]) <= MAX_LABEL), key=lambda t: (len(enc[t]), t))
        self.dropped = sorted(set(texts) - set(self.texts))
        self.parts = [DBScorer(ps, self.texts[i:i + CHUNK]) for i in range(0, len(self.texts), CHUNK)]

    @torch.no_grad()
    def score(self, rgb):
        """조각 한 장 → (K,) 글자당 평균 로그확률 (DBScorer.score 와 같은 계산, 사진 인코딩은 한 번만)."""
        x = self.parts[0].images([rgb])
        model = self.ps.m.model
        mem = model.encode(x)
        out = []
        for s in self.parts:
            K, L = len(s.texts), s.tgt_in.shape[1]
            d = model.decode(s.tgt_in, mem.expand(K, -1, -1), s.mask, s.pad_mask,
                             tgt_query=model.pos_queries[:, :L].expand(K, -1, -1), tgt_query_mask=s.mask)
            lp = model.head(d).log_softmax(-1)
            tout = torch.where(s.tgt_out == s.pad_id, torch.full_like(s.tgt_out, s.eos_id), s.tgt_out)
            a = lp.gather(-1, tout[..., None])[..., 0]
            alt = s.alt[tout]
            b = lp.gather(-1, alt[..., None])[..., 0]
            tok = torch.where(alt != tout, torch.logaddexp(a, b), a) * s.valid
            out.append(tok.sum(1) / s.n_tok)
        return torch.cat(out).cpu().numpy()


def line_order(boxes):
    """join_line 과 같은 순서 (글자 자리에 상자 번호를 넣어 돌려받음)."""
    text, _ = ocr.join_line([(*b, chr(0xE000 + i), 1.0) for i, b in enumerate(boxes)])
    return [boxes[ord(c) - 0xE000] for c in text]


def hconcat(crops, h=32, gap=6):
    """상자 조각들을 높이 h 로 맞춰 옆으로 이어 붙임 (dbscore_exp300.hconcat 의 컬러판, 사이는 각 조각의 중간 색)."""
    parts = []
    for c in crops:
        r = cv2.resize(c, (max(4, round(c.shape[1] * h / c.shape[0])), h), interpolation=cv2.INTER_AREA)
        parts += [r, np.full((h, gap, 3), np.median(r.reshape(-1, 3), axis=0), np.uint8)]
    return np.hstack(parts[:-1])


def photo_scores(tf, img, fn, angs, aligned, saved):
    best = np.full(len(tf.texts), -np.inf, np.float32)
    for ang in angs:
        rot = oc.rotate_color(img, ang)
        boxes = [b for b in saved.get(fn, ang)[0] if b[3] > b[1] and b[2] > b[0]]
        boxes = line_order(boxes) if aligned else ordered_boxes(boxes)
        crops = [rot[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes] or [rot]
        one = crops[0] if len(crops) == 1 else hconcat(crops)
        best = np.maximum(best, tf.score(cv2.cvtColor(one, cv2.COLOR_BGR2RGB)))
    return best


def step_score(limit):
    out = OUT / "try" if limit else OUT
    out.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    reader = ocr.Reader(str(WEIGHTS))
    tf = Tf(reader.ps, sorted({x for d in db for x in d["imprints"]}))
    print(f"후보 각인 {len(tf.texts)}개, 뺀 각인 {len(tf.dropped)}개: {tf.dropped[:20]}", flush=True)
    for part in ("dev", "val"):
        path = out / f"tf_scores_{part}.npz"
        done, names, rows = set(), [], []
        if path.exists():
            z = np.load(path, allow_pickle=False)
            assert list(z["texts"]) == tf.texts
            names, rows = list(z["names"]), list(z["scores"])
            done = set(names)
        items = {it["file_name"]: it for it in oc.load_split(SPLIT[part], 0)}
        prep = oc.Prep("orig", SPLIT[part])
        saved = ocr.SavedBoxes(oc.out_dir(BOX_SPLIT[part], DET, "orig", 0, 12, "long") / "boxes.json")
        assert saved.d is not None
        targets = [(r, pred) for r, pred, ranked in dc.score("pft1", part, db)
                   if dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)[0] != "sure"]
        print(f"{part}: 불확실 {len(targets)}장, 이미 {len(done)}장", flush=True)
        if limit:
            targets = targets[:: max(1, len(targets) // limit)][:limit]
        t0, n = time.time(), 0
        for r, _ in targets:
            fn = r["file_name"]
            if fn in done:
                continue
            bgr = cc.imread(items[fn]["image_path"])
            img, _ = prep(items[fn], bgr)
            angs, _ = al.align_angles(bgr)
            aligned = len(angs) == 2
            assert int(aligned) == int(r["aligned"]), fn
            if limit:                                                # 시험 때만: 저장된 읽기와 같은 사진·상자·가중치인지 확인
                dirs = ocr.read_all(img, fn, angs, DET, reader, saved, "line" if aligned else "tb")
                assert ocr.pick_best(dirs)["text"] == r["read"], (fn, ocr.pick_best(dirs)["text"], r["read"])
            rows.append(photo_scores(tf, img, fn, angs, aligned, saved))
            names.append(fn)
            n += 1
            if n % SAVE_EVERY == 0:
                np.savez_compressed(path, texts=np.array(tf.texts), names=np.array(names), scores=np.stack(rows))
                print(f"  {part} {len(names)}/{len(targets)}  장당 {(time.time() - t0) / n:.1f}초", flush=True)
        if rows:
            np.savez_compressed(path, texts=np.array(tf.texts), names=np.array(names), scores=np.stack(rows))
        print(f"{part} 끝: {len(names)}장, 장당 {(time.time() - t0) / max(n, 1):.1f}초", flush=True)


def tf_answer(row, idx, prob, db):
    """(TF 답, 정답 확인용 약별 점수). 약 점수 = 각인 점수 최고, 같은 점수면 겉모습 A."""
    s = {d["drug_N"]: max(row[idx[x]] for x in d["imprints"] if x in idx) for d in db if any(x in idx for x in d["imprints"])}
    top = max(s.values())
    same = [d for d in db if d["drug_N"] in s and s[d["drug_N"]] >= top - EPS]
    return (same[0]["drug_N"] if len(same) == 1 else m1.decide({d["drug_N"]: m1.attr_score(prob, d) for d in same})[0]), s


def step_rules():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    db = m1.load_db()
    table, changed = [], []
    for part in ("val", "dev"):
        z = np.load(OUT / f"tf_scores_{part}.npz", allow_pickle=False)
        idx = {t: i for i, t in enumerate(z["texts"])}
        scores = dict(zip(z["names"], z["scores"]))
        probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
                 for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
        out, unsure, top1, top3 = [], 0, 0, 0
        for r, pred, ranked in dc.score("pft1", part, db):
            fn = r["file_name"]
            res = {"base": (pred, ranked[:3]), "TF1": (pred, ranked[:3]), "TF2": (pred, ranked[:3])}
            if dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)[0] != "sure":
                new, s = tf_answer(scores[fn], idx, probs[fn], db)
                unsure += 1
                top1 += new == r["drug_N"]
                me = s.get(r["drug_N"], -np.inf)
                top3 += sum(v > me + EPS for v in s.values()) < 3
                res["TF2"] = (new, ranked[:3])
                if not pred:
                    res["TF1"] = (new, ranked[:3])
            out.append((r, res))
        print(f"{part}: 불확실 {unsure}장, TF 1등이 정답 {top1}장, 3등 안 {top3}장")
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for name in ("base", "TF1", "TF2"):
                table.append({"part": part, "group": gname, "config": name, **m7.measure(its, name)})
        for r, res in out:
            for name in ("TF1", "TF2"):
                k = m7.kind(r["drug_N"], res["base"][0], res[name][0])
                if k:
                    changed.append({"config": name, "part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                    "change": k, "pred_base": res["base"][0], "pred_new": res[name][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "base_pred", "TF1_pred", "TF2_pred"])
            for r, res in out:
                w.writerow([r["file_name"], r["drug_N"], r["truth"], r["read"], res["base"][0], res["TF1"][0], res["TF2"][0]])
    m7.write_csv(OUT / "summary.csv", table)
    m7.write_csv(OUT / "changed.csv", changed)
    for t in table:
        print(t)
    get = lambda part, name: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == name)
    passed = []
    for name in ("TF1", "TF2"):
        v, d = get("val", name), get("dev", name)
        vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
        ok = vd >= BETTER_MIN and dd >= DEV_MIN
        print(f"{name}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}) → {'통과' if ok else '미달'}")
        if ok:
            passed.append((v["hit"], name))
    print("채택:", max(passed)[1] if passed else "없음 (기준 유지)")
    print(f"저장: {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["score", "rules"])
    ap.add_argument("--limit", type=int, default=0, help="시험 운전: 부분마다 고르게 N 장만 (try\\ 에 저장, 저장된 읽기와 같은지 확인)")
    a = ap.parse_args()
    if a.step == "score":
        step_score(a.limit)
    else:
        step_rules()


if __name__ == "__main__":
    main()
