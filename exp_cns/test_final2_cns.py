# =====================================================================
#  test_final2_cns.py  —  최종 평가 (CLAUDE.md 10장 3-21 사전 기록 그대로, 한 번만)
# =====================================================================
#
#  [평가용]  pill_image_testset 432장 중 정답 각인이 있는 사진 (--split test). 카메라 75 / 90
#  [주 방식]  pft1 + TF3 (g 0.3) + 표시 D1
#  [비교용]  base (사전학습 PARSeq, λ 0.95, w 0.02) · pft1 (TF 없음) · pft1 + R2 · pft1 + TF3 g 1.0
#            → 숫자는 모두 미리 정한 값, 평가용으로 고르는 것 없음
#  [필요]    probs_test.csv, ocr_v1\test\paddle_ft1_u2.0\orig_alignlong_line 과 ..._line_pft1 의 pred.csv
#  [저장]    exp_cns\results\test_final_pillset\  summary.csv, per_drug.csv, pred_test.csv, tf_scores_test.npz
#            summary.csv 가 이미 있으면 멈춤 (한 번만). tf 점수는 끊겨도 이어 감
#  [실행]    python exp_cns\test_final2_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv

import numpy as np

import align_long_cns as al
import common_cns as cc
import display_cns as dc
import match2_cns as m2
import match7_cns as m7
import match_cns as m1
import ocr_cns as ocr
import ocr_common_cns as oc
import rerank_tf_cns as rt
from test_final_cns import macro

OUT = cc.RESULTS / "test_final_pillset"
READ_DIR = oc.OCR_OUT / "test" / rt.DET
READ = {"base": READ_DIR / "orig_alignlong_line" / "pred.csv", "pft1": READ_DIR / "orig_alignlong_line_pft1" / "pred.csv"}
MAX_LEN, W = 30, 0.02
LAM = {"base": 0.95, "pft1": 0.9}
GAP_MAIN, GAP_REF = 0.3, 1.0
SAVE_EVERY = 50
DEV_ACC = {"main": 87.8, "base": 87.2, "pft1": 87.4, "r2": 88.2, "tf3_1": None}
NAMES = {"main": "주 방식 pft1 + TF3 0.3", "base": "이전 주 방식 base", "pft1": "pft1", "r2": "pft1 + R2", "tf3_1": "pft1 + TF3 1.0"}


def load_probs():
    return {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
            for x in csv.DictReader(open(m1.ATTR / "probs_test.csv", encoding="utf-8-sig"))}


def base_scores(tag, db, probs):
    """읽기 파일 → 사진마다 (OCR 줄, 답, 점수순)."""
    out = []
    for r in csv.DictReader(open(READ[tag], encoding="utf-8-sig")):
        A = {d["drug_N"]: m1.attr_score(probs[r["file_name"]], d) for d in db}
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, False), m2.text_scores(co, db, False), LAM[tag])
        s = {d: T.get(d, 0.0) + W * A[d] for d in A}
        out.append((r, m1.decide(s)[0], m1.rank(s)))
    return out


def r2_answers(rows, db, probs):
    """pft1 + R2 (match7_cns 의 정의 그대로)."""
    out = {}
    for r, _, _ in rows:
        A = {d["drug_N"]: m1.attr_score(probs[r["file_name"]], d) for d in db}
        T = m7.total_T(m7.scored_sets(r, db), m7.LAM, True)
        out[r["file_name"]] = m1.decide({d: T.get(d, 0.0) + W * A[d] for d in A})[0]
    return out


def tf_scores(rows, db):
    """판정 없음 사진의 TF 점수 (사진 × 각인). 끊기면 이어 감."""
    path = OUT / "tf_scores_test.npz"
    names, mat = [], []
    reader = ocr.Reader(str(rt.WEIGHTS))
    tf = rt.Tf(reader.ps, sorted({x for d in db for x in d["imprints"]}))
    if path.exists():
        z = np.load(path, allow_pickle=False)
        assert list(z["texts"]) == tf.texts
        names, mat = list(z["names"]), list(z["scores"])
    done = set(names)
    items = {it["file_name"]: it for it in oc.load_split("test", 0)}
    prep = oc.Prep("orig", "test")
    saved = ocr.SavedBoxes(oc.out_dir("test", rt.DET, "orig", 0, 12, "long") / "boxes.json")
    assert saved.d is not None
    for k, (r, pred, _) in enumerate(rows, 1):
        fn = r["file_name"]
        if pred or fn in done:
            continue
        bgr = cc.imread(items[fn]["image_path"])
        img, _ = prep(items[fn], bgr)
        angs, _ = al.align_angles(bgr)
        mat.append(rt.photo_scores(tf, img, fn, angs, len(angs) == 2, saved))
        names.append(fn)
        if len(names) % SAVE_EVERY == 0:
            np.savez_compressed(path, texts=np.array(tf.texts), names=np.array(names), scores=np.stack(mat))
            print(f"  TF {len(names)}장", flush=True)
    np.savez_compressed(path, texts=np.array(tf.texts), names=np.array(names), scores=np.stack(mat) if mat else np.zeros((0, len(tf.texts))))
    return {t: i for i, t in enumerate(tf.texts)}, dict(zip(names, mat))


def tf_answers(rows, idx, scores, db, probs, gap_min):
    """TF3: 판정 없음이고 각인 1·2등 차이 ≥ gap_min 이면 TF 답. → {사진: 답}"""
    out = {}
    for r, pred, _ in rows:
        fn = r["file_name"]
        if not pred and fn in scores:
            top2 = np.sort(scores[fn])[-2:]
            if float(top2[1] - top2[0]) >= gap_min:
                new = rt.tf_answer(scores[fn], idx, probs[fn], db)[0]
                if new:
                    out[fn] = new
    return out


def line(name, rows, answers, tf3=None, display=None, split_flags=None):
    n = len(rows)
    pct = lambda v, m=n: round(100 * v / max(m, 1), 1)
    hit = [answers[r["file_name"]] == r["drug_N"] for r, _, _ in rows]
    none = sum(answers[r["file_name"]] == "" for r, _, _ in rows)
    wrong = n - sum(hit) - none
    group = lambda f: [h for (r, _, _), h in zip(rows, hit) if f(r)]
    by = collections.defaultdict(list)
    for (r, _, _), h in zip(rows, hit):
        by[r["drug_N"]].append(h)
    mp, mr, mf = macro([(r["drug_N"], answers[r["file_name"]]) for r, _, _ in rows], sorted(by))
    out = {"method": name, "n": n, "acc": pct(sum(hit)), "hit": sum(hit), "none": none, "wrong": wrong,
           "macro_P": mp, "macro_R": mr, "macro_F1": mf, "drug_acc": round(100 * sum(sum(v) * 2 > len(v) for v in by.values()) / len(by), 1),
           "long": pct(sum(group(lambda r: r["aligned"] == "1")), len(group(lambda r: r["aligned"] == "1"))),
           "other": pct(sum(group(lambda r: r["aligned"] != "1")), len(group(lambda r: r["aligned"] != "1"))),
           "cam75": pct(sum(group(lambda r: r["camera_la"] == "75")), len(group(lambda r: r["camera_la"] == "75"))),
           "cam90": pct(sum(group(lambda r: r["camera_la"] == "90")), len(group(lambda r: r["camera_la"] == "90"))),
           "read_exact": pct(sum(r["correct"] == "1" for r, _, _ in rows))}
    if tf3 is not None:
        truth = {r["file_name"]: r["drug_N"] for r, _, _ in rows}
        ok = sum(truth[f] == a for f, a in tf3.items())
        out.update(tf3_answered=len(tf3), tf3_right=ok, tf3_precision=pct(ok, len(tf3)))
    if display is not None:
        has = sum(r["drug_N"] in display[r["file_name"]] for r, _, _ in rows)
        sizes = [len(display[r["file_name"]]) for r, _, _ in rows]
        out.update(include_truth=pct(has), mean_shown=round(sum(sizes) / n, 2), max_shown=max(sizes))
    return out


def d1_lists(rows, tf3):
    """표시 D1: display_cns 규칙 + TF3 답을 맨 앞에."""
    out = {}
    for r, pred, ranked in rows:
        _, sh = dc.shown(pred, ranked, dc.SURE_TOP, dc.SURE_GAP)
        a = tf3.get(r["file_name"])
        out[r["file_name"]] = ([a] + [d for d in sh if d != a]) if a else sh
    return out


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음 (최종 평가는 한 번만): {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    probs = load_probs()
    rows_b, rows_p = base_scores("base", db, probs), base_scores("pft1", db, probs)
    assert {r["file_name"] for r, _, _ in rows_b} == {r["file_name"] for r, _, _ in rows_p}
    print(f"평가 사진 {len(rows_p)}장, 약 {len({r['drug_N'] for r, _, _ in rows_p})}종", flush=True)
    idx, scores = tf_scores(rows_p, db)
    ans_b = {r["file_name"]: p for r, p, _ in rows_b}
    ans_p = {r["file_name"]: p for r, p, _ in rows_p}
    tf_main = tf_answers(rows_p, idx, scores, db, probs, GAP_MAIN)
    tf_ref = tf_answers(rows_p, idx, scores, db, probs, GAP_REF)
    ans_main, ans_ref = {**ans_p, **tf_main}, {**ans_p, **tf_ref}
    ans_r2 = r2_answers(rows_p, db, probs)
    table = [line(NAMES["main"], rows_p, ans_main, tf_main, d1_lists(rows_p, tf_main)),
             line(NAMES["base"], rows_b, ans_b, None, d1_lists(rows_b, {})),
             line(NAMES["pft1"], rows_p, ans_p, None, d1_lists(rows_p, {})),
             line(NAMES["r2"], rows_p, ans_r2),
             line(NAMES["tf3_1"], rows_p, ans_ref, tf_ref, d1_lists(rows_p, tf_ref))]
    for t, k in zip(table, ("main", "base", "pft1", "r2", "tf3_1")):
        t["dev_acc"] = DEV_ACC[k]
    keys = list(dict.fromkeys(k for t in table for k in t))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(t)
    name = {d["drug_N"]: d["name"] for d in db}
    with open(OUT / "pred_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "drug_N", "truth", "read_pft1", "read_base", "main", "base", "pft1", "r2", "tf3_1", "tf3_used", "camera", "aligned"])
        rb = {r["file_name"]: r for r, _, _ in rows_b}
        for r, _, _ in rows_p:
            fn = r["file_name"]
            w.writerow([fn, r["drug_N"], r["truth"], r["read"], rb[fn]["read"], ans_main[fn], ans_b[fn], ans_p[fn], ans_r2[fn], ans_ref[fn], int(fn in tf_main), r["camera_la"], r["aligned"]])
    by = collections.defaultdict(list)
    for r, _, _ in rows_p:
        by[r["drug_N"]].append(r)
    with open(OUT / "per_drug.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["drug_N", "name", "photos", "correct", "none", "wrong", "read_exact", "truths", "wrong_preds"])
        for d in sorted(by):
            v = by[d]
            a = [ans_main[r["file_name"]] for r in v]
            w.writerow([d, name[d], len(v), a.count(d), a.count(""), sum(x not in ("", d) for x in a), sum(r["correct"] == "1" for r in v),
                        "|".join(sorted({r["truth"] for r in v})), "|".join(sorted({name.get(x, x)[:16] for x in a if x not in ("", d)}))])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
