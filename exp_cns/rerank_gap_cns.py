# =====================================================================
#  rerank_gap_cns.py  —  TF3: 판정 없음 사진에서 teacher forcing 점수 차가 충분히 클 때만 답을 냄 (저장된 점수로 계산, GPU·OCR 안 씀)
# =====================================================================
#
#  [기준]  pft1 (λ 0.9, w 0.02), 저장된 답과 사진마다 같은지 확인 (display_cns.score)
#  [TF3, 실행 전 고정]
#     대상 = 기준이 판정 없음인 사진. 점수는 rerank_tf_v1\tf_scores_<part>.npz 그대로
#     차이 = TF 1등 각인 점수 − 2등 각인 점수 (각인 단위, 글자당 평균 로그확률). 차이 ≥ g 이면 TF 답, 아니면 판정 없음 유지
#     g 는 0.2 / 0.3 / 0.5 / 1.0 중 검증에서 (좋아짐 − 판정없음→틀림) 최대, 같으면 큰 값. 개발용으로 고르지 않음
#  [채택 규칙, 실행 전 고정]  모두 만족해야 함
#     ① (좋아짐 − 나빠짐) 검증 ≥ +5, 개발 ≥ 0
#     ② 검증에서 답을 낸 사진 중 맞은 비율 ≥ 70%
#     ③ 개발에서 틀린 답 증가 ≤ 맞은 수 증가
#  [대상]  검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [저장]  exp_cns\results\rerank_tf_v2\  summary.csv, grid_val.csv, changed.csv, pred_val.csv, pred_dev.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]  python exp_cns\rerank_gap_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import numpy as np

import display_cns as dc
import match7_cns as m7
import match_cns as m1
import rerank_tf_cns as rt

OUT = m1.cc.RESULTS / "rerank_tf_v2"
GAPS = [0.2, 0.3, 0.5, 1.0]
BETTER_MIN, DEV_MIN, PRECISION_MIN = 5, 0, 70.0


def predict(part, db, gaps):
    z = np.load(rt.OUT / f"tf_scores_{part}.npz", allow_pickle=False)
    idx = {t: i for i, t in enumerate(z["texts"])}
    scores = dict(zip(z["names"], z["scores"]))
    probs = {x["file_name"]: {k: float(v) for k, v in x.items() if k[:2] in ("s_", "f_", "c_")}
             for x in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    out = []
    for r, pred, ranked in dc.score("pft1", part, db):
        res = {"base": (pred, ranked[:3]), **{f"TF3_{g:g}": (pred, ranked[:3]) for g in gaps}}
        if not pred:
            row = scores[r["file_name"]]
            top2 = np.sort(row)[-2:]
            gap = float(top2[1] - top2[0])
            new = rt.tf_answer(row, idx, probs[r["file_name"]], db)[0]
            for g in gaps:
                if gap >= g:
                    res[f"TF3_{g:g}"] = (new, ranked[:3])
        out.append((r, res))
    return out


def precision(m):
    answered = m["better"] + m["none_to_wrong"]
    return round(100 * m["better"] / answered, 1) if answered else 0.0


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    val = predict("val", db, GAPS)
    grid = []
    for g in GAPS:
        m = m7.measure(val, f"TF3_{g:g}")
        grid.append({"gap": g, **m, "answered_precision": precision(m)})
    m7.write_csv(OUT / "grid_val.csv", grid)
    for x in grid:
        print(x)
    best = max(grid, key=lambda x: (x["better"] - x["none_to_wrong"], x["gap"]))
    g = best["gap"]
    name = f"TF3_{g:g}"
    print(f"검증으로 고른 차이 기준 = {g:g}", flush=True)
    dev = predict("dev", db, [g])
    table, changed = [], []
    for part, out in (("val", val), ("dev", dev)):
        groups = [("전체", out)]
        if part == "dev":                                            # K-000147 = 사진·라벨 불일치 (사용자 확인)
            groups.append(("K-000147 제외", [x for x in out if x[0]["drug_N"] != "K-000147"]))
        for gname, its in groups:
            for cfg in ("base", name):
                m = m7.measure(its, cfg)
                table.append({"part": part, "group": gname, "config": cfg, **m, "answered_precision": precision(m)})
        for r, p in out:
            k = m7.kind(r["drug_N"], p["base"][0], p[name][0])
            if k:
                changed.append({"part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "truth": r["truth"], "read": r["read"],
                                "change": k, "pred_base": p["base"][0], "pred_new": p[name][0]})
        with open(OUT / f"pred_{part}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "drug_N", "truth", "read", "base_pred", "TF3_pred"])
            for r, p in out:
                w.writerow([r["file_name"], r["drug_N"], r["truth"], r["read"], p["base"][0], p[name][0]])
    m7.write_csv(OUT / "summary.csv", table)
    if changed:
        m7.write_csv(OUT / "changed.csv", changed)
    for t in table:
        print(t)
    for x in sorted(changed, key=lambda x: (x["part"], x["change"], x["drug_N"])):
        print(x["part"], x["change"], x["drug_N"], x["truth"], "| 읽기", x["read"], "|", x["pred_base"] or "(없음)", "→", x["pred_new"])
    get = lambda part, cfg: next(t for t in table if t["part"] == part and t["group"] == "전체" and t["config"] == cfg)
    v, d, d0 = get("val", name), get("dev", name), get("dev", "base")
    vd, dd = v["better"] - v["worse"], d["better"] - d["worse"]
    r1 = vd >= BETTER_MIN and dd >= DEV_MIN
    r2 = v["answered_precision"] >= PRECISION_MIN
    r3 = d["wrong"] - d0["wrong"] <= d["hit"] - d0["hit"]
    print(f"{name}: 검증 {vd:+d} (+{v['better']}/-{v['worse']}), 개발 {dd:+d} (+{d['better']}/-{d['worse']}) → ① {r1}")
    print(f"  검증 답 낸 사진 맞은 비율 {v['answered_precision']}% → ② {r2}")
    print(f"  개발 틀린 답 증가 {d['wrong'] - d0['wrong']}, 맞은 수 증가 {d['hit'] - d0['hit']} → ③ {r3}")
    print("채택:", name if r1 and r2 and r3 else "없음")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
