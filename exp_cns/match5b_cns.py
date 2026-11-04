# =====================================================================
#  match5b_cns.py  —  약 판별 v5b 시험: 모양·제형을 유지한 채 Lab 색을 함께 쓰면 나아지는가 (OCR 은 다시 돌리지 않음)
# =====================================================================
#
#  [왜]  match5_cns.py 는 모양·제형 점수를 뺀 '색만'을 시험했고, 모든 방식이 base 보다 나빴다.
#        이번에는 모양·제형을 그대로 두고 Lab 색을 더하거나 바꾸거나 순서를 나눴을 때를 본다.
#  [점수 이름]
#     T    각인 점수 (match2 base 그대로)
#     SF   CNN 모양 + 제형 log 확률  (A 에서 색 부분을 뺀 것)
#     Cc   CNN 색 12칸 log 확률      (A = SF + Cc)
#     dist 사진의 Lab a·b 와 약의 예상 색 사이 거리 (작을수록 비슷함, match5 의 대표값 color_protos.csv 그대로)
#  [비교하는 방식]
#     base  T + 0.02 × (SF + Cc)                        지금 방식
#     D2    T + 0.02 × (SF + Cc) − v × dist             CNN 색을 두고 Lab 색을 더함
#     D3    T + 0.02 × SF − v × dist                    CNN 색을 Lab 색으로 바꿈
#     D4    T 1등 동점 → SF 최고 → 그래도 동점이면 dist 최소   점수를 더하지 않고 순서대로 거름. 그래도 동점이면 판정 없음
#     v 는 V_GRID 중 검증으로 고름. 약 색에 대표값이 없으면 그 사진에서 잰 거리의 중앙값을 씀 (이득도 손해도 크게 안 주려고)
#  [고르는 규칙, 결과를 보기 전에 정함 = match5 와 같음]
#     검증 828장(val)의 약 정확도가 가장 높은 방식. 같으면 D4 → D3 → D2 (작은 v) 순서.
#     base 보다 검증에서 2장 이상 더 맞히지 못하면 base 유지. 개발용 468장은 확인만 함. 평가용 사진은 읽지 않음.
#     주의: base 의 λ·w 는 검증에서 고른 값이라, 검증에서는 base 가 조금 유리함.
#  [저장]  exp_cns\results\match_v5b\  grid_val.csv, summary.csv, pred_val.csv, pred_dev.csv
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\match5b_cns.py   (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv
import os

import numpy as np

import match_cns as m1
import match2_cns as m2
import match5_cns as m5                                              # color_score, pill_ab, expected_ab, 사진 경로

OUT = m1.cc.RESULTS / "match_v5b"
PROTOS = m1.cc.RESULTS / "match_v5" / "color_protos.csv"
V_GRID = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
LAM, REV, MAX_LEN, W = m5.LAM, m5.REV, m5.MAX_LEN, m5.W_BASE         # base 설정 그대로


def load_protos():
    """match5 가 학습 부분 사진으로 만든 색 이름별 a·b 대표값."""
    return {r["color"]: np.array([float(r["a"]), float(r["b"])], np.float32)
            for r in csv.DictReader(open(PROTOS, encoding="utf-8-sig"))}


def order_decide(T, SF, dist):
    """D4: T 1등 → SF 최고 → dist 최소 순서로 거름 → (1등 약, 1~3등). 끝까지 동점이면 판정 없음."""
    if not T:
        return "", []
    order = sorted(T, key=lambda d: (-round(T[d], 9), -round(SF[d], 9), round(dist[d], 6)))
    t3 = [(d, T[d]) for d in order[:3]]
    best = max(T.values())
    top = [d for d in T if abs(T[d] - best) < 1e-9]
    best_sf = max(SF[d] for d in top)
    top = [d for d in top if abs(SF[d] - best_sf) < 1e-9]
    best_d = min(dist[d] for d in top)
    top = [d for d in top if abs(dist[d] - best_d) < 1e-6]
    return (top[0], t3) if len(top) == 1 else ("", t3)


def predict(part, db, protos):
    """사진마다 {방식 이름: (1등, 1~3등)}. 저장된 읽기·CNN 확률 + 사진에서 잰 Lab 색."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(m1.PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs), (part, len(ocr), len(probs))
    exp = {d["drug_N"]: m5.expected_ab(d, protos) for d in db}
    out = []
    for k, (fn, r) in enumerate(sorted(ocr.items()), 1):
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, REV), m2.text_scores(co, db, REV), LAM)
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        C = {d["drug_N"]: m5.color_score(probs[fn], d) for d in db}
        SF = {d: A[d] - C[d] for d in A}
        ab, _ = m5.pill_ab(m5.PART_DIR[fn]["image_path"])
        known = {d: float(np.linalg.norm(ab - e)) for d, e in exp.items() if e is not None}
        mid = float(np.median(list(known.values())))
        dist = {d: known.get(d, mid) for d in A}
        res = {"base": m1.decide({d: T.get(d, 0.0) + W * A[d] for d in A})}
        for v in V_GRID:
            res[f"D2_{v:g}"] = m1.decide({d: T.get(d, 0.0) + W * A[d] - v * dist[d] for d in A})
            res[f"D3_{v:g}"] = m1.decide({d: T.get(d, 0.0) + W * SF[d] - v * dist[d] for d in A})
        res["D4"] = order_decide(T, SF, dist)
        out.append((r, res))
        if k % 200 == 0:
            print(f"  {part} {k}/{len(ocr)}")
    return out


def main():
    assert not (OUT / "summary.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT}"
    os.makedirs(OUT, exist_ok=True)
    db = m1.load_db()                                                # m1.OUT 을 바꾸기 전에 읽어야 함
    m1.OUT = OUT                                                     # save_pred 가 쓰는 폴더
    protos = load_protos()
    val = predict("val", db, protos)
    correct = lambda out, k: sum(r["drug_N"] == x[k][0] for r, x in out)
    # 검증에서 고르기: 정확도 높은 순, 같으면 앞쪽(단순한 쪽) 먼저
    simple = ["D4"] + [f"D3_{v:g}" for v in V_GRID] + [f"D2_{v:g}" for v in V_GRID]
    grid = [dict(method=k, correct=correct(val, k), acc=m1.metrics(val, k)["acc"], none=m1.metrics(val, k)["none"]) for k in ["base"] + simple]
    with open(OUT / "grid_val.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    best_of = lambda pre: max((k for k in simple if k.startswith(pre)), key=lambda k: (correct(val, k), -simple.index(k)))
    best = max(simple, key=lambda k: (correct(val, k), -simple.index(k)))
    base_ok, best_ok = correct(val, "base"), correct(val, best)
    chosen = best if best_ok >= base_ok + 2 else "base"
    print(f"검증: base {base_ok}장, 최고 {best} {best_ok}장 → 선택 {chosen}")
    dev = predict("dev", db, protos)
    keys = ["base", best_of("D2"), best_of("D3"), "D4"]
    rows = []
    for part, out in (("val", val), ("dev", dev)):
        m1.save_pred(part, out, keys)
        for k in keys:
            mm = m1.metrics(out, k)
            up, down = m5.updown(out, k)
            rows.append(dict(part=part, method=k, chosen=int(k == chosen), correct=correct(out, k), **mm, better=up, worse=down))
            print(f"  {part:3s} {k:10s} 약 {mm['acc']:5.1f}% ({correct(out, k)})  3등 안 {mm['top3']:5.1f}  판정없음 {mm['none']:3d}  "
                  f"긴 {mm['long']:5.1f} 그 외 {mm['other']:5.1f}  base 대비 +{up}/−{down}" + ("  ← 선택" if k == chosen else ""))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
