# =====================================================================
#  match5_cns.py  —  약 판별 v5 시험: 겉모습 점수에서 '색'만 쓰면 나아지는가 (OCR 은 다시 돌리지 않음)
# =====================================================================
#
#  [왜]  같은 각인을 가진 약이 많은데, 지금 겉모습 점수 A 는 모양·제형·색을 모두 더한다.
#        CNN 이 모양을 틀리면 색이 맞아도 다른 약을 고를 수 있어서, 색만 쓰는 방식과 비교한다.
#  [각인 점수 T]  match2 의 base 그대로: T = max(고른 방향 후보, 0.95 × 다른 방향 후보), 후보 길이 ≤ 30, 반대 방향 부분일치 끔
#  [비교하는 방식]
#     base  T + 0.02 × A                  지금 방식. A = 모양 + 제형 + 색 CNN log 확률
#     C1    T + w × Cc                    Cc = A 중 색 12칸 부분만. w 는 W_GRID 중 검증으로 고름
#     C2    T 1등이 동점일 때만 Cc 로 결정   동점이 아니면 T 1등 그대로
#     C3    T 1등이 동점일 때만 Lab 색으로 결정 (exp300 '색 동점 깨기' 방식, 아래 참고)
#     C2·C3 에서 색으로도 못 가르면 판정 없음. 어느 방식도 약 크기는 쓰지 않음 (촬영 거리를 고정할 수 없음).
#  [Lab 색 (C3)]
#     1. 사진에서 알약 영역을 찾아(segment_pill, 못 찾으면 사진 전체) a·b 중앙값을 잰다. a·b = 밝기를 뺀 색 값
#     2. 색 이름별 대표값 = 학습 부분(part=train) 중 '한 가지 색' 약 사진의 a·b 평균. 검증·개발용 사진은 대표값에 쓰지 않음
#     3. 약의 예상 색 = 대표값, 두 색 캡슐은 두 대표값의 평균 (사진 중앙값에 두 색이 섞이므로)
#     4. 동점인 약 중 예상 색이 재어 낸 색과 가장 가까운 약. 약의 색에 대표값이 없거나 거리가 같으면 판정 없음
#  [고르는 규칙, 결과를 보기 전에 정함]
#     검증 828장(val)의 약 정확도가 가장 높은 방식을 고름. 같으면 C2 → C3 → C1 (작은 w) 순서.
#     고른 방식이 base 보다 검증에서 2장 이상 더 맞히지 못하면 base 를 유지. 개발용 468장은 확인만 하고 고르는 데 쓰지 않음.
#     평가용 사진은 이 코드에서 읽지 않음.
#     주의: base 의 λ·w 는 검증에서 고른 값이라, 검증에서는 base 가 조금 유리함.
#  [저장]  exp_cns\results\match_v5\  color_protos.csv, grid_val.csv, summary.csv, pred_val.csv, pred_dev.csv
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\match5_cns.py   (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv
import math
import os

import cv2
import numpy as np

import match_cns as m1
import match2_cns as m2
from attr_cnn_cns import COLORS
from run_exp300 import segment_pill                                  # match_cns 가 exp300 경로를 sys.path 에 넣어 둠

OUT = m1.cc.RESULTS / "match_v5"
LABELS = m1.ATTR / "labels.csv"
LAM, REV, MAX_LEN, W_BASE = 0.95, False, 30, 0.02                    # match2 base 설정 그대로
W_GRID = [w for w in m1.W_GRID if w > 0]                             # C1 의 w 후보 (0 이면 각인만과 같아서 뺌)
PART_DIR = {r["file_name"]: r for r in csv.DictReader(open(LABELS, encoding="utf-8-sig"))}   # 사진 경로·색 라벨


def color_score(p, d):
    """색 CNN 점수 Cc: match_cns.attr_score 의 색 12칸 부분만 (확률은 같은 0.02~0.98 로 자름)."""
    cl = lambda x: min(max(x, 0.02), 0.98)
    return sum(math.log(cl(p[f"c_{c}"]) if c in d["cols"] else 1 - cl(p[f"c_{c}"])) for c in COLORS)


def pill_ab(path):
    """사진 → (알약 부분 Lab a·b 중앙값, 영역을 찾았는지). OpenCV 8비트 Lab 은 128 이 무채색."""
    img = m1.cc.imread(path)
    mask = segment_pill(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    sel = mask > 0 if mask is not None else np.ones(img.shape[:2], bool)
    return np.array([np.median(lab[..., 1][sel]), np.median(lab[..., 2][sel])], np.float32), mask is not None


def make_protos():
    """학습 부분의 한 가지 색 약 사진 → 색 이름별 a·b 평균. 두 색 약은 어느 색인지 나눌 수 없어 쓰지 않음."""
    acc = collections.defaultdict(list)
    for r in PART_DIR.values():
        if r["part"] == "train" and "|" not in r["colors"] and r["colors"]:
            acc[r["colors"]].append(pill_ab(r["image_path"])[0])
    protos = {c: np.mean(v, 0) for c, v in acc.items()}
    with open(OUT / "color_protos.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["color", "a", "b", "photos"])
        for c in sorted(protos, key=lambda x: -len(acc[x])):
            w.writerow([c, round(float(protos[c][0]), 2), round(float(protos[c][1]), 2), len(acc[c])])
    print("색 대표값:", {c: (round(float(v[0]), 1), round(float(v[1]), 1), len(acc[c])) for c, v in protos.items()})
    return protos


def expected_ab(d, protos):
    """약의 예상 a·b. 라벨 색이 없거나 대표값이 없는 색이 있으면 None."""
    if not d["cols"] or any(c not in protos for c in d["cols"]):
        return None
    return np.mean([protos[c] for c in d["cols"]], 0)


def tie_decide(T, key):
    """T 1등이 동점이면 key(약 → 점수, 높을수록 좋음, None = 모름)로 가름 → (1등 약, 1~3등).
       후보 약 하나라도 key 가 None 이면 판정 없음. 1~3등은 T 순서, 같은 T 안에서는 key 순서."""
    if not T:
        return "", []
    best = max(T.values())
    top = [d for d, v in T.items() if abs(v - best) < 1e-9]
    k = {d: key(d) for d in T}
    fin = lambda d: k[d] if k[d] is not None else -1e9
    order = sorted(T, key=lambda d: (-round(T[d], 9), -fin(d)))
    t3 = [(d, T[d]) for d in order[:3]]
    if len(top) == 1:
        return top[0], t3
    if any(k[d] is None for d in top):
        return "", t3
    vals = sorted((k[d] for d in top), reverse=True)
    return ("", t3) if vals[0] - vals[1] < 1e-9 else (max(top, key=lambda d: k[d]), t3)


def predict(part, db, protos):
    """사진마다 {방식 이름: (1등, 1~3등)}. 저장된 읽기·CNN 확률만 씀, Lab 색을 잴 때만 사진을 읽음."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(m1.PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs), (part, len(ocr), len(probs))
    exp = {d["drug_N"]: expected_ab(d, protos) for d in db}
    out, n_tie, n_nomask = [], 0, 0
    for k, (fn, r) in enumerate(sorted(ocr.items()), 1):
        cm, co = m2.split_candidates(r, MAX_LEN)
        T = m2.combine(m2.text_scores(cm, db, REV), m2.text_scores(co, db, REV), LAM)
        A = {d["drug_N"]: m1.attr_score(probs[fn], d) for d in db}
        C = {d["drug_N"]: color_score(probs[fn], d) for d in db}
        res = {"base": m1.decide({d: T.get(d, 0.0) + W_BASE * A[d] for d in A})}
        for w in W_GRID:
            res[f"C1_{w:g}"] = m1.decide({d: T.get(d, 0.0) + w * C[d] for d in C})
        res["C2"] = tie_decide(T, lambda d: C[d])
        top = [d for d, v in T.items() if T and abs(v - max(T.values())) < 1e-9]
        if len(top) > 1:                                             # 동점일 때만 사진을 읽어 Lab 색을 잼 (시간 절약)
            n_tie += 1
            ab, found = pill_ab(PART_DIR[fn]["image_path"])
            n_nomask += not found
            res["C3"] = tie_decide(T, lambda d: None if exp[d] is None else -float(np.linalg.norm(ab - exp[d])))
        else:
            res["C3"] = tie_decide(T, lambda d: 0.0)                 # 동점이 아니면 색이 필요 없음
        out.append((r, res))
        if k % 200 == 0:
            print(f"  {part} {k}/{len(ocr)}")
    print(f"  {part}: 각인 1등 동점 {n_tie}장 (그중 알약 영역 못 찾음 {n_nomask}장)")
    return out


def updown(out, key):
    """base 대비 (좋아짐, 나빠짐) 장수."""
    ok = lambda r, x, k: r["drug_N"] == x[k][0]
    return sum(ok(r, x, key) and not ok(r, x, "base") for r, x in out), sum(ok(r, x, "base") and not ok(r, x, key) for r, x in out)


def main():
    assert not (OUT / "summary.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT}"
    os.makedirs(OUT, exist_ok=True)
    db = m1.load_db()                                                # match_v1\search_db_cns.csv (m1.OUT 을 바꾸기 전에 읽어야 함)
    m1.OUT = OUT                                                     # save_pred 가 쓰는 폴더
    protos = make_protos()
    val = predict("val", db, protos)

    # 검증에서 고르기: 정확도 높은 순, 같으면 C2 → C3 → C1 (작은 w) 순서
    simple = ["C2", "C3"] + [f"C1_{w:g}" for w in W_GRID]
    correct = lambda out, k: sum(r["drug_N"] == x[k][0] for r, x in out)
    grid = [dict(method=k, correct=correct(val, k), acc=m1.metrics(val, k)["acc"], none=m1.metrics(val, k)["none"]) for k in ["base"] + simple]
    with open(OUT / "grid_val.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    best = max(simple, key=lambda k: (correct(val, k), -simple.index(k)))
    base_ok, best_ok = correct(val, "base"), correct(val, best)
    chosen = best if best_ok >= base_ok + 2 else "base"
    print(f"검증: base {base_ok}장, 최고 {best} {best_ok}장 → 선택 {chosen}")

    dev = predict("dev", db, protos)
    c1 = max((k for k in simple if k.startswith("C1")), key=lambda k: (correct(val, k), -simple.index(k)))   # 검증에서 고른 C1
    keys = ["base", "C2", "C3", c1]
    rows = []
    for part, out in (("val", val), ("dev", dev)):
        m1.save_pred(part, out, keys)
        for k in keys:
            mm = m1.metrics(out, k)
            up, down = updown(out, k)
            rows.append(dict(part=part, method=k, chosen=int(k == chosen), correct=correct(out, k), **mm, better=up, worse=down))
            print(f"  {part:3s} {k:8s} 약 {mm['acc']:5.1f}% ({correct(out, k)})  3등 안 {mm['top3']:5.1f}  판정없음 {mm['none']:3d}  "
                  f"약 단위 {mm['drug_acc']:5.1f}  긴 {mm['long']:5.1f} 그 외 {mm['other']:5.1f}  base 대비 +{up}/−{down}"
                  + ("  ← 선택" if k == chosen else ""))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
