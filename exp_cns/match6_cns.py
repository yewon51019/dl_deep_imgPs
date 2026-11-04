# =====================================================================
#  match6_cns.py  —  약 판별 v6 1차 시험: 앞뒤 사진 2장으로 약 찾기 (OCR 은 다시 돌리지 않음)
# =====================================================================
#
#  [왜]  사진 한 장은 한 면만 보여서 OR 만 읽으면 오르필 150·300·600mg 을 못 가른다. 두 면을 함께 보면 갈린다.
#  [사진 짝]  같은 약 · 같은 카메라 각도 · 같은 회전각의 앞면 사진 + 뒷면 사진 (파일 이름 기준)
#  [점수]  설정은 base 그대로 고정: λ 0.95, 후보 길이 ≤ 30, 반대 방향 부분일치 끔, w 0.02. 평가 사진으로 고르는 값이 없음
#     s(사진, 각인) = 그 사진의 후보와 각인의 유사도 = max(고른 방향 후보 점수, 0.95 × 다른 방향 후보 점수), 0.3 미만은 0
#     사진 a, b 와 약 d 에 대해:
#       두 면 각인 약 (앞 f, 뒤 r): T = max( [s(a,f) + s(b,r)] / 2 , [s(a,r) + s(b,f)] / 2 )
#                                  사진이 어느 면인지 모르므로 두 가지 배치 중 높은 쪽
#       한 면 각인 약 (각인 x)    : T = max( s(a,x), s(b,x) )   각인 있는 면의 점수만 씀
#       각인 없는 약              : T = 0
#     겉모습 A = 두 사진 attr_score 의 평균, 최종 점수 = T + 0.02 × A, 1등 동점이면 판정 없음
#  [빈 면 점수는 넣지 않음, 사용자 결정 2026-11-02]
#     학습·개발용에 각인 없는 면의 사진이 0장이라 판정 기준을 잴 수 없다.
#     그래서 한 면 약은 두 번째 사진에 글자가 보여도 깎이지 않는다 → 이 때문에 생긴 오답을 onesided_wrong 으로 따로 센다.
#  [--tie-bonus, v6b]  v6 에서 YH + PZM 을 맞게 읽어도 유한피라진아미드정(YH/PZM)과 한 면 약 유한짓정(YH)이 T 1.0 동점이었다.
#     → 두 면 약의 T 에 '고른 배치에서 낮은 쪽 면의 점수 × 0.001' 을 더함. 두 사진의 글자를 모두 설명하는 약을 동점에서 우선하려는 것.
#     동점을 가르는 용도의 고정값이라 따로 고르지 않음. v6 결과를 보고 만든 규칙이라 이 숫자는 낙관적일 수 있음.
#  [오답 dt 그림, --tie-bonus 와 상관없이 저장]  짝이 틀린 사진마다 dt_errors\<약>_<카메라>_<회전각>.png 한 장:
#     앞면 원본 | 앞면 고른 방향 + 상자·읽은 글자 | 뒷면 원본 | 뒷면 고른 방향 + 상자·읽은 글자 | 정답·읽기·답·1~3등 (그림 글자는 영문만, 약은 번호로 표시)
#  [비교 기준]  같은 사진을 base 로 한 장씩 판별한 결과 (match_v3\base 와 같은 규칙).
#  [한계]  짝이 있는 약이 적다 (개발 3종, 검증 4종). 숫자는 참고 수준이고 이 코드로 방식을 확정하지 않는다.
#          한 면 약은 뒷면 사진이 없어 이 코드로 재지 못한다.
#  [저장]  exp_cns\results\match_v6\  pairs.csv (짝 목록), pred_pairs.csv (짝마다 결과), summary.csv
#          이미 있으면 덮어쓰지 않고 멈춤. 평가용 사진·결과는 읽지 않음
#          --tie-bonus 를 쓰면 exp_cns\results\match_v6b\ 에 같은 파일 + dt_errors\ (사진이 든 PNG 는 .gitignore 로 안 올라감)
#  [실행]  python exp_cns\match6_cns.py                     (v6 그대로, 이미 저장돼 있으면 멈춤)
#          python exp_cns\match6_cns.py --tie-bonus 0.001   (v6b, 저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import argparse
import collections
import csv
import json
import os

import cv2
import numpy as np

import match_cns as m1
import match2_cns as m2
from correct_exp300 import MIN_SIM, sim_v                            # match_cns 가 exp300 경로를 sys.path 에 넣어 둠

OUT = m1.cc.RESULTS / "match_v6"
LAM, MAX_LEN, W = 0.95, 30, 0.02
TIE_BONUS = 0.0                                                      # main 에서 --tie-bonus 값으로 바꿈
LABELS = m1.ATTR / "labels.csv"                                      # 사진 경로 (검증·개발용)


def load_part(part):
    """(읽기 결과 {파일: 줄}, CNN 확률 {파일: 칸별 값}). val 은 trainall 읽기 중 검증 부분만."""
    probs = {r["file_name"]: {k: float(v) for k, v in r.items() if k[:2] in ("s_", "f_", "c_")}
             for r in csv.DictReader(open(m1.ATTR / f"probs_{part}.csv", encoding="utf-8-sig"))}
    ocr = {r["file_name"]: r for r in csv.DictReader(open(m1.PRED[part], encoding="utf-8-sig")) if r["file_name"] in probs}
    assert len(ocr) == len(probs), (part, len(ocr), len(probs))
    return ocr, probs


def make_pairs(ocr):
    """같은 약·카메라·회전각의 앞면 + 뒷면 → [(앞면 줄, 뒷면 줄)]. 회전각 = 파일 이름 7번째 칸."""
    g = collections.defaultdict(dict)
    for r in ocr.values():
        g[(r["drug_N"], r["camera_la"], r["file_name"].split("_")[6])][r["face"]] = r
    return [(v["앞면"], v["뒷면"]) for _, v in sorted(g.items()) if len(v) == 2]


def imprint_scores(r, imprints):
    """사진 한 장 → {각인: s}. s = max(고른 방향 후보, λ × 다른 방향 후보), 0.3 미만은 0."""
    cm, co = m2.split_candidates(r, MAX_LEN)
    out = {}
    for imp in imprints:
        sm = max((sim_v(c, imp, *m1.OPT) for c in cm), default=0.0)
        so = LAM * max((sim_v(c, imp, *m1.OPT) for c in co), default=0.0)
        s = max(sm if sm >= MIN_SIM else 0.0, so if so >= MIN_SIM else 0.0)
        if s > 0:
            out[imp] = s
    return out


def pair_text(sa, sb, d):
    """약 d 의 두 사진 각인 점수 T (위 [점수] 규칙)."""
    f, r = d["front"], d["back"]
    if f and r:
        # 두 배치 (a=앞·b=뒤 / a=뒤·b=앞) 를 각각 (평균 점수, 낮은 쪽 면 점수) 로 만들어, 평균이 높은 배치를 씀
        one = ((sa.get(f, 0.0) + sb.get(r, 0.0)) / 2, min(sa.get(f, 0.0), sb.get(r, 0.0)))
        two = ((sa.get(r, 0.0) + sb.get(f, 0.0)) / 2, min(sa.get(r, 0.0), sb.get(f, 0.0)))
        mean, low = max(one, two)
        return mean + TIE_BONUS * low                                # TIE_BONUS 0 이면 v6 와 같음
    x = f or r
    return max(sa.get(x, 0.0), sb.get(x, 0.0)) if x else 0.0


def single_answer(s, A, db):
    """사진 한 장 base: 약마다 앞·뒷면 중 높은 쪽 T + 0.02 × A."""
    return m1.decide({d["drug_N"]: max((s.get(x, 0.0) for x in d["imprints"]), default=0.0) + W * A[d["drug_N"]] for d in db})


def run_part(part, db, imprints):
    """짝마다 2장 판별과, 같은 사진의 1장씩 base 판별 → 결과 줄 목록."""
    ocr, probs = load_part(part)
    D = {d["drug_N"]: d for d in db}
    rows = []
    for a, b in make_pairs(ocr):
        sa, sb = imprint_scores(a, imprints), imprint_scores(b, imprints)
        Aa = {d["drug_N"]: m1.attr_score(probs[a["file_name"]], d) for d in db}
        Ab = {d["drug_N"]: m1.attr_score(probs[b["file_name"]], d) for d in db}
        A = {k: (Aa[k] + Ab[k]) / 2 for k in Aa}
        T = {d["drug_N"]: pair_text(sa, sb, d) for d in db}
        pred, top = m1.decide({k: T[k] + W * A[k] for k in A})
        pa, pb = single_answer(sa, Aa, db)[0], single_answer(sb, Ab, db)[0]
        drug = a["drug_N"]
        one_sided = bool(pred) and pred != drug and not (D[pred]["front"] and D[pred]["back"])
        rows.append(dict(part=part, drug_N=drug, name=D[drug]["name"], camera=a["camera_la"], front_file=a["file_name"], back_file=b["file_name"],
                         label_front=a["truth"], label_back=b["truth"], read_front=a["read"], read_back=b["read"],
                         pair_pred=pred, pair_pred_name=D[pred]["name"] if pred else "", pair_ok=int(pred == drug),
                         pair_top3=" ".join(f"{d}:{v:.3f}" for d, v in top), pair_in_top3=int(drug in [d for d, _ in top]),
                         single_front_pred=pa, single_back_pred=pb, single_front_ok=int(pa == drug), single_back_ok=int(pb == drug),
                         T_truth=round(T[drug], 3), onesided_wrong=int(one_sided)))
    return rows


def photo_panels(row, paths):
    """사진 한 장의 그림 두 칸: 원본(400) | 고른 방향으로 돌린 사진 + 상자·상자별 글자 (400 안에 맞춤)."""
    img = m1.cc.fit_side(m1.cc.imread(paths[row["file_name"]]), 400)
    rot = m1.oc.rotate_color(img, int(float(row["angle"]))).copy()
    for x0, y0, x1, y1, t in json.loads(row["best_boxes"] or "[]"):
        cv2.rectangle(rot, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(rot, t or "-", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    s = 400 / max(rot.shape[:2])                                     # 돌리면 캔버스가 커지므로 400 안에 맞춤
    return [img, m1.cc.fit_side(cv2.resize(rot, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 400)]


def save_error_dt(rows, ocr_by_file, paths, folder):
    """짝이 틀린 사진마다 한 장: 앞면 원본 | 앞면 상자 | 뒷면 원본 | 뒷면 상자 | 글자 칸."""
    os.makedirs(folder, exist_ok=True)
    n = 0
    for r in rows:
        if r["pair_ok"]:
            continue
        a, b = ocr_by_file[r["front_file"]], ocr_by_file[r["back_file"]]
        panel = np.full((400, 400, 3), 255, np.uint8)
        lines = [(f"{r['part']}  {r['drug_N']}  cam{r['camera']}", (0, 0, 0)),
                 (f"label front: {r['label_front']}", (0, 0, 0)),
                 (f"label back : {r['label_back']}", (0, 0, 0)),
                 (f"read front : {r['read_front'] or '(empty)'}", (0, 0, 0)),
                 (f"read back  : {r['read_back'] or '(empty)'}", (0, 0, 0)),
                 (f"answer: {r['pair_pred'] or 'NO DECISION'}", (0, 0, 220)),
                 (f"truth T: {r['T_truth']}", (60, 60, 60))]
        for t in r["pair_top3"].split()[:3]:
            lines.append((f"  {t}", (60, 60, 60)))
        for i, (t, col) in enumerate(lines):
            cv2.putText(panel, t, (10, 32 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        tile = np.hstack(photo_panels(a, paths) + photo_panels(b, paths) + [panel])
        rot = r["front_file"].split("_")[6]
        m1.cc.imwrite(str(folder / f"{r['part']}_{r['drug_N']}_{r['camera']}_{rot}.png"), tile)
        n += 1
    return n


def summarize(rows, part, drug=None):
    """한 줄 요약. drug 를 주면 그 약만."""
    v = [r for r in rows if r["part"] == part and (drug is None or r["drug_N"] == drug)]
    n = len(v)
    single = sum(r["single_front_ok"] + r["single_back_ok"] for r in v)
    return dict(part=part, drug_N=drug or "전체", name=v[0]["name"] if drug else "", pairs=n,
                pair_correct=sum(r["pair_ok"] for r in v), pair_acc=round(100 * sum(r["pair_ok"] for r in v) / n, 1),
                pair_none=sum(r["pair_pred"] == "" for r in v), pair_top3=round(100 * sum(r["pair_in_top3"] for r in v) / n, 1),
                single_photos=2 * n, single_correct=single, single_acc=round(100 * single / (2 * n), 1),
                single_none=sum((r["single_front_pred"] == "") + (r["single_back_pred"] == "") for r in v),
                onesided_wrong=sum(r["onesided_wrong"] for r in v))


def main():
    global OUT, TIE_BONUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tie-bonus", type=float, default=0.0, help="두 면 약의 T 에 '낮은 쪽 면 점수 × 이 값' 을 더함 (0 이면 v6, 0 이 아니면 결과 폴더 match_v6b)")
    a = ap.parse_args()
    TIE_BONUS = a.tie_bonus
    if TIE_BONUS:
        OUT = m1.cc.RESULTS / "match_v6b"
    assert not (OUT / "summary.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT}"
    os.makedirs(OUT, exist_ok=True)
    db = m1.load_db()
    imprints = sorted({x for d in db for x in d["imprints"]})
    rows = []
    for part in ("val", "dev"):
        rows += run_part(part, db, imprints)
    with open(OUT / "pred_pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["part", "drug_N", "camera", "front_file", "back_file"])
        for r in rows:
            w.writerow([r["part"], r["drug_N"], r["camera"], r["front_file"], r["back_file"]])
    table = []
    for part in ("val", "dev"):
        table.append(summarize(rows, part))
        for d in sorted({r["drug_N"] for r in rows if r["part"] == part}):
            table.append(summarize(rows, part, d))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(f"  {t['part']:3s} {t['drug_N']:9s} {t['name'][:14]:14s} 짝 {t['pairs']:3d}  2장 {t['pair_acc']:5.1f}% ({t['pair_correct']}) 판정없음 {t['pair_none']:2d} 3등 안 {t['pair_top3']:5.1f}  "
              f"| 1장씩 {t['single_acc']:5.1f}% ({t['single_correct']}/{t['single_photos']}) 판정없음 {t['single_none']:2d}  | 한 면 약 오답 {t['onesided_wrong']}")
    # 오답 dt 그림: 검증·개발 읽기를 합쳐서 파일 이름으로 찾음
    ocr_by_file = {}
    for part in ("val", "dev"):
        ocr_by_file.update(load_part(part)[0])
    paths = {r["file_name"]: r["image_path"] for r in csv.DictReader(open(LABELS, encoding="utf-8-sig"))}
    n = save_error_dt(rows, ocr_by_file, paths, OUT / "dt_errors")
    print(f"오답 dt 그림 {n}장: {OUT / 'dt_errors'}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
