# =====================================================================
#  angle_select_exp300.py  —  각인 정렬 실험 ② '확신도로 각도 고르기' (OCR 재실행 없음)
# =====================================================================
#
#  [아이디어]
#   지금은 12방향(둥근 알약)을 다 읽고 '가장 많이 나온 글자'(다수결)를 고릅니다.
#   그런데 각인이 똑바로 선 방향에서는 모델이 더 확신할 테니, 확신도가 가장 높은 방향 하나를 고르면 어떨까?
#   run_exp300.py 가 저장한 cand_detail 열(각도별 [각도, 글자, 확신도])로 고르는 규칙만 바꿔 다시 채점합니다.
#
#  [비교하는 규칙]
#   S0 다수결(기준)  : 지금 방식. 가장 많이 나온 글자, 동수면 확신도 높은 쪽
#   S1 확신도 최고   : 글자가 나온 방향 중 확신도가 가장 높은 방향 하나
#   S2 확신도 합산   : 같은 글자끼리 확신도를 더해서 가장 큰 글자 (다수결 + 확신도 절충)
#   S3 0도 한 번     : 돌리지 않고 0도에서 읽은 것 (회전 투표가 얼마나 도움 되는지 보는 참고값)
#   ※ 약 판별은 run_exp300.identify 의 'edit'(편집거리) 방식 그대로. 보정(부분일치 등)은 넣지 않음.
#
#  [정렬이 맞았는지 보는 법] (참고용, 고르는 데 쓰지 않음)
#   같은 약 같은 면이면 "고른 각도 ± 파일 이름의 회전각"이 일정해야 합니다.
#   약(+면)별로 이 값의 원형 퍼짐(0=완전히 일정, 1=제멋대로)을 구해 평균을 냅니다. 회전 방향 약속을
#   모르므로 (고른 각도 + 회전각), (고른 각도 − 회전각) 두 가지를 다 보여 줍니다.
#
#  [공정성 규칙]  규칙은 개발용(dev) 점수로만 고르고, 평가용(test)은 마지막에 한 번만.
#
#  [실행 예] (PowerShell)
#    python exp300\angle_select_exp300.py --pred exp300\results\pred_dev_angle.csv `
#        --db exp300\results\search_db.csv --combo parseq:P0 --out exp300\results\angle_select_dev.csv
# =====================================================================

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_exp300 import drug_metrics          # noqa: E402
from run_exp300 import identify, load_db           # noqa: E402

RULES = [("S0", "다수결(기준)"), ("S1", "확신도 최고"), ("S2", "확신도 합산"), ("S3", "0도 한 번")]


def valid(t):
    return 1 <= len(t) <= 12                      # run_exp300 의 다수결과 같은 길이 조건


def pick(detail, rule):
    """detail = [[각도, 글자, 확신도], ...] → (고른 글자, 고른 각도 또는 None)"""
    ok = [(a, t, c) for a, t, c in detail if valid(t)]
    if not ok:
        return "", None
    if rule == "S0":
        votes = Counter(t for _, t, _ in ok)
        a, t, _ = max(ok, key=lambda x: (votes[x[1]], x[2]))
        return t, a
    if rule == "S1":
        a, t, _ = max(ok, key=lambda x: x[2])
        return t, a
    if rule == "S2":
        tot = defaultdict(float)
        for _, t, c in ok:
            tot[t] += c
        best = max(tot, key=lambda t: tot[t])
        a = max((x for x in ok if x[1] == best), key=lambda x: x[2])[0]
        return best, a
    if rule == "S3":
        z = [x for x in detail if x[0] == 0]
        return (z[0][1], 0) if z and valid(z[0][1]) else ("", 0)
    raise ValueError(rule)


def circ_spread(degs):
    """각도들의 원형 퍼짐 1-R (0 = 모두 같은 각도, 1 = 고르게 흩어짐)"""
    if len(degs) < 2:
        return None
    c = sum(math.cos(math.radians(d)) for d in degs) / len(degs)
    s = sum(math.sin(math.radians(d)) for d in degs) / len(degs)
    return 1 - math.hypot(c, s)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="각인 정렬 ②: 각도 고르는 규칙별 재채점")
    ap.add_argument("--pred", required=True, help="cand_detail 열이 있는 run_exp300 예측 CSV")
    ap.add_argument("--db", required=True, help="search_db.csv")
    ap.add_argument("--combo", default="parseq:P0", help="채점할 조합 (예: parseq:P0)")
    ap.add_argument("--out", default="angle_select.csv", help="사진별 결과 CSV")
    a = ap.parse_args()

    with open(a.pred, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["combo"] == a.combo]
    if not rows:
        sys.exit(f"{a.pred} 에 {a.combo} 행이 없어요.")
    if "cand_detail" not in rows[0] or not rows[0]["cand_detail"]:
        sys.exit("cand_detail 열이 없어요. 고친 run_exp300.py 로 새 파일 이름(--out)에 다시 돌려 주세요.")
    db = load_db(a.db)
    print(f"{a.combo}: {len(rows)}장 (split: {', '.join(sorted({r['split'] for r in rows}))}), 검색 DB {len(db)}종\n")

    out = []
    for r in rows:
        detail = json.loads(r["cand_detail"])
        rec = {"file_name": r["file_name"], "split": r["split"], "det_shape": r["det_shape"],
               "camera_la": r.get("camera_la", ""), "rot_deg": r.get("rot_deg", ""),
               "drug_dir": r.get("drug_dir", ""), "truth_drug": r["truth_drug"], "truth_imprint": r["truth_imprint"]}
        for code, _ in RULES:
            t, ang = pick(detail, code)
            _, edit, _, _, _ = identify(t, r["det_shape"], r["det_color"], db)
            rec[f"{code}_text"], rec[f"{code}_angle"], rec[f"{code}_drug"] = t, ("" if ang is None else ang), edit
        out.append(rec)

    groups = [("전체", out), ("둥근", [x for x in out if x["det_shape"] == "둥근"]),
              ("길쭉", [x for x in out if x["det_shape"] == "길쭉"])]
    print(f"{'규칙':<14}{'묶음':<6}{'n':>4}{'정확도':>8}{'정밀도':>8}{'재현율':>8}{'F1':>7}{'각인일치':>9}")
    for code, name in RULES:
        for gname, g in groups:
            if not g:
                continue
            acc, p, rc, f1 = drug_metrics([x["truth_drug"] for x in g], [x[f"{code}_drug"] for x in g])
            ex = mean([x[f"{code}_text"] == x["truth_imprint"] for x in g])
            print(f"{code} {name:<10}{gname:<6}{len(g):>4}{acc*100:8.1f}{p*100:8.1f}{rc*100:8.1f}{f1*100:7.1f}{ex*100:9.1f}")
    n = len(out)
    print(f"\n※ 사진 {n}장이면 1장 ≈ {100/n:.1f}%p. 이보다 몇 배 큰 차이가 아니면 우연일 수 있어요.")

    # 둥근 알약: 규칙끼리 서로 다른 결과가 나온 사진 수 (S0 대비)
    rnd = groups[1][1]
    for code, name in RULES[1:]:
        win = sum(x[f"{code}_drug"] == x["truth_drug"] != x["S0_drug"] for x in rnd)
        lose = sum(x["S0_drug"] == x["truth_drug"] != x[f"{code}_drug"] for x in rnd)
        print(f"둥근 {len(rnd)}장 중 {code}가 S0보다  맞힘 +{win} / 틀림 -{lose}")

    # 정렬 일관성 (둥근 알약, 회전각이 있는 사진만)
    print("\n[정렬 일관성] 약(+앞뒤 각인)별 '고른 각도 ± 회전각' 의 원형 퍼짐 평균 (0 = 일정, 1 = 제멋대로)")
    for code, name in RULES[:3]:
        by_p, by_m = defaultdict(list), defaultdict(list)
        for x in rnd:
            if x[f"{code}_angle"] == "" or x["rot_deg"] in ("", None):
                continue
            key = (x["truth_drug"], x["truth_imprint"])
            by_p[key].append((x[f"{code}_angle"] + float(x["rot_deg"])) % 360)
            by_m[key].append((x[f"{code}_angle"] - float(x["rot_deg"])) % 360)
        print(f"  {code} {name:<10} (+): {mean([circ_spread(v) for v in by_p.values()]):.2f}"
              f"   (−): {mean([circ_spread(v) for v in by_m.values()]):.2f}   약 묶음 {len(by_p)}개")

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n사진별 결과 저장: {a.out}")


if __name__ == "__main__":
    main()
