# =====================================================================
#  tie_cause_cns.py  —  판정 없음(1등 동점) 원인 세기 + 읽기 실패를 저장된 값으로 알아챌 수 있는지 (OCR 안 돌림)
# =====================================================================
#
#  [대상]  파인튜닝 PARSeq(pft1) 약 판별, 검증 828장 + 개발 468장. 평가용 사진·결과는 읽지 않음
#  [동점 칸]  동점이면 각인 점수와 겉모습 라벨이 모두 같은 약들이라는 뜻. 정답 약이 동점 안에 있는지와 읽기로 나눔
#     A 같은 각인, 읽기 맞음     : 정답이 동점 안 + 읽기 완전일치 → 이 면만으로는 구분 불가
#     B 정답 포함, 읽기 틀림     : 정답이 동점 안 + 읽기 틀림 (조각·부분 읽기가 여러 약에 똑같이 맞음)
#     C 정답 빠짐, 짧은 읽기     : 정답이 동점 밖 + 읽은 글자 2자 이하
#     D 정답 빠짐, 그 밖         : 정답이 동점 밖 + 3자 이상
#     two_face = 동점 약들의 (앞, 뒤) 각인 쌍이 모두 달라 라벨상 두 면이면 구분되는 경우
#  [읽기 실패 신호]  확신도, 상자 수, 큰 상자(넓이 ≥ 400×400 의 절반), 1등 점수 구간별 완전일치율과 약 정확도
#     정답을 쓰지 않는 값이라 실제 프로그램에서 쓸 수 있는 신호인지 보는 것
#  [각인 없는 사진]  이 세트의 정답 각인이 빈 사진 수를 함께 셈 (각인 없는 면을 잴 수 있는지 확인)
#  [저장]  exp_cns\results\tie_cause\  ties.csv, tie_summary.csv, tie_by_drug.csv, signal_bins.csv
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\tie_cause_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv
import json

import display_cns as dp
import match_cns as m1

OUT = m1.cc.RESULTS / "tie_cause"
BIG = 400 * 400 / 2
EPS = dp.EPS


def tie_kind(has_truth, exact, read):
    if has_truth:
        return "A 같은 각인, 읽기 맞음" if exact else "B 정답 포함, 읽기 틀림"
    return "C 정답 빠짐, 짧은 읽기(2자 이하)" if len(read) <= 2 else "D 정답 빠짐, 3자 이상"


def box_info(r):
    boxes = json.loads(r["best_boxes"] or "[]")
    big = any((b[2] - b[0]) * (b[3] - b[1]) >= BIG for b in boxes)
    return len(boxes), big


def bins(rows, part, name, key, edges):
    out = []
    for lo, hi in edges:
        rs = [r for r in rows if lo <= key(r) < hi]
        n = len(rs)
        out.append({"part": part, "signal": name, "from": lo, "to": hi, "n": n,
                    "read_exact": round(100 * sum(r["exact"] for r in rs) / max(n, 1), 1),
                    "drug_top1": round(100 * sum(r["ok"] for r in rs) / max(n, 1), 1)})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if (OUT / "tie_summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = m1.load_db()
    info = {d["drug_N"]: d for d in db}
    ties, signal = [], []
    for part in ("val", "dev"):
        items = dp.score("pft1", part, db)
        print(part, "정답 각인이 빈 사진:", sum(not r["truth"] for r, _, _ in items))
        rows = []
        for r, pred, ranked in items:
            read = r["read"]
            exact = r["correct"] == "1"
            n_box, big = box_info(r)
            rows.append({"exact": exact, "ok": pred == r["drug_N"], "conf": float(r["conf"] or 0) / 100, "n_box": n_box, "big": big,
                         "fallback": r["fallback"] == "1", "top1": ranked[0][1], "gap": ranked[0][1] - ranked[1][1]})
            if pred:
                continue
            group = [d for d, v in ranked if v >= ranked[0][1] - EPS]
            has = r["drug_N"] in group
            pairs = {(info[d]["front"], info[d]["back"]) for d in group}
            ties.append({"part": part, "file_name": r["file_name"], "drug_N": r["drug_N"], "name": info[r["drug_N"]]["name"], "truth": r["truth"], "read": read,
                         "kind": tie_kind(has, exact, read), "tie_size": len(group), "truth_in_tie": int(has), "score": round(ranked[0][1], 3),
                         "two_face_ok": int(len(pairs) == len(group)),
                         "tied": " ".join(f"{d}({info[d]['front']}/{info[d]['back']})" for d in group[:6])})
        signal += bins(rows, part, "확신도 conf", lambda x: x["conf"], [(0, .5), (.5, .8), (.8, .9), (.9, .95), (.95, .99), (.99, 1.01)])
        signal += bins(rows, part, "상자 수", lambda x: x["n_box"], [(0, 1), (1, 2), (2, 3), (3, 99)])
        signal += bins(rows, part, "큰 상자", lambda x: int(x["big"]), [(0, 1), (1, 2)])
        signal += bins(rows, part, "1등 점수", lambda x: x["top1"], [(-9, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 9)])
        signal += bins(rows, part, "1등-2등 차이", lambda x: x["gap"], [(0, .01), (.01, .1), (.1, .2), (.2, .4), (.4, 9)])
    write_csv(OUT / "ties.csv", ties)
    summ, drug = [], []
    for part in ("val", "dev"):
        ts = [t for t in ties if t["part"] == part]
        for kind, n in sorted(collections.Counter(t["kind"] for t in ts).items()):
            sub = [t for t in ts if t["kind"] == kind]
            summ.append({"part": part, "kind": kind, "n": n, "drugs": len({t["drug_N"] for t in sub}), "two_face_ok": sum(t["two_face_ok"] for t in sub),
                         "mean_tie_size": round(sum(t["tie_size"] for t in sub) / n, 1), "max_tie_size": max(t["tie_size"] for t in sub)})
        by = collections.defaultdict(list)
        for t in ts:
            by[t["drug_N"]].append(t)
        for d, sub in sorted(by.items(), key=lambda kv: -len(kv[1])):
            drug.append({"part": part, "drug_N": d, "name": info[d]["name"], "n": len(sub), "kinds": dict(collections.Counter(t["kind"][0] for t in sub)),
                         "reads": dict(collections.Counter(t["read"] for t in sub).most_common(4)), "truth": sub[0]["truth"],
                         "tie_size_max": max(t["tie_size"] for t in sub), "two_face_ok": sum(t["two_face_ok"] for t in sub)})
    write_csv(OUT / "tie_summary.csv", summ)
    write_csv(OUT / "tie_by_drug.csv", drug)
    write_csv(OUT / "signal_bins.csv", signal)
    print("--- 동점 칸")
    for s in summ:
        print(s)
    print("--- 약별")
    for d in drug:
        print(d["part"], d["drug_N"], d["name"][:16], d["truth"], d["n"], d["kinds"], d["reads"], "최대", d["tie_size_max"], "두면", d["two_face_ok"])
    print("--- 신호")
    for s in signal:
        print(s["part"], s["signal"], s["from"], s["to"], "n", s["n"], "읽기", s["read_exact"], "약", s["drug_top1"])
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
