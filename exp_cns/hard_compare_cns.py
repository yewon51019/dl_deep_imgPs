# =====================================================================
#  hard_compare_cns.py  —  '박스를 잘 못 찾은 사진'에서 검출기 비교표 (저장된 pred.csv 만 읽음, OCR 다시 안 돌림)
# =====================================================================
#
#  [비교 대상]  같은 사진들(ocr_common_cns.hard_names, 전처리마다 따로)에서
#     기준 = Paddle 기본 (paddle_t0\<prep>_alignlong_line\pred.csv 에서 그 사진들만)
#     vs   paddle_low / dbpp / pse  ×  기준 0.1 / 0.05  (<det>_th<t>\<prep>_alignlong_line_hard\pred.csv)
#     모두 축 정렬(긴 알약 2방향, 나머지 12방향) + 줄 묶기 + PARSeq, 확신도 최고 방향
#  [지표]  정답 수·정답률, 박스 0개(fallback), 아무것도 못 읽음, 정답의 일부만 읽음, 어느 방향이든 정답을 읽은 사진(상한),
#          기준 대비 좋아짐/나빠짐, 한 장당 검출·읽기 시간(모든 방향 합, ms)
#  [저장]  exp_cns\results\ocr_v1\dev\hard_det_compare.csv
#  [실행 예]  python exp_cns\hard_compare_cns.py
# =====================================================================

import argparse
import csv

import numpy as np

import ocr_common_cns as oc

PREPS = ["orig", "K0L17R6-q2-Q2-F4-P1-D4-R4-A4-T4-H8-M2-C6-G12_auto_link17_low85_rim0.6"]
DETS = [("paddle_low", t) for t in (0.1, 0.05)] + [("dbpp", t) for t in (0.1, 0.05)] + [("pse", t) for t in (0.1, 0.05)]


def load(path):
    return {r["file_name"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def stats(rows, base=None):
    """pred.csv 줄들 → 지표 dict. base 가 있으면 같은 사진끼리 좋아짐/나빠짐."""
    n = len(rows)
    ok = sum(int(r["correct"]) for r in rows.values())
    part = sum(1 for r in rows.values() if r["read"] and r["read"] != r["truth"] and r["read"] in r["truth"])
    anyok = sum(any(x.split(":")[1] == r["truth"] for x in r["all_dirs"].split(" | ")) for r in rows.values())
    d = dict(n=n, n_correct=ok, acc=round(100 * ok / n, 1) if n else "",
             fallback=sum(int(r["fallback"]) for r in rows.values()),
             empty=sum(1 for r in rows.values() if not r["read"]), partial=part, any_dir_correct=anyok,
             ms_det=round(float(np.mean([float(r["det_ms_12dir"]) for r in rows.values()])), 0),
             ms_rec=round(float(np.mean([float(r["rec_ms_12dir"]) for r in rows.values()])), 0))
    if base is not None:
        d["better"] = sum(int(r["correct"]) > int(base[f]["correct"]) for f, r in rows.items())
        d["worse"] = sum(int(r["correct"]) < int(base[f]["correct"]) for f, r in rows.items())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])
    a = ap.parse_args()
    out_rows = []
    for prep in PREPS:
        hard = oc.hard_names(a.split, prep)
        allb = load(oc.OCR_OUT / a.split / "paddle_t0" / f"{prep}_alignlong_line" / "pred.csv")
        base = {f: allb[f] for f in hard}
        out_rows.append(dict(prep=prep[:7], det="paddle 기본 (기준)", thresh="", **stats(base), better="", worse=""))
        for det, t in DETS:
            f = oc.out_dir(a.split, f"{det}_th{t:g}", prep, 0, 12, "long", "_line_hard") / "pred.csv"
            if not f.exists():
                print(f"  (없음) {f}")
                continue
            out_rows.append(dict(prep=prep[:7], det=det, thresh=t, **stats(load(f), base)))
    path = oc.OCR_OUT / a.split / "hard_det_compare.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장: {path}")
    for r in out_rows:
        print(f"  {r['prep']:7s} {r['det']:18s} {str(r['thresh']):5s} 정답 {r['n_correct']:3d}/{r['n']} ({r['acc']}%)  "
              f"박스0 {r['fallback']:3d}  빈칸 {r['empty']:3d}  일부만 {r['partial']:3d}  상한 {r['any_dir_correct']:3d}  "
              f"+{r['better']}/-{r['worse']}  검출 {r['ms_det']:.0f}ms 읽기 {r['ms_rec']:.0f}ms")


if __name__ == "__main__":
    main()
