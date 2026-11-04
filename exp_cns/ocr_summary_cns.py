# =====================================================================
#  ocr_summary_cns.py  —  ocr_cns.py 결과(pred.csv)들을 표 하나로 모으기
# =====================================================================
#
#  [읽는 것]  exp_cns\results\ocr_v1\<split>\<글자 찾기>\<전처리>\pred.csv  (시험 운전 '__try' 와 재확인 '_recheck'·'_gpu' 폴더는 뺌)
#  [저장]
#    summary.csv          : 조합마다 정답률·CER·fallback·O·0 끼어듦·카메라별 정답률·한 장당 시간
#                           + 기준(같은 글자 찾기의 원본) 대비 좋아진/나빠진 사진 수
#    summary_by_form.csv  : 조합 × 제형별 정답률 (연질캡슐 등)
#  [주의]  개발용 468장 = 36종이라 사진 1장 ≈ 0.2%p, 약 하나(13장) ≈ 2.8%p. 몇 %p 차이는 우연 범위일 수 있음
#  [실행 예]  python exp_cns\ocr_summary_cns.py --split dev
# =====================================================================

import argparse
import csv
from collections import defaultdict

import numpy as np

import common_cns as cc

OCR_OUT = cc.RESULTS / "ocr_v1"
SKIP = ("__try", "_recheck", "_gpu")


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def mean(v):
    v = [float(x) for x in v if x != ""]
    return round(float(np.mean(v)), 1) if v else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    root = OCR_OUT / a.split
    runs = {}
    for f in sorted(root.glob("*/*/pred.csv")):
        det, prep = f.parent.parent.name, f.parent.name
        if any(s in prep for s in SKIP):
            continue
        runs[(det, prep)] = load(f)

    rows, form_rows = [], []
    for (det, prep), rs in runs.items():
        base = {r["file_name"]: r for r in runs.get((det, "orig"), [])}
        n = len(rs)
        cam = {c: [int(r["correct"]) for r in rs if r["camera_la"] == c] for c in ("75", "90")}
        better = sum(1 for r in rs if r["file_name"] in base and int(r["correct"]) > int(base[r["file_name"]]["correct"]))
        worse = sum(1 for r in rs if r["file_name"] in base and int(r["correct"]) < int(base[r["file_name"]]["correct"]))
        rows.append({"det": det, "prep": prep, "n": n,
                     "acc": round(100 * np.mean([int(r["correct"]) for r in rs]), 1),
                     "n_correct": sum(int(r["correct"]) for r in rs),
                     "cer": round(float(np.mean([float(r["cer"]) for r in rs])), 3),
                     "fallback": sum(int(r["fallback"]) for r in rs),
                     "extra_o0": sum(int(r["extra_o0"]) for r in rs),
                     "acc_cam75": round(100 * np.mean(cam["75"]), 1) if cam["75"] else "",
                     "acc_cam90": round(100 * np.mean(cam["90"]), 1) if cam["90"] else "",
                     "vs_orig_better": better if prep != "orig" else "", "vs_orig_worse": worse if prep != "orig" else "",
                     "ms_prep": mean(r["prep_ms"] for r in rs), "ms_det_12dir": mean(r["det_ms_12dir"] for r in rs),
                     "ms_rec_12dir": mean(r["rec_ms_12dir"] for r in rs)})
        by = defaultdict(list)
        for r in rs:
            by[r["form"] or "(빈칸)"].append(int(r["correct"]))
        for form, v in sorted(by.items()):
            form_rows.append({"det": det, "prep": prep, "form": form, "n": len(v), "acc": round(100 * np.mean(v), 1)})

    for name, data in (("summary.csv", rows), ("summary_by_form.csv", form_rows)):
        with open(root / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"저장: {root / 'summary.csv'}, {root / 'summary_by_form.csv'}")
    for r in rows:
        print(f"  {r['det']:9s} {r['prep'][:40]:40s} 정답률 {r['acc']:5.1f}% ({r['n_correct']}/{r['n']})  CER {r['cer']:.3f}  "
              f"fallback {r['fallback']}  O0 {r['extra_o0']}  원본 대비 +{r['vs_orig_better']}/-{r['vs_orig_worse']}")


if __name__ == "__main__":
    main()
