# =====================================================================
#  det_ft3_compare_cns.py  —  K0L17 사진으로 학습한 검출기(det_ft3) 채택 여부 판단 (저장된 pred.csv 만 읽음, 개발용 468장)
# =====================================================================
#
#  [왜]  전처리 사진으로 박스를 찾으면 정답률이 떨어졌는데(65.8%), 검출기를 원본으로만 학습해서일 수 있다.
#        전처리 사진으로 학습한 검출기로 같은 비교를 다시 해서 이 설명이 맞는지 본다.
#  [비교]  세 경우 모두 읽기는 원본 사진 + 사전학습 PARSeq, 축 정렬 + 줄 묶기, 상자 넓히기(unclip) 2.0
#     기준    원본으로 학습한 검출기(det_ft1) + 원본 사진에서 박스 찾기
#     ft1K0  원본으로 학습한 검출기(det_ft1) + K0L17 사진에서 박스 찾기  (이미 있는 결과)
#     ft3K0  K0L17 로 학습한 검출기(det_ft3) + K0L17 사진에서 박스 찾기  (이번 시험)
#  [채택 규칙, 결과를 보기 전에 정함]  ft3K0 이 기준보다 5장 이상 더 맞히고, 좋아진 장수 > 나빠진 장수일 때만 채택
#  [주의]  정답률은 박스 위치를 직접 잰 값이 아니라, 박스 + 읽기를 합친 간접 지표다.
#  [저장]  exp_cns\results\det_ft3\compare.csv  (조건별 표), changed.csv  (기준과 맞고 틀림이 달라진 사진)
#          이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\det_ft3_compare_cns.py   (저장소 맨 위에서)
# =====================================================================

import csv
import json
import os

import common_cns as cc
import read_order_cns as ro

OCR = cc.RESULTS / "ocr_v1" / "dev"
OUT = cc.RESULTS / "det_ft3"
BIG = 400 * 400 * 0.5                             # 상자 넓이가 이 값 이상이면 '큰 상자' (알약 전체를 덮은 상자)
RUNS = {"기준 (원본 학습 + 원본 박스)": OCR / "paddle_ft1_u2.0" / "orig_alignlong_line",
        "ft1K0 (원본 학습 + K0L17 박스)": OCR / "paddle_ft1_u2.0" / "orig_alignlong_boxK0L17-c_line",
        "ft3K0 (K0L17 학습 + K0L17 박스)": OCR / "paddle_ft3_u2.0" / "orig_alignlong_boxK0L17-c_line"}


def load(folder):
    """pred.csv → {파일명: 한 줄}. 468장이 다 있는지 확인."""
    rows = {r["file_name"]: r for r in csv.DictReader(open(folder / "pred.csv", encoding="utf-8-sig"))}
    assert len(rows) == 468, (folder, len(rows))
    return rows


def has_big(r):
    """고른 방향의 상자 중 큰 상자가 있으면 True. best_boxes 한 칸 = [x0, y0, x1, y1, 읽은 글자, ...]"""
    return any((b[2] - b[0]) * (b[3] - b[1]) >= BIG for b in json.loads(r["best_boxes"] or "[]"))


def pct(rows):
    """맞은 비율(%). 사진이 없으면 0."""
    return round(100 * sum(r["correct"] == "1" for r in rows) / max(1, len(rows)), 1)


def main():
    assert not (OUT / "compare.csv").exists(), f"이미 있음, 덮어쓰지 않음: {OUT / 'compare.csv'}"
    os.makedirs(OUT, exist_ok=True)
    data = {k: load(v) for k, v in RUNS.items()}
    base_name = next(iter(RUNS))
    base = data[base_name]
    table = []
    for name, d in data.items():
        v = list(d.values())
        better = [f for f in d if d[f]["correct"] == "1" and base[f]["correct"] != "1"]
        worse = [f for f in d if d[f]["correct"] != "1" and base[f]["correct"] == "1"]
        table.append({
            "조건": name,
            "맞음": sum(r["correct"] == "1" for r in v),
            "완전일치": pct(v),
            "순서무시": round(100 * sum(ro.order_free(r) for r in v) / len(v), 1),
            "좋아짐": len(better),
            "나빠짐": len(worse),
            "긴알약": pct([r for r in v if r["aligned"] == "1"]),
            "그외": pct([r for r in v if r["aligned"] != "1"]),
            "카메라75": pct([r for r in v if r["camera_la"] == "75"]),
            "카메라90": pct([r for r in v if r["camera_la"] == "90"]),
            "박스0개": sum(r["fallback"] == "1" for r in v),
            "큰상자": sum(has_big(r) for r in v),
        })
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    new = data[list(RUNS)[2]]
    with open(OUT / "changed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "drug_N", "truth", "change", "read_base", "read_ft3", "angle_base", "angle_ft3"])
        for fn in sorted(new):
            a, b = base[fn]["correct"] == "1", new[fn]["correct"] == "1"
            if a != b:
                w.writerow([fn, new[fn]["drug_N"], new[fn]["truth"], "좋아짐" if b else "나빠짐",
                            base[fn]["read"], new[fn]["read"], base[fn]["angle"], new[fn]["angle"]])
    for row in table:
        print(row)
    t = table[2]
    gain = t["맞음"] - table[0]["맞음"]
    ok = gain >= 5 and t["좋아짐"] > t["나빠짐"]
    print(f"차이 {gain:+d}장, 좋아짐 {t['좋아짐']} / 나빠짐 {t['나빠짐']} → {'채택' if ok else '채택 안 함'}")
    print(f"저장: {OUT / 'compare.csv'}, {OUT / 'changed.csv'}")


if __name__ == "__main__":
    main()
