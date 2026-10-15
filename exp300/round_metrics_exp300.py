# =====================================================================
#  round_metrics_exp300.py  —  둥근 알약 그림 결과(round_boxes.csv)로 방식별 성능 지표 표 만들기 (몇 초)
# =====================================================================
#
#  [입력]  viz_round_exp300.py 가 만든 round_boxes.csv  (OCR 을 다시 돌리지 않음)
#  [지표]  방식 3개(CRAFT+PARSeq / Paddle 기본 / Paddle 0.0) × 지표 5개
#     정답률  : 읽은 각인이 정답 각인과 완전히 같은 사진 비율 (각인 수준)
#     정확도  : 읽은 각인으로 DB 에서 찾은 약이 정답 약인 비율 (약 수준, 실험과 같은 edit 매칭)
#     정밀도 / 민감도(재현율) / F1 : 약 종류마다 계산해 평균(macro) — 실험 보고서와 같은 방식
#   · 정답이 '마크'뿐인 사진은 실험과 같이 제외해요.
#   · 0° 에서 한 번만 읽은 값이라 본 실험(12방향 다수결)의 둥근 알약 점수와는 달라요.
#
#  [출력]  --out 엑셀 파일 (시트: 요약 / 사진별 / 지표 설명)
#     요약 시트는 전체 · 시험 세트만 · 카메라 60° · 카메라 90° 로 나눠서 보여 줘요.
#
#  [실행 예]
#     python round_metrics_exp300.py --boxes viz_round/round_boxes.csv --manifest manifest.csv \
#            --db search_db.csv --out viz_round/round_metrics.xlsx
# =====================================================================

import argparse
import csv

import run_exp300 as base
from evaluate_exp300 import drug_metrics

METHODS = [("craft", "CRAFT+PARSeq"), ("paddle", "PaddleOCR 기본 검출 기준"), ("paddle_t0", "PaddleOCR 검출 기준 0.0")]
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def build_rows(boxes_path, manifest_path, db_path):
    with open(manifest_path, encoding="utf-8-sig") as f:
        drug_of = {r["file_name"]: r["drug_N"] for r in csv.DictReader(f)}
    db = base.load_db(db_path)
    out = []
    with open(boxes_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("truth_norm"):                       # 정답이 '마크'뿐 → 채점 제외 (실험과 같음)
                continue
            read = r.get("read", "")
            pred = base.identify(read, "", "", db)[1] if read else ""     # edit 매칭 (모양·색 안 씀)
            out.append({"method": r["method"], "file_name": r["file_name"], "split": r.get("split", ""),
                        "camera_la": r.get("camera_la", ""), "truth_imprint": r["truth_norm"], "read": read,
                        "truth_drug": drug_of.get(r["file_name"], ""), "pred_drug": pred or NONE_LABEL,
                        "imprint_ok": int(read == r["truth_norm"]),
                        "drug_ok": int(pred == drug_of.get(r["file_name"], ""))})
    return out


def scores(rows):
    """[정답률, 정확도, 정밀도, 민감도, F1] (0~1), 사진 수"""
    if not rows:
        return None, 0
    acc, p, r, f = drug_metrics([x["truth_drug"] for x in rows], [x["pred_drug"] for x in rows])
    imp = sum(x["imprint_ok"] for x in rows) / len(rows)
    return [imp, acc, p, r, f], len(rows)


def groups(rows):
    cams = sorted({x["camera_la"] for x in rows if x["camera_la"]})
    g = [("전체 (정답 각인 있는 둥근 알약)", lambda x: True),
         ("시험 세트만 (보고서 둥근 알약 99장과 같은 사진)", lambda x: x["split"] == "test")]
    g += [(f"카메라 {c}°", (lambda c: lambda x: x["camera_la"] == c)(c)) for c in cams]
    return g


def write_xlsx(rows, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    F = "Arial"
    thin = Side(style="thin", color="BFBFBF")
    bd = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="1F3864")
    grp_fill = PatternFill("solid", fgColor="DCE6F5")
    wb = Workbook()

    # ---- 사진별 (원자료)
    det = wb.active
    det.title = "사진별"
    cols = ["method", "file_name", "split", "camera_la", "truth_imprint", "read", "truth_drug", "pred_drug",
            "imprint_ok", "drug_ok"]
    det.append(cols)
    for c in det[1]:
        c.font = Font(name=F, bold=True, color="FFFFFF"); c.fill = head_fill
    for x in rows:
        det.append([x[k] for k in cols])
    for row in det.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=F, size=10)
    for i, w in enumerate([12, 34, 8, 10, 16, 18, 12, 12, 11, 9], 1):
        det.column_dimensions[chr(64 + i)].width = w
    det.freeze_panes = "A2"
    n_det = len(rows) + 1

    # ---- 요약
    ws = wb.create_sheet("요약", 0)
    ws["A1"] = "둥근 알약 검출·읽기 성능 (0° 한 번 읽기, 정답 '마크'뿐인 사진 제외)"
    ws["A1"].font = Font(name=F, size=14, bold=True)
    ws["A2"] = ("정답률·정확도는 '사진별' 시트를 세는 수식이고, 정밀도·민감도·F1(약 종류별 평균)은 Python 으로 계산한 값이에요. "
                "본 실험(12방향 다수결)의 둥근 알약 점수와는 달라요.")
    ws["A2"].font = Font(name=F, size=9, italic=True, color="595959")
    r0 = 4
    for gname, cond in groups(rows):
        ws.cell(r0, 1, gname).font = Font(name=F, size=11, bold=True)
        ws.cell(r0, 1).fill = grp_fill
        for k in range(2, 3 + len(METRICS)):
            ws.cell(r0, k).fill = grp_fill
        r0 += 1
        for k, h in enumerate(["방식", "사진 수"] + METRICS, 1):
            c = ws.cell(r0, k, h)
            c.font = Font(name=F, size=10, bold=True, color="FFFFFF"); c.fill = head_fill; c.border = bd
            c.alignment = Alignment(horizontal="center")
        for m, label in METHODS:
            r0 += 1
            sub = [x for x in rows if x["method"] == m and cond(x)]
            vals, n = scores(sub)
            ws.cell(r0, 1, label).font = Font(name=F, size=10)
            ws.cell(r0, 2, n).font = Font(name=F, size=10)
            if vals is None:
                continue
            # 정답률·정확도는 사진별 시트에서 직접 세는 수식 (전체 묶음일 때)  / 나머지 묶음은 계산값
            if gname.startswith("전체"):
                ws.cell(r0, 3, f'=AVERAGEIFS(사진별!$I$2:$I${n_det},사진별!$A$2:$A${n_det},"{m}")')
                ws.cell(r0, 4, f'=AVERAGEIFS(사진별!$J$2:$J${n_det},사진별!$A$2:$A${n_det},"{m}")')
            elif gname.startswith("시험"):
                ws.cell(r0, 3, f'=AVERAGEIFS(사진별!$I$2:$I${n_det},사진별!$A$2:$A${n_det},"{m}",사진별!$C$2:$C${n_det},"test")')
                ws.cell(r0, 4, f'=AVERAGEIFS(사진별!$J$2:$J${n_det},사진별!$A$2:$A${n_det},"{m}",사진별!$C$2:$C${n_det},"test")')
            else:
                cam = gname.split()[1].rstrip("°")
                ws.cell(r0, 3, f'=AVERAGEIFS(사진별!$I$2:$I${n_det},사진별!$A$2:$A${n_det},"{m}",사진별!$D$2:$D${n_det},"{cam}")')
                ws.cell(r0, 4, f'=AVERAGEIFS(사진별!$J$2:$J${n_det},사진별!$A$2:$A${n_det},"{m}",사진별!$D$2:$D${n_det},"{cam}")')
            for k, v in enumerate(vals[2:], 5):
                ws.cell(r0, k, round(v, 4)).font = Font(name=F, size=10, color="0000FF")   # 계산값(파란 글씨)
            for k in range(1, 3 + len(METRICS)):
                ws.cell(r0, k).border = bd
                if k >= 3:
                    ws.cell(r0, k).number_format = "0.0%"
                if k in (3, 4):
                    ws.cell(r0, k).font = Font(name=F, size=10)                # 수식(검은 글씨)
        r0 += 2
    for i, w in enumerate([46, 9, 10, 10, 10, 14, 10], 1):
        ws.column_dimensions[chr(64 + i)].width = w

    # ---- 지표 설명
    ex = wb.create_sheet("지표 설명")
    lines = [("지표", "뜻", "계산"),
             ("정답률", "읽은 각인이 정답 각인과 완전히 같은 비율 (각인 수준)", "imprint_ok 평균"),
             ("정확도", "읽은 각인으로 DB 에서 찾은 약이 정답 약인 비율 (약 수준)", "drug_ok 평균"),
             ("정밀도", "'A약'이라고 판정한 것 중 진짜 A약 비율, 약 종류별 평균(macro)", "TP / (TP+FP)"),
             ("민감도(재현율)", "진짜 A약 중 A약으로 맞힌 비율, 약 종류별 평균(macro)", "TP / (TP+FN)"),
             ("F1", "정밀도와 민감도의 조화평균, 약 종류별 평균(macro)", "2PR / (P+R)"),
             ("주의", "0° 에서 한 번만 읽은 결과. 본 실험은 12방향 다수결이라 수치가 달라요.", ""),
             ("주의", "약 판별은 실험과 같은 edit 매칭(편집거리 유사도 ≥ 0.3, 1등 동점이면 판정없음)", "")]
    for row in lines:
        ex.append(row)
    for c in ex[1]:
        c.font = Font(name=F, bold=True, color="FFFFFF"); c.fill = head_fill
    for row in ex.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=F, size=10)
    ex.column_dimensions["A"].width = 16
    ex.column_dimensions["B"].width = 70
    ex.column_dimensions["C"].width = 18
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="viz_round_exp300.py 가 만든 round_boxes.csv")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="round_metrics.xlsx")
    a = ap.parse_args()
    rows = build_rows(a.boxes, a.manifest, a.db)
    print(f"채점 대상: 둥근 알약 {len({x['file_name'] for x in rows})}장 (정답이 '마크'뿐인 사진 제외)\n")
    for gname, cond in groups(rows):
        print(f"[{gname}]")
        print(f"  {'방식':<26} {'사진':>4} " + " ".join(f"{m:>8}" for m in METRICS))
        for m, label in METHODS:
            vals, n = scores([x for x in rows if x["method"] == m and cond(x)])
            if vals:
                print(f"  {label:<26} {n:4d} " + " ".join(f"{v * 100:7.1f}%" for v in vals))
        print()
    print("※ 0° 한 번 읽기 기준. 본 실험(12방향 다수결)의 둥근 알약 점수와는 달라요.")
    write_xlsx(rows, a.out)
    print(f"\n저장 완료: {a.out}  (시트: 요약 / 사진별 / 지표 설명)")


if __name__ == "__main__":
    main()
