# =====================================================================
#  final_report_exp300.py  —  평가용 최종 결과 채점·오류 분류·개발용 비교·엑셀 (OCR 다시 안 돌림)
# =====================================================================
#
#  [입력]  final_exp300.py 가 만든 final_dev.csv / final_test.csv, 그림 근거 CSV(사람이 사진을 보고 적은 오류 원인)
#  [지표]  정답률(읽은 각인 = 정답 각인), 정확도, 정밀도·민감도(재현율)·F1 (약 종류별 평균, macro)
#          묶음: 전체 / 둥근 / 길쭉 / 카메라 60° / 카메라 90°
#          · V3+색·C4 의 '읽은 각인' = 고른 약에 가장 높은 점수를 준 후보 글자 (C4 가 dbscore 로 정한 경우 그 DB 각인)
#  [오류 분류]  각인 같은 약 > 마크뿐 > 라벨 앞뒤 뒤바뀜 > 각인 일부만 보임 > 박스 못 찾음 > 잘못 읽음 (앞 칸이 먼저)
#               가운데 세 칸은 그림 근거 CSV 의 visual 값, '박스 못 찾음' = 길 A 방향에서 CRAFT 박스 0개
#  [저장]  --out (test_final): metrics_{dev,test}.csv, dev_vs_test.csv, error_table.csv, error_list.csv, up_down_test.csv
#          --report-dir (viz_final_report): final_report.xlsx (+ --charts 로 받은 그림을 시트에 붙임)
# =====================================================================

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from evaluate_exp300 import drug_metrics

METHODS = ["V3+색", "C4", "A", "vote"]
LABELS = {"V3+색": "보정 V3 + 색 동점 깨기 (주 방식)", "C4": "C4: V3 → 색 → dbscore", "A": "길 A: ② + 박스 순서 바꾸기",
          "vote": "12방향 다수결 (지금 방식)"}
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]
GROUPS = [("전체", lambda r: True), ("둥근", lambda r: r["shape"] == "둥근"), ("길쭉", lambda r: r["shape"] == "길쭉"),
          ("카메라 60°", lambda r: r["camera_la"] == "60"), ("카메라 90°", lambda r: r["camera_la"] == "90")]
ERROR_ORDER = ["각인 같은 약", "마크뿐", "라벨 앞뒤 뒤바뀜", "각인 일부만 보임", "박스 못 찾음", "잘못 읽음"]
NONE_LABEL = "(판정없음)"


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metrics(rows):
    acc, p, r, f = drug_metrics([x["truth_drug"] for x in rows], [x["pred_drug"] for x in rows])
    imp = float(np.mean([x["judge"] == "O" for x in rows]))
    return [imp, acc, p, r, f], sum(x["pred_drug"] == NONE_LABEL for x in rows)


def metric_table(rows, split):
    out = []
    for g, cond in GROUPS:
        for m in METHODS:
            sub = [r for r in rows if r["method"] == m and cond(r)]
            if not sub:
                continue
            v, none = metrics(sub)
            out.append({"split": split, "group": g, "n": len(sub), "method": m, "label": LABELS[m],
                        **{k: round(s * 100, 1) for k, s in zip(METRICS, v)}, "판정없음": none})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def classify(rows, notes, same_imprint):
    """틀린 사진 → 원인. 돌려주는 값: [{split, method, file_name, cause, ...}]"""
    out = []
    for r in rows:
        if r["drug_ok"] == "1":
            continue
        v = notes.get(r["file_name"], {}).get("visual", "")
        if r["truth_imprint"] in same_imprint:
            c = "각인 같은 약"
        elif v in ("마크뿐", "라벨 앞뒤 뒤바뀜", "각인 일부만 보임"):
            c = v
        elif r["n_box_A"] == "0":
            c = "박스 못 찾음"
        else:
            c = "잘못 읽음"
        out.append({"split": r["split"], "method": r["method"], "file_name": r["file_name"], "cause": c,
                    "truth_drug": r["truth_drug"], "truth_imprint": r["truth_imprint"], "pred_drug": r["pred_drug"],
                    "read": r["read"], "decided_by": r["decided_by"], "note": notes.get(r["file_name"], {}).get("note", "")})
    return out


def write_xlsx(path, sheets, images):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    F = "Malgun Gothic"
    thin = Side(style="thin", color="BFBFBF")
    head = PatternFill("solid", fgColor="1F3864")
    hl = PatternFill("solid", fgColor="FFF2CC")
    wb = Workbook()
    wb.remove(wb.active)
    for title, (note, rows) in sheets.items():
        ws = wb.create_sheet(title)
        ws["A1"] = title
        ws["A1"].font = Font(name=F, size=14, bold=True)
        ws["A2"] = note
        ws["A2"].font = Font(name=F, size=10, italic=True, color="595959")
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        for j, c in enumerate(cols, 1):
            cell = ws.cell(4, j, c)
            cell.font = Font(name=F, size=11, bold=True, color="FFFFFF")
            cell.fill = head
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        for i, r in enumerate(rows, 5):
            main = r.get("method") in ("V3+색",) or "주 방식" in str(r.get("label", ""))
            for j, c in enumerate(cols, 1):
                v = r.get(c, "")
                cell = ws.cell(i, j, v)
                cell.font = Font(name=F, size=11, bold=main)
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                if main:
                    cell.fill = hl
                if isinstance(v, float):
                    cell.number_format = "0.0"
        for j, c in enumerate(cols, 1):
            width = max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) + 2
            ws.column_dimensions[ws.cell(4, j).column_letter].width = min(48, max(9, width * 1.15))
        ws.freeze_panes = "A5"
    if images:
        ws = wb.create_sheet("그림")
        ws["A1"] = "방식별 비교 그림"
        ws["A1"].font = Font(name=F, size=14, bold=True)
        row = 3
        for p in images:
            if os.path.exists(p):
                img = XLImage(p)
                scale = min(1.0, 900 / img.width)
                img.width, img.height = int(img.width * scale), int(img.height * scale)
                ws.add_image(img, f"A{row}")
                row += int(img.height / 20) + 3
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="test_final/final_dev.csv")
    ap.add_argument("--test", required=True, help="test_final/final_test.csv")
    ap.add_argument("--dev-notes", required=True)
    ap.add_argument("--test-notes", required=True)
    ap.add_argument("--db", required=True, help="search_db_v3.csv")
    ap.add_argument("--dev-all", required=True, help="combined_cv410/combined_metrics.csv (개발용 모든 방식)")
    ap.add_argument("--dev-align", required=True, help="viz_align2/align_metrics.csv (개발용 정렬 방식, 카메라별 포함)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--charts", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.report_dir, exist_ok=True)

    dev, test = load(a.dev), load(a.test)
    notes = {r["file_name"]: r for p in (a.dev_notes, a.test_notes) for r in load(p)}
    owners = defaultdict(set)
    for r in load(a.db):
        for col in ("front_norm", "back_norm"):
            t = "".join(ch for ch in r[col].upper() if ch.isalnum())
            if t:
                owners[t].add(r["drug_N"])
    same_imprint = {t for t, d in owners.items() if len(d) > 1}

    mdev, mtest = metric_table(dev, "개발용"), metric_table(test, "평가용")
    write_csv(os.path.join(a.out, "metrics_dev.csv"), mdev)
    write_csv(os.path.join(a.out, "metrics_test.csv"), mtest)

    # 개발용 vs 평가용 나란히
    side = []
    key = lambda rows: {(r["group"], r["method"]): r for r in rows}
    kd, kt = key(mdev), key(mtest)
    for g, _ in GROUPS:
        for m in METHODS:
            d, t = kd.get((g, m)), kt.get((g, m))
            if not d or not t:
                continue
            row = {"group": g, "method": m, "label": LABELS[m], "n_dev": d["n"], "n_test": t["n"]}
            for k in METRICS:
                row[f"{k}_개발"], row[f"{k}_평가"], row[f"{k}_차이"] = d[k], t[k], round(t[k] - d[k], 1)
            side.append(row)
    write_csv(os.path.join(a.out, "dev_vs_test.csv"), side)

    # 남은 오류
    errs = classify(dev + test, notes, same_imprint)
    write_csv(os.path.join(a.out, "error_list.csv"), errs)
    etab = []
    for split, rows in (("개발용", dev), ("평가용", test)):
        n = len({r["file_name"] for r in rows})
        for m in METHODS:
            cnt = defaultdict(int)
            for e in errs:
                if e["method"] == m and e["split"] == ("dev" if split == "개발용" else "test"):
                    cnt[e["cause"]] += 1
            etab.append({"split": split, "n": n, "method": m, "label": LABELS[m], **{c: cnt[c] for c in ERROR_ORDER},
                         "합계": sum(cnt.values())})
    write_csv(os.path.join(a.out, "error_table.csv"), etab)

    # 평가용 좋아짐/나빠짐
    by = {(r["file_name"], r["method"]): r for r in test}
    files = sorted({r["file_name"] for r in test})
    ud = []
    for m in METHODS:
        row = {"method": m, "label": LABELS[m]}
        for ref in ("vote", "A"):
            row[f"좋아짐_vs_{ref}"] = sum(by[(f, m)]["drug_ok"] == "1" and by[(f, ref)]["drug_ok"] == "0" for f in files)
            row[f"나빠짐_vs_{ref}"] = sum(by[(f, m)]["drug_ok"] == "0" and by[(f, ref)]["drug_ok"] == "1" for f in files)
        ud.append(row)
    write_csv(os.path.join(a.out, "up_down_test.csv"), ud)
    decided = defaultdict(lambda: defaultdict(int))
    for r in test:
        if r["method"] in ("V3+색", "C4"):
            decided[r["method"]][r["decided_by"]] += 1

    # 화면 출력
    for title, rows in (("평가용 237장", mtest), ("개발용 51장 (색 대표값: 자기 사진 뺀 개발용)", mdev)):
        print(f"\n[{title}]")
        cur = None
        for r in rows:
            if r["group"] != cur:
                cur = r["group"]
                print(f"  ({cur} {r['n']}장)  " + " ".join(f"{k:>9}" for k in METRICS) + "  판정없음")
            print(f"    {LABELS[r['method']]:<30} " + " ".join(f"{r[k]:8.1f}%" for k in METRICS) + f"  {r['판정없음']:5d}")
    print("\n[남은 오류 (틀린 사진 수)]  " + " | ".join(ERROR_ORDER))
    for r in etab:
        print(f"  {r['split']} {LABELS[r['method']]:<30} " + "  ".join(f"{r[c]:3d}" for c in ERROR_ORDER) + f"   합계 {r['합계']}")
    print("\n[평가용 좋아짐/나빠짐]")
    for r in ud:
        print(f"  {LABELS[r['method']]:<30} 다수결 대비 +{r['좋아짐_vs_vote']}/-{r['나빠짐_vs_vote']}   길A 대비 +{r['좋아짐_vs_A']}/-{r['나빠짐_vs_A']}")
    print("\n[평가용에서 무엇으로 정했나]", {m: dict(v) for m, v in decided.items()})

    # 엑셀
    dev_all = load(a.dev_all)
    dev_align = [r for r in load(a.dev_align) if r["group"].startswith("둥근·카메라")]
    note_env = "모든 숫자: 같은 환경 (PARSeq + P0, OpenCV 4.10, 로컬 GPU). 개발용 51장 / 평가용 237장. 단위 %."
    sheets = {
        "요약_개발vs평가": (note_env + " 평가용은 사전 기록한 4가지 방식만 한 번 돌림.", side),
        "평가용_조건별": (note_env, mtest),
        "개발용_조건별": (note_env + " 색 대표값은 자기 사진을 뺀 개발용(leave-one-out).", mdev),
        "개발용_모든방식": (note_env + " 정렬 13가지 + 색·합친 방식. 평가용에는 돌리지 않음 (사전 기록).",
                       [{k: v for k, v in r.items()} for r in dev_all]),
        "개발용_정렬_카메라별": (note_env + " 둥근 알약만, 정렬 13가지 (viz_align2).", dev_align),
        "남은오류": (note_env + " 원인 순서: " + " > ".join(ERROR_ORDER) + ". 가운데 세 칸은 그림 근거 CSV.", etab),
        "평가용_좋아짐나빠짐": (note_env, ud),
        "오류_사진별": (note_env, errs),
        "지표설명": ("", [
            {"지표": "정답률", "뜻": "읽은 각인이 정답 각인과 완전히 같은 비율 (V3+색·C4 는 고른 약에 최고점을 준 후보 글자)"},
            {"지표": "정확도", "뜻": "찾은 약이 정답 약인 비율"},
            {"지표": "정밀도", "뜻": "'A약'이라고 판정한 것 중 진짜 A약 비율, 약 종류별 평균"},
            {"지표": "민감도(재현율)", "뜻": "진짜 A약 중 A약으로 맞힌 비율, 약 종류별 평균"},
            {"지표": "F1", "뜻": "정밀도와 민감도의 조화평균, 약 종류별 평균"},
            {"지표": "판정없음", "뜻": "약을 정하지 못한 사진 수 (틀린 것으로 셈)"}]),
    }
    xlsx = os.path.join(a.report_dir, "final_report.xlsx")
    write_xlsx(xlsx, sheets, a.charts)
    print(f"\n저장 완료: {a.out} (metrics_*.csv, dev_vs_test.csv, error_table.csv, error_list.csv, up_down_test.csv), {xlsx}")


if __name__ == "__main__":
    main()
