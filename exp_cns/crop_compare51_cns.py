# =====================================================================
#  crop_compare51_cns.py  —  자르기 방법 고르기: exp300 개발용 51장에서만 비교 (나머지 237장은 채점하지 않음)
# =====================================================================
#
#  [비교하는 방식]  읽기·약 판별은 모두 base 와 같음 (ft1 검출 u2.0 + 줄 묶기 + 사전학습 PARSeq, match2 λ 0.95, 길이 ≤ 30, w 0.02)
#     full   = 자르지 않은 전체 사진 (지금 공식 결과의 그 51장, 저장된 읽기·CNN 확률을 그대로 씀)
#     crop1  = 기존 영역으로 자른 사진 (crop_v1)
#     crop2  = 넉넉 자르기로 자른 사진 (crop_v2)
#     union  = crop1 + crop2 후보 합치기
#              · 사진마다 확신도가 더 높은 자르기의 '고른 방향'을 답으로 함 (겉모습 CNN 확률도 그쪽 사진 것)
#              · 다른 자르기에서 읽은 글자는 '다른 방향 후보'에 더함 (점수 × 0.95)
#  [고르는 규칙, 결과를 보기 전에 정함]
#     약 정확도가 높은 쪽. 단 사진 1장 = 약 2%p 라, union 이 crop2 보다 2장 이상 많이 맞히지 못하면
#     더 간단하고 빠른 crop2 를 고른다. crop1 은 잘림을 알고 있으므로 3장 이상 차이로 이길 때만 고른다.
#  [주의]  이 51장은 288장의 일부라 완전히 깨끗한 선택은 아니다. 사후 보조 지표용이고, 공식 결과 60.8% 는 바꾸지 않는다.
#  [지표]  약 정확도·판정없음·3등 안, 읽기 완전일치·순서 무시(read_order_cns 와 같은 정의), 긴 알약 / 그 외, 카메라 60°·90°, 흰 알약, full 대비 좋아짐/나빠짐
#  [저장]  exp_cns\results\crop_compare51\  summary.csv, pred_all.csv (사진마다 방식별 읽기·예측), in\ (방식별 입력 파일)
#  [실행]  python exp_cns\crop_compare51_cns.py   (저장소 맨 위에서, ocr_cns·attr_cnn_cns 를 먼저 돌린 뒤)
# =====================================================================

import csv
import itertools
import json
import os

import match_cns as m1
import match2_cns as m2

OCR = m1.oc.OCR_OUT
OUT = m1.cc.RESULTS / "crop_compare51"
ATTR = m1.cc.RESULTS / "attr_cnn_v1"
KEY = "main_wA_0.02"
READ = "paddle_ft1_u2.0/orig_alignlong_line/pred.csv"
SPLITS = {"crop1": "exp300dev51_crop1", "crop2": "exp300dev51_crop2"}


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def merge(a, b):
    """같은 사진의 두 읽기 결과 → 합친 줄. 확신도 높은 쪽이 답, 낮은 쪽의 모든 방향 읽기는 '다른 방향 후보'로 더함."""
    win, lose = (a, b) if float(a["conf"]) >= float(b["conf"]) else (b, a)
    m = dict(win)
    # lose 쪽 방향 표시는 'x' 를 붙여 win 쪽과 구분 (후보 글자는 split_candidates 가 ':' 뒤 글자만 씀)
    m["all_dirs"] = win["all_dirs"] + " | " + " | ".join("x" + s for s in lose["all_dirs"].split(" | "))
    return m, win is a


def order_free(r):
    """read_order_cns 와 같은 정의: 완전일치이거나, 상자 2~4개의 어떤 순서가 정답과 같음."""
    if r["correct"] == "1":
        return True
    p = [b[4] for b in json.loads(r["best_boxes"] or "[]") if b[4]]
    return 2 <= len(p) <= 4 and any("".join(x) == r["truth"] for x in itertools.permutations(p))


def main():
    os.makedirs(OUT / "in", exist_ok=True)
    assert not (OUT / "summary.csv").exists(), "이미 있음, 덮어쓰지 않음"
    preds = {k: load(OCR / s / READ) for k, s in SPLITS.items()}
    probs = {k: {r["file_name"]: r for r in load(ATTR / f"probs_{s}.csv")} for k, s in SPLITS.items()}
    names = {r["file_name"] for r in preds["crop1"]}
    assert len(names) == 51
    # 전체 사진: 저장된 결과에서 이 51장만 꺼냄 (237장은 읽지 않고 버림)
    preds["full"] = [r for r in load(OCR / "exp300test" / READ) if r["file_name"] in names]
    probs["full"] = {r["file_name"]: r for r in load(ATTR / "probs_exp300test.csv") if r["file_name"] in names}
    # 합치기
    c1, c2 = ({r["file_name"]: r for r in preds[k]} for k in ("crop1", "crop2"))
    preds["union"], probs["union"] = [], {}
    n_c1 = 0
    for fn in sorted(names):
        m, from_c1 = merge(c1[fn], c2[fn])
        preds["union"].append(m)
        probs["union"][fn] = probs["crop1" if from_c1 else "crop2"][fn]
        n_c1 += from_c1
    print(f"union: 답을 crop1 에서 고른 사진 {n_c1}장 / crop2 에서 {51 - n_c1}장")

    # match2 를 그대로 쓰도록 방식마다 입력 파일을 만들어 넣음
    m1.ATTR = OUT / "in"
    db = m1.load_db()
    white = {d["drug_N"] for d in db if d["cols"] == {"하양"}}
    table, per = [], {}
    for k in ("full", "crop1", "crop2", "union"):
        part = f"m_{k}"
        write(OUT / "in" / f"pred_{k}.csv", preds[k])
        write(OUT / "in" / f"probs_{part}.csv", list(probs[k].values()))
        m1.PRED[part] = OUT / "in" / f"pred_{k}.csv"
        out = m2.predict_all(part, db, [("main", 0.95, False, 30)])
        per[k] = {r["file_name"]: (r, x[KEY][0], x[KEY][1]) for r, x in out}
        pct = lambda v: round(100 * sum(v) / max(len(v), 1), 1)
        hit = lambda r, x: r["drug_N"] == x[KEY][0]
        g = lambda f: pct([hit(r, x) for r, x in out if f(r)])
        table.append(dict(
            method=k, n=len(out), correct=sum(hit(r, x) for r, x in out), acc=pct([hit(r, x) for r, x in out]),
            none=sum(x[KEY][0] == "" for _, x in out), top3=pct([r["drug_N"] in [d for d, _ in x[KEY][1]] for r, x in out]),
            read_exact=pct([r["correct"] == "1" for r, _ in out]), read_order_free=pct([order_free(r) for r, _ in out]),
            aligned=sum(r["aligned"] == "1" for r, _ in out), long=g(lambda r: r["aligned"] == "1"), other=g(lambda r: r["aligned"] != "1"),
            cam60=g(lambda r: r["camera_la"] == "60"), cam90=g(lambda r: r["camera_la"] == "90"),
            white=g(lambda r: r["drug_N"] in white), n_white=sum(r["drug_N"] in white for r, _ in out)))
    base = per["full"]
    for t in table:
        cur = per[t["method"]]
        t["better"] = sum(cur[f][1] == cur[f][0]["drug_N"] and base[f][1] != base[f][0]["drug_N"] for f in names)
        t["worse"] = sum(cur[f][1] != cur[f][0]["drug_N"] and base[f][1] == base[f][0]["drug_N"] for f in names)
    write(OUT / "summary.csv", table)
    rows = []
    for fn in sorted(names):
        r = per["full"][fn][0]
        row = dict(file_name=fn, drug_N=r["drug_N"], truth=r["truth"], camera=r["camera_la"])
        for k in per:
            row[f"{k}_read"], row[f"{k}_pred"] = per[k][fn][0]["read"], per[k][fn][1]
            row[f"{k}_ok"] = int(per[k][fn][1] == r["drug_N"])
        rows.append(row)
    write(OUT / "pred_all.csv", rows)
    for t in table:
        print(f"  {t['method']:6s} 약 {t['acc']:5.1f}% ({t['correct']})  판정없음 {t['none']:2d}  3등 안 {t['top3']:5.1f}  읽기 {t['read_exact']:5.1f} / 순서 무시 {t['read_order_free']:5.1f}  "
              f"긴 {t['long']:5.1f} ({t['aligned']}) 그 외 {t['other']:5.1f}  60° {t['cam60']:5.1f} 90° {t['cam90']:5.1f}  흰 {t['white']:5.1f} ({t['n_white']})  full 대비 +{t['better']}/−{t['worse']}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
