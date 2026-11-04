# =====================================================================
#  error_test_final_cns.py  —  평가용 288장 주 방식(v2) 오답 원인 분류 (사후 분석, 방식은 안 바꿈)
# =====================================================================
#
#  [대상]  test_final_exp300\pred_exp300test.csv 에서 주 방식이 틀린 사진 (판정없음 + 틀린 약 답함)
#          저장된 결과만 읽음 — OCR·약 판별을 다시 돌리지 않음
#  [분류]  앞 칸부터 먼저 해당하는 것:
#     1    라벨 문제       : 사진에 라벨에 없는 면 (K-024770 콜민콜드캡슐, 라벨 CMCYJ 인데 한글 면)
#     2-1  같은 각인·겉모습 약 있음, 각인은 정확히 읽음
#     2-2  같은 각인·겉모습 약 있음, 각인도 틀리게 읽음
#     3    박스 못 찾음    : 고른 방향의 검출 박스 0개 (fallback)
#     4    잘못 읽음       : 박스는 찾았는데 다른 글자
#     5    기타            : 위 어디에도 안 듦 (읽기는 맞았는데 약 판별에서 틀림)
#     - '같은 각인·겉모습 약' = DB 의 다른 약 중 사진에 보이는 각인을 (앞·뒷면 어디든) 갖고 모양·제형·색 조합이 같은 약
#     - '정확히 읽음' = 읽은 글자가 사진에 보이는 각인과 완전히 같음
#     - 사진에 보이는 각인 = 라벨 정답. 단, 라벨 앞뒤가 뒤바뀐 K-003105 오르필서방정 300mg 은 같은 약의 다른 면
#       (24장 모두 라벨의 반대 면으로 읽힘 + 그림 2장 확인). 공식 점수의 정답은 그대로 두고 분류에만 씀
#  [저장]  test_final_exp300\errors_post_cause.csv (오답 사진 목록), errors_post_cause_summary.csv (전체·60°·90° 표)
#          같은 이름의 파일이 이미 있으면 덮어쓰지 않고 멈춤
#  [실행]  python exp_cns\error_test_final_cns.py   (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import collections
import csv
import os

import match_cns as m1

BASE = m1.cc.RESULTS / "test_final_exp300"
OCR = m1.oc.OCR_OUT / "exp300test" / "paddle_ft1_u2.0" / "orig_alignlong_line" / "pred.csv"
OUT_LIST = BASE / "errors_post_cause.csv"
OUT_SUMMARY = BASE / "errors_post_cause_summary.csv"
KEY = "main_wA_0.02"                                                 # 주 방식 (v2 base) 열 이름
LABEL_MISSING = {"K-024770"}                                         # 사진에 라벨에 없는 면
LABEL_FLIP = {"K-003105"}                                            # 라벨 앞뒤 뒤바뀜
CAUSES = ["1 라벨 문제", "2-1 같은 각인·겉모습 약, 각인 정확히 읽음", "2-2 같은 각인·겉모습 약, 각인도 틀리게 읽음",
          "3 박스 못 찾음", "4 잘못 읽음", "5 기타"]
SCOPES = {"전체": None, "60°": "60", "90°": "90"}


def look(d):
    """겉모습 = 모양·제형·색 조합 (색 순서는 무시)."""
    return d["shape"], d["form"], frozenset(d["cols"])


def visible_imprint(r, db):
    """사진에 보이는 각인: 라벨 앞뒤가 뒤바뀐 약이면 같은 약의 다른 면, 그 밖에는 라벨 정답."""
    if r["drug_N"] in LABEL_FLIP:
        other = [x for x in db[r["drug_N"]]["imprints"] if x != r["truth"]]
        assert len(other) == 1, r["file_name"]
        return other[0]
    return r["truth"]


def same_look_drugs(drug_N, imprint, db):
    """DB 의 다른 약 중 이 각인을 갖고 겉모습이 같은 약."""
    me = db[drug_N]
    return [d["drug_N"] for d in db.values() if d["drug_N"] != drug_N and imprint in d["imprints"] and look(d) == look(me)]


def classify(r, o, visible, same):
    """앞 칸부터 먼저 해당하는 원인 (r = 약 판별 결과, o = 읽기 결과)."""
    if r["drug_N"] in LABEL_MISSING:
        return CAUSES[0]
    if same:
        return CAUSES[1] if r["read"] == visible else CAUSES[2]
    if o["fallback"] == "1":
        return CAUSES[3]
    if r["read"] != visible:
        return CAUSES[4]
    return CAUSES[5]


def counts(xs, total_photos, total_errors):
    """표의 한 줄: 장수, 사진 대비 %, 오답 대비 %, 판정없음, 틀린 약 답함."""
    n = len(xs)
    return dict(photos=total_photos, n=n, pct_of_photos=round(100 * n / total_photos, 1),
                pct_of_errors=round(100 * n / total_errors, 1),
                none=sum(x["kind"] == "판정없음" for x in xs), wrong=sum(x["kind"] == "틀린 약 답함" for x in xs))


def main():
    for p in (OUT_LIST, OUT_SUMMARY):
        assert not os.path.exists(p), f"이미 있음, 덮어쓰지 않음: {p}"
    db = {d["drug_N"]: d for d in m1.load_db()}
    pred = list(csv.DictReader(open(BASE / "pred_exp300test.csv", encoding="utf-8-sig")))
    ocr = {r["file_name"]: r for r in csv.DictReader(open(OCR, encoding="utf-8-sig"))}
    # 오답 사진마다 원인 정하기
    out = []
    for r in pred:
        if r[f"{KEY}_ok"] == "1":
            continue
        o = ocr[r["file_name"]]
        visible = visible_imprint(r, db)
        same = same_look_drugs(r["drug_N"], visible, db)
        p = r[f"{KEY}_pred"]
        out.append(dict(file_name=r["file_name"], drug_N=r["drug_N"], name=db[r["drug_N"]]["name"], truth=r["truth"], visible=visible,
                        read=r["read"], cause=classify(r, o, visible, same), kind="판정없음" if p == "" else "틀린 약 답함",
                        pred=p, pred_name=db[p]["name"] if p else "", camera=o["camera_la"], n_boxes=o["n_boxes"],
                        same_look="|".join(same)))
    out.sort(key=lambda x: (CAUSES.index(x["cause"]), x["file_name"]))
    # 기존 결과와 맞는지 확인 (summary.csv 주 방식: 288장, 맞음 175, 판정없음 34)
    assert len(pred) == 288 and len(out) == 113 and sum(x["kind"] == "판정없음" for x in out) == 34, (len(pred), len(out))
    with open(OUT_LIST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    # 표: 전체·60°·90° 마다 원인별 줄 + 2번 소계 + 합계
    n_cam = collections.Counter(ocr[r["file_name"]]["camera_la"] for r in pred)
    rows = []
    for scope, cam in SCOPES.items():
        xs = [x for x in out if cam is None or x["camera"] == cam]
        photos = len(pred) if cam is None else n_cam[cam]
        for c in CAUSES:
            sub = [x for x in xs if x["cause"] == c]
            if c == CAUSES[1]:                                       # 2-1 앞에 2번 소계
                two = [x for x in xs if x["cause"] in CAUSES[1:3]]
                rows.append(dict(scope=scope, cause="2 같은 각인·겉모습 약 있음 (소계)", **counts(two, photos, len(xs)), drugs=""))
            rows.append(dict(scope=scope, cause=c, **counts(sub, photos, len(xs)),
                             drugs="|".join(sorted({f"{x['name'][:14]}({x['truth']})" for x in sub}))))
        rows.append(dict(scope=scope, cause="합계", **counts(xs, photos, len(xs)), drugs=""))
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['scope']:3s} {r['cause']:32s} {r['n']:3d}장  사진 대비 {r['pct_of_photos']:5.1f}%  오답 대비 {r['pct_of_errors']:5.1f}%  "
              f"판정없음 {r['none']:3d}  틀린 약 {r['wrong']:3d}")
    print(f"저장: {OUT_LIST}\n      {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
