# =====================================================================
#  curved_exp300.py  —  휜 각인(알약 가장자리를 따라 새긴 글자) 대응: 극좌표 띠 후보 + 돌림판 비교, 개발용 51장만
# =====================================================================
#
#  [기준]  B = PARSeq 최종 방식(보정 V3 + 색 동점 깨기). 12방향 후보는 pred_dev_parseq_P0_align_vote_cv410.csv 의 저장값 그대로
#  [추가 후보]  둥근 알약만, 극좌표로 펼친 띠를 정방향·180° 뒤집어 PARSeq 로 읽은 글자 2개를 V3 후보에 더함
#     P1 : 기존 극좌표 띠 (align_exp300.polar_strip, 잘라낸 사진을 정사각형으로 늘려 펼침)
#     P2 : 개선한 띠 — 알약 테두리에 타원을 맞춰 원으로 편 뒤 펼침(카메라 60° 물결 줄이기)
#          + 띠를 1.5바퀴로 이어 붙여 이음매가 글자를 자르지 않게
#  [돌림판 비교 (…C)]  원형 각인은 어디서부터 읽어도 같은 글자 → 띠 후보를 DB 각인의 '글자 순서를 돌린 모든 형태'와 비교
#     ('ICEPTARI' ↔ 'ICEPTAR'·'ARICEPT'…), 돌린 형태가 띠 글자 안에 통째로 들어 있으면 만점. 4글자 이상 각인에만 (짧은 각인 우연 일치 방지)
#     12방향 후보는 지금처럼 일반 비교 (띠 후보에만 돌림판 비교)
#  [고르는 기준, 미리 정함]  개발용 정확도 최고 → 같으면 판정 없음이 적은 쪽 → 더 단순한 쪽. 1장 = 2%p.
#  [저장]  --out: strips_dev.csv (띠에서 읽은 글자), metrics.csv, updown.csv / --viz-out: 띠 그림 (원본 사진 포함 → GitHub 금지)
#  [실행 예]  python exp300\curved_exp300.py --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#             --parseq exp300\results\test_final\final_dev.csv --parseq-vote exp300\results\pred_dev_parseq_P0_align_vote_cv410.csv ^
#             --out exp300\results\curved_cv410 --viz-out exp300\results\viz_curved_dev
# =====================================================================

import argparse
import csv
import json
import os

import torch                                         # Paddle 보다 먼저

import cv2
import numpy as np

import align_exp300 as al
import combined_exp300 as cb
import correct_exp300 as corr
import final_exp300 as fin
import run_exp300 as base
from evaluate_exp300 import drug_metrics
from viz_boxes_exp300 import judge

VARIANTS = ["B", "P1", "P1C", "P2", "P2C"]
NONE_LABEL = "(판정없음)"
METRICS = ["정답률", "정확도", "정밀도", "민감도(재현율)", "F1"]


def pill_ellipse(g):
    """잘라낸 사진(배경은 알약 중간 밝기로 칠해져 있음)에서 알약 윤곽에 타원을 맞춤 → cv2.fitEllipse 결과 또는 None"""
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    fill = int(np.bincount(border).argmax())
    m = (np.abs(g.astype(np.int16) - fill) > 2).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 0.2 * g.size:
        return None, None
    return cv2.fitEllipse(c), c


def ellipse_to_circle(g):
    """타원으로 보이는 알약을 원으로 펴서, 가운데에 알약이 있는 정사각형(반지름 R, 크기 2R×1.1)으로 → polar_strip 에 그대로 넣을 수 있게.
       회전 방향 부호가 헷갈리지 않게 두 부호를 다 해 보고 다시 맞춘 타원이 더 둥근 쪽을 씀"""
    ell, cnt = pill_ellipse(g)
    if ell is None:
        return g
    (cx, cy), (w, h), ang = ell
    R = max(w, h) / 2
    S = int(round(2 * R * 1.1))
    best, best_ratio = g, 9.0
    for sgn in (1, -1):
        Rm = cv2.getRotationMatrix2D((cx, cy), sgn * ang, 1.0)            # 타원 축을 가로·세로로
        A = np.vstack([Rm, [0, 0, 1]])
        Sc = np.array([[2 * R / w, 0, 0], [0, 2 * R / h, 0], [0, 0, 1]])
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        T2 = np.array([[1, 0, S / 2], [0, 1, S / 2], [0, 0, 1]])
        M = (T2 @ Sc @ T1 @ A)[:2]
        out = cv2.warpAffine(g, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        pts = cv2.transform(cnt.astype(np.float32), M)
        (_, _), (w2, h2), _ = cv2.fitEllipse(pts)
        ratio = max(w2, h2) / max(1e-6, min(w2, h2))
        if ratio < best_ratio:
            best, best_ratio = out, ratio
    return best


def strip_reads(g0, ocr, variant):
    """둥근 알약의 극좌표 띠 → [(글자, 신뢰도, 이름)] 정방향·180° 뒤집힘, 그리고 그림용 띠"""
    if variant == "P1":
        S, seam = al.polar_strip(g0)
    else:
        S, seam = al.polar_strip(ellipse_to_circle(g0))
        S = np.ascontiguousarray(np.hstack([S, S[:, : S.shape[1] // 2]]))   # 1.5바퀴: 이음매에 걸린 글자도 한 번은 통째로
    out = []
    for flip in (0, 1):
        img = S if not flip else np.ascontiguousarray(S[::-1, ::-1])
        t, c = ocr(img)
        out.append((base.norm(t), float(c), f"{variant}{'r' if flip else ''}"))
    return out, S


def cyc_scores(t, cdb):
    """돌림판 비교: DB 각인(4글자 이상)의 글자 순서를 돌린 모든 형태와 비교 → {약: 점수}. 일반 V3 규칙(부분일치·헷갈리는 글자) 포함"""
    out = {}
    if not t:
        return out
    for d in cdb:
        best = 0.0
        for imp in d["imprints"]:
            rots = [imp[i:] + imp[:i] for i in range(len(imp))] if len(imp) >= 4 else [imp]
            if len(imp) >= 4 and any(r in t for r in rots):
                s = 1.0                                     # 돌린 형태가 띠 글자 안에 통째로
            else:
                s = max(corr.sim_v(t, r, True, True, 1.0) for r in rots)
            best = max(best, s)
        if best >= corr.MIN_SIM:
            out[d["drug_N"]] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--parseq", required=True)
    ap.add_argument("--parseq-vote", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz-out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.viz_out, exist_ok=True)

    items = base.load_manifest(a.manifest, "dev")                          # 개발용만
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    cdb = corr.load_db(a.db)
    ab = {it["file_name"]: cb.pill_ab(base.imread_any(it["image_path"])) for it in items}
    protos = {it["file_name"]: cb.color_protos(items, ab, dbrow, leave_out=it["file_name"]) for it in items}
    ref = {r["file_name"]: r for r in csv.DictReader(open(a.parseq, encoding="utf-8-sig")) if r["method"] == "V3+색"}
    c12 = {r["file_name"]: json.loads(r["candidates"]) for r in csv.DictReader(open(a.parseq_vote, encoding="utf-8-sig"))}
    ocr = base.PARSeq()

    strips, srows = {}, []
    for k, it in enumerate(items, 1):
        fn = it["file_name"]
        gray, aspect, color, ok = base.crop_pill(base.imread_any(it["image_path"]))
        g0 = base.PREPS["P0"][1](gray)
        strips[fn] = {"aspect": aspect}
        if aspect > 1.3:
            continue
        imgs = []
        for v in ("P1", "P2"):
            reads, S = strip_reads(g0, ocr, v)
            strips[fn][v] = reads
            imgs.append(cv2.resize(S, (900, int(S.shape[0] * 900 / S.shape[1]))))
            srows.append({"file_name": fn, "variant": v, "truth": base.norm(it["truth"]),
                          "read_forward": reads[0][0], "conf_forward": round(reads[0][1], 1),
                          "read_flipped": reads[1][0], "conf_flipped": round(reads[1][1], 1)})
        cv2.imencode(".png", np.vstack(imgs))[1].tofile(os.path.join(a.viz_out, f"{k:02d}_{it['drug_N']}_strips_P1top_P2bottom.png"))
        print(f"  {k:2d} {base.norm(it['truth']):<9} P1 {[r[0] for r in strips[fn]['P1']]}  P2 {[r[0] for r in strips[fn]['P2']]}")
    with open(os.path.join(a.out, "strips_dev.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    res = []
    for v in VARIANTS:
        for it in items:
            fn = it["file_name"]
            aspect = strips[fn]["aspect"]
            texts = [t for t in c12[fn] if 1 <= len(t) <= 12]
            v3map = cb.edit_scores(texts, cdb)
            extra = strips[fn].get(v[:2], []) if v != "B" else []
            for t, c, name in extra:
                sm = cyc_scores(t, cdb) if v.endswith("C") else corr.drug_scores(t, cdb, cb.V3_OPT)
                for d, s in sm.items():
                    v3map[d] = max(v3map.get(d, 0.0), s)
            d, why = fin.v3_color(v3map, ab[fn], aspect, dbrow, protos[fn])
            truth = base.norm(it["truth"])
            cyc_full = any(len(truth) >= 4 and any((truth[i:] + truth[:i]) in t for i in range(len(truth))) for t, _, _ in extra)
            res.append({"variant": v, "file_name": fn, "shape": ref[fn]["shape"], "truth_drug": it["drug_N"], "truth": truth,
                        "pred_drug": d or NONE_LABEL, "drug_ok": int(d == it["drug_N"]), "decided_by": why,
                        "strip_full_read": int(cyc_full), "strip_reads": "|".join(t for t, _, _ in extra)})
    chk = [r for r in res if r["variant"] == "B"]
    print(f"\n[확인] B 가 PARSeq 최종 방식과 다른 사진: {sum(r['pred_drug'] != ref[r['file_name']]['pred_drug'] for r in chk)}/{len(chk)}장")

    met, ud = [], []
    ari = [it["file_name"] for it in items if base.norm(it["truth"]) == "ARICEPT"]
    print(f"\n[개발용 51장]  1장 = 2.0%p, 둥근 23장 1장 = 4.3%p, ARICEPT 사진 {len(ari)}장")
    print(f"  {'방식':<6} {'정확도':>7} {'정밀도':>7} {'민감도':>7} {'F1':>7}  판정없음  둥근   ARICEPT맞힘  띠에서 ARICEPT 통째로 읽음  좋아짐/나빠짐")
    for v in VARIANTS:
        sub = [r for r in res if r["variant"] == v]
        acc, p, rc, f1 = drug_metrics([x["truth_drug"] for x in sub], [x["pred_drug"] for x in sub])
        none = sum(x["pred_drug"] == NONE_LABEL for x in sub)
        rnd = np.mean([x["drug_ok"] for x in sub if x["shape"] == "둥근"])
        a_ok = sum(x["drug_ok"] for x in sub if x["file_name"] in ari)
        a_full = sum(x["strip_full_read"] for x in sub if x["file_name"] in ari)
        up = [x["file_name"] for x in sub if x["drug_ok"] == 1 and ref[x["file_name"]]["drug_ok"] == "0"]
        dn = [x["file_name"] for x in sub if x["drug_ok"] == 0 and ref[x["file_name"]]["drug_ok"] == "1"]
        print(f"  {v:<6} {acc * 100:6.1f}% {p * 100:6.1f}% {rc * 100:6.1f}% {f1 * 100:6.1f}%  {none:6d}  {rnd * 100:5.1f}  {a_ok:4d}/{len(ari)}"
              f"        {a_full:4d}/{len(ari)}                  +{len(up)} / -{len(dn)}")
        met.append({"variant": v, "정확도": round(acc * 100, 1), "정밀도": round(p * 100, 1), "민감도(재현율)": round(rc * 100, 1),
                    "F1": round(f1 * 100, 1), "판정없음": none, "둥근_정확도": round(rnd * 100, 1),
                    "ARICEPT_맞힘": a_ok, "ARICEPT_띠통째": a_full, "좋아짐": len(up), "나빠짐": len(dn)})
        ud += [{"variant": v, "change": "좋아짐", "file_name": f} for f in up] + [{"variant": v, "change": "나빠짐", "file_name": f} for f in dn]
    for name, rows in (("results_dev.csv", res), ("metrics.csv", met), ("updown.csv", ud)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["variant"])
            w.writeheader()
            w.writerows(rows)
    for x in ud:
        print(f"  {x['variant']}: {x['change']} {x['file_name']}")
    print(f"\n저장 완료: {a.out}, 띠 그림 {a.viz_out}")


if __name__ == "__main__":
    main()
