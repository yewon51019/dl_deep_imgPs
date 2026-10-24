# =====================================================================
#  final_exp300.py  —  평가용 최종 실행: 보정 V3 + 색 동점 깨기 (주 방식) / C4 / 길 A / 12방향 다수결
# =====================================================================
#
#  [사전 기록]  CLAUDE.md 6·7장 '평가용 최종 방식' (2026-10-24, 평가용 실행 전에 적고 커밋함)
#  [방식]
#    vote   : 12방향 다수결 (지금 방식)                     A    : ② 확신도 방향 + 박스 순서 바꾸기 (길 A)
#    V3+색  : 보정 V3 → 1등 동점이면 색 → 크기 → 판정 없음  C4   : V3+색 에서 못 정하면 dbscore(concat_mean)
#  [색 대표값]
#    --split dev  : 사진마다 자기 사진을 뺀 나머지 개발용으로 대표값 (leave-one-out, combined_exp300 과 같음)
#                   + 개발용 51장 전체로 만든 대표값을 --protos 파일로 저장 (평가용에 쓸 고정값)
#    --split test : --protos 파일을 읽기만 함. 평가용 사진의 색은 대표값 계산에 쓰지 않음
#  [평가용은 한 번만]  --split test 결과 파일(final_test.csv)이 이미 있으면 실행하지 않고 멈춤
#  [저장]
#    --out          : final_{split}.csv (사진별·방식별), color_protos_dev.csv (dev 실행 때)
#    --aligned-dir  : 방식별 하위 폴더에 '정렬해서 세운 알약'만  (파일 이름: 번호_약코드_rot회전각_cam_ang고른각도_각인판정_약판정)
#    --photo-dir    : 사진별 비교 그림 [잘라낸 알약 | 다수결 | 길 A | V3+색 | C4] + 요약  ← 원본 사진 포함, GitHub 금지
#  [실행 예]
#    python exp300\final_exp300.py --split dev  --manifest exp300\results\manifest.csv --db exp300\results\search_db_v3.csv ^
#           --protos exp300\results\test_final\color_protos_dev.csv --out exp300\results\test_final ^
#           --aligned-dir exp300\results\viz_aligned_only\dev_final --photo-dir exp300\results\viz_final_report\photos_dev
#    python exp300\final_exp300.py --split test ... (같은 인자, 폴더 이름만 test)
# =====================================================================

import argparse
import csv
import os
import sys
import time

import torch

import cv2
import numpy as np

import combined_exp300 as cb
import correct_exp300 as corr
import dbscore_exp300 as dbs
import run_exp300 as base
from viz_align_exp300 import Recorder, cell, fit, perm_best
from viz_boxes_exp300 import JCOL, ascii_only, judge

METHODS = ["vote", "A", "V3+색", "C4"]
TITLES = {"vote": "12-dir vote (now)", "A": "path A: (2)+box perm", "V3+색": "V3 + color tie (MAIN)", "C4": "C4: V3->color->dbscore"}
COLORS = {"vote": (60, 180, 60), "A": (150, 80, 200), "V3+색": (0, 80, 200), "C4": (0, 120, 120)}
NONE_LABEL = "(판정없음)"


def load_protos(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["color"]: np.array([float(r["a"]), float(r["b"])], np.float32) for r in csv.DictReader(f)}


def save_protos(path, protos, counts):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["color", "a", "b", "n_photos", "source"])
        for c, v in sorted(protos.items()):
            w.writerow([c, round(float(v[0]), 3), round(float(v[1]), 3), counts[c], "개발용 51장 전체 (평가용 사진 안 씀)"])


def best_candidate(c12, drug, cdb):
    """그 약을 고르게 한 후보(가장 높은 V3 점수) → (글자, 각도). 없으면 None"""
    best = None
    for t, c, a in c12:
        if 1 <= len(t) <= 12:
            s = corr.drug_scores(t, cdb, cb.V3_OPT).get(drug, 0.0)
            if s > 0 and (best is None or s > best[0]):
                best = (s, t, a)
    return (best[1], best[2]) if best else None


def v3_color(v3map, obs_ab, aspect, dbrow, protos):
    """주 방식: V3 → 1등 동점이면 색 → 크기 → 판정 없음. (약, 근거)"""
    e = corr.decide(v3map)
    if e:
        return e, "글자"
    tied = cb.top_set(v3map)
    if len(tied) >= 2:
        d, why = cb.tiebreak(tied, obs_ab, aspect, dbrow, protos)
        if d:
            return d, why
    return "", "판정없음"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--db", required=True, help="search_db_v3.csv")
    ap.add_argument("--protos", required=True, help="색 대표값 CSV (dev: 저장, test: 읽기만)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--photo-dir", required=True)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="시험용(dev 만)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"final_{a.split}.csv")
    if a.split == "test":
        if os.path.exists(out_csv):
            sys.exit(f"[멈춤] {out_csv} 가 이미 있어요. 평가용은 한 번만 돌립니다 (사전 기록).")
        if a.limit:
            sys.exit("[멈춤] 평가용은 일부만 돌리지 않아요.")
        if not os.path.exists(a.protos):
            sys.exit("[멈춤] 색 대표값 파일이 없어요. 먼저 --split dev 로 만드세요.")

    items = base.load_manifest(a.manifest, a.split)
    if a.limit:
        items = items[: a.limit]
    dbrow = {r["drug_N"]: r for r in csv.DictReader(open(a.db, encoding="utf-8-sig"))}
    db = base.load_db(a.db)
    cdb = corr.load_db(a.db)
    cands = dbs.db_candidates(a.db)
    rec = Recorder(base.PARSeq())
    scorer = dbs.DBScorer(rec.m, [c["text"] for c in cands])
    for m in METHODS:
        os.makedirs(os.path.join(a.aligned_dir, m.replace("+", "_plus_").replace("색", "color")), exist_ok=True)
    os.makedirs(a.photo_dir, exist_ok=True)

    # 색: 원본 사진에서 먼저 모두 계산 (dev 는 대표값 만들기, test 는 관측값만)
    ab = {}
    for it in items:
        ab[it["file_name"]] = cb.pill_ab(base.imread_any(it["image_path"]))
    if a.split == "dev":
        full = cb.color_protos(items, ab, dbrow)
        counts = {c: sum(1 for it in items if cb.single_color(dbrow[it["drug_N"]]) == c) for c in full}
        if not a.limit:
            save_protos(a.protos, full, counts)
            print(f"색 대표값 저장 (개발용 {len(items)}장 전체): {a.protos}  {sorted(full)}")
    else:
        fixed = load_protos(a.protos)
        print(f"색 대표값 읽음 (개발용으로 고정): {sorted(fixed)}  — 평가용 사진은 대표값에 안 씀")

    rows = []
    t_start = time.time()
    for k, it in enumerate(items, 1):
        fn = it["file_name"]
        gray, aspect, color, ok = base.crop_pill(base.imread_any(it["image_path"]))
        g0 = base.PREPS["P0"][1](gray)
        shape = base.shape_group(aspect)
        truth = base.norm(it["truth"])
        protos = cb.color_protos(items, ab, dbrow, leave_out=fn) if a.split == "dev" else fixed

        rec.log = []
        pv, cv_, c12, av = base.read_with_rotation(g0, aspect, rec, a.step)
        log = list(rec.log)
        angles = [x[2] for x in c12]
        # 다수결
        res = {"vote": {"drug": base.identify(pv, "", "", db)[1], "read": pv, "angle": av, "why": "다수결"}}
        # 길 A
        bc = base.pick_by_conf(c12)
        gA, boxesA, textsA = log[angles.index(bc[2])]
        tA, dA, _ = perm_best([(b, t) for b, t in zip(boxesA, textsA) if t], db)
        if not tA:
            tA, dA = bc[0], base.identify(bc[0], "", "", db)[1]
        nboxA = 0 if (len(boxesA) == 1 and tuple(boxesA[0]) == (0, 0, gA.shape[1], gA.shape[0])) else len(boxesA)
        res["A"] = {"drug": dA, "read": tA, "angle": bc[2], "why": "② 방향+순서"}
        # V3 + 색 (주 방식)
        v3map = cb.edit_scores([c[0] for c in c12 if 1 <= len(c[0]) <= 12], cdb)
        dV, why = v3_color(v3map, ab[fn], aspect, dbrow, protos)
        bcV = best_candidate(c12, dV, cdb) if dV else None
        res["V3+색"] = {"drug": dV, "read": bcV[0] if bcV else "", "angle": bcV[1] if bcV else bc[2], "why": why}
        # C4: 그래도 못 정하면 dbscore (같은 글자는 같은 점수로)
        sc, ang_db, nb_db = dbs.score_image(scorer, g0, aspect, a.step)["concat_mean"]
        tbest = {}
        for c, s in zip(cands, sc):
            tbest[c["text"]] = max(tbest.get(c["text"], -np.inf), float(s))
        dmap = {}
        for c in cands:
            dmap[c["drug_N"]] = max(dmap.get(c["drug_N"], -np.inf), tbest[c["text"]])
        if dV:
            res["C4"] = dict(res["V3+색"])
        else:
            pool = cb.top_set(v3map) or list(dmap)
            d4 = max(pool, key=lambda d: dmap[d])
            i4 = max((i for i, c in enumerate(cands) if c["drug_N"] == d4), key=lambda i: sc[i])
            res["C4"] = {"drug": d4, "read": cands[i4]["text"], "angle": int(ang_db[i4]), "why": "dbscore"}

        tag = "round" if shape == "둥근" else "long"
        for m in METHODS:
            r = res[m]
            r["judge"] = judge(r["read"], truth)
            r["ok"] = int(r["drug"] == it["drug_N"])
            # 정렬해서 세운 알약 (고른 각도로 돌린 P0 사진)
            r["img"] = base.rotate(g0, r["angle"])
            jname = {"O": "O", "~": "part", "X": "X", "-": "none"}[r["judge"]]     # 각인 판정 (파일 이름에 ~ 대신 part)
            name = (f"{k:03d}_{it['drug_N']}_rot{it.get('rot_deg', '')}_cam{it.get('camera_la', '')}_ang{r['angle']:g}_"
                    f"{jname}_{'OK' if r['ok'] else 'NG'}.png")
            sub = m.replace("+", "_plus_").replace("색", "color")
            cv2.imencode(".png", r["img"])[1].tofile(os.path.join(a.aligned_dir, sub, name))
            rows.append({"file_name": fn, "split": a.split, "shape": shape, "camera_la": it.get("camera_la", ""),
                         "rot_deg": it.get("rot_deg", ""), "form": it.get("form", ""), "method": m,
                         "truth_drug": it["drug_N"], "truth_imprint": truth, "pred_drug": r["drug"] or NONE_LABEL,
                         "drug_ok": r["ok"], "read": r["read"], "judge": r["judge"], "angle": r["angle"],
                         "decided_by": r["why"], "n_box_A": nboxA, "db_nbox": int(nb_db[int(np.argmax(sc))]),
                         "color_a": round(float(ab[fn][0]), 1), "color_b": round(float(ab[fn][1]), 1)})

        # 사진별 비교 그림
        cells = [cell(fit(cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)), "cropped pill (P0, 0deg)", (60, 60, 60),
                      [(f"truth: {ascii_only(truth)}  ({it['drug_N']})", (60, 60, 60)),
                       (f"color a,b = {ab[fn][0]:.0f},{ab[fn][1]:.0f}", (90, 90, 90))])]
        for m in METHODS:
            r = res[m]
            okc = (0, 150, 0) if r["ok"] else (0, 0, 200)
            cells.append(cell(fit(cv2.cvtColor(r["img"], cv2.COLOR_GRAY2BGR)), f"{TITLES[m]}  ang={r['angle']:g}", COLORS[m],
                              [(f"[{r['judge']}] read: {ascii_only(r['read']) or '(none)'}", JCOL[r["judge"]]),
                               (f"drug {ascii_only(r['drug']) or '(none)'} {'OK' if r['ok'] else 'NG'}", okc)]))
        grid = np.hstack(cells)
        bar = np.full((34, grid.shape[1], 3), 245, np.uint8)
        cv2.putText(bar, f"{k:03d} {a.split} {it['drug_N']} {tag} rot={it.get('rot_deg', '')} cam={it.get('camera_la', '')} "
                         f"truth={ascii_only(truth)}  V3+color by: {ascii_only(res['V3+색']['why'].replace('색', 'color').replace('글자', 'text').replace('크기', 'size').replace('판정없음', 'none'))}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imencode(".png", np.vstack([bar, grid]))[1].tofile(
            os.path.join(a.photo_dir, f"{k:03d}_{tag}_{it['drug_N']}_rot{it.get('rot_deg', '')}_cam{it.get('camera_la', '')}.png"))
        print(f"  {k:3d}/{len(items)} {tag:<5} {truth:<14} " +
              " ".join(f"{m}={res[m]['drug'] or '-'}({'OK' if res[m]['ok'] else 'NG'})" for m in METHODS))

    with open(out_csv if not a.limit else out_csv.replace(".csv", "_limit.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장 완료: {out_csv}  ({len(items)}장, {time.time() - t_start:.0f}s)  → 채점: final_report_exp300.py")


if __name__ == "__main__":
    main()
