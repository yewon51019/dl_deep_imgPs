# =====================================================================
#  match6c_cns.py  —  앞뒤 사진 2장 판별을 pft1 읽기로 다시 재기 (match6 v6b 와 같은 코드·규칙, OCR 안 돌림)
# =====================================================================
#
#  [바뀐 것]  읽기 = pft1 (orig_alignlong_line_pft1), λ = 0.9 (검증에서 고른 값). 동점 규칙 0.001, w 0.02, 후보 길이 ≤ 30 은 v6b 그대로
#  [확인]    한 장씩 낸 답이 저장된 pft1 답(match_v3\pft1)과 사진마다 같은지 확인
#  [보고]    검증 4종 48쌍, 개발 3종 36쌍, 개발 K-000147 제외 2종 24쌍. 평가용 사진·결과는 읽지 않음
#  [한계]    약이 적고, 동점 규칙은 v6 결과를 보고 만든 값이라 낙관적일 수 있음. 한 면 약은 재지 못함
#  [저장]    exp_cns\results\match_v6c\  pairs.csv, pred_pairs.csv, summary.csv  (이미 있으면 덮어쓰지 않고 멈춤)
#  [실행]    python exp_cns\match6c_cns.py      (저장소 맨 위 D:\dl_deep_imgps 에서)
# =====================================================================

import csv

import compare_pft1_cns as cp
import match6_cns as m6
import match_cns as m1
import match7_cns as m7

OUT = m1.cc.RESULTS / "match_v6c"


def main():
    if (OUT / "summary.csv").exists():
        raise SystemExit(f"이미 있음: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    m6.LAM, m6.TIE_BONUS = cp.CFG["pft1"][0], 0.001
    m1.PRED = {part: cp.CFG["pft1"][1][part][0] for part in ("val", "dev")}
    db = m1.load_db()
    imprints = sorted({x for d in db for x in d["imprints"]})
    rows = []
    for part in ("val", "dev"):
        cur = m6.run_part(part, db, imprints)
        _, saved_path, col = cp.CFG["pft1"][1][part]
        saved = {r["file_name"]: r[col] for r in csv.DictReader(open(saved_path, encoding="utf-8-sig"))}
        for r in cur:
            assert r["single_front_pred"] == saved[r["front_file"]], r["front_file"]
            assert r["single_back_pred"] == saved[r["back_file"]], r["back_file"]
        print(f"{part} 한 장씩 답 재현 확인 ({len(cur)}쌍)", flush=True)
        rows += cur
    m7.write_csv(OUT / "pred_pairs.csv", rows)
    m7.write_csv(OUT / "pairs.csv", [{k: r[k] for k in ("part", "drug_N", "camera", "front_file", "back_file")} for r in rows])
    table = []
    for part in ("val", "dev"):
        table.append(m6.summarize(rows, part))
        for d in sorted({r["drug_N"] for r in rows if r["part"] == part}):
            table.append(m6.summarize(rows, part, d))
    # 개발 K-000147 제외 (사진·라벨 불일치, 사용자 확인)
    rest = [r for r in rows if not (r["part"] == "dev" and r["drug_N"] == "K-000147")]
    extra = m6.summarize(rest, "dev")
    extra["drug_N"] = "K-000147 제외"
    table.append(extra)
    m7.write_csv(OUT / "summary.csv", table)
    for t in table:
        print(f"  {t['part']:3s} {t['drug_N']:14s} 짝 {t['pairs']:3d}  2장 {t['pair_acc']:5.1f}% ({t['pair_correct']}) 판정없음 {t['pair_none']:2d} 3등 안 {t['pair_top3']:5.1f}  "
              f"| 1장씩 {t['single_acc']:5.1f}% ({t['single_correct']}/{t['single_photos']}) 판정없음 {t['single_none']:2d}  | 한 면 약 오답 {t['onesided_wrong']}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
