# =====================================================================
#  run_exp300_paddle_b.py  —  PaddleOCR 'B 방식' 점수 (팀원 val_paddle.py 와 같은 방식)
# =====================================================================
#
#  [왜 따로 있나요?]
#   run_exp300_paddle.py 는 A실험과 같은 파이프라인(알약 자르기 → 전처리 → 12방향 읽기 → DB 검색)이라
#   PARSeq 등 A 모델과 같은 표에서 비교됩니다. (메인 결과)
#   이 파일은 팀원 B실험 방식으로 같은 사진을 한 번 더 재서, 팀원의 기존 결과와 이어서 보기 위한 보조 점수예요.
#
#  [B 방식 = 팀원 val_paddle.py 와 같음]
#   · 원본 사진 전체를 PaddleOCR 에 그대로 넣음 (자르기·전처리·회전 투표 없음)
#   · 검출 기준 3개를 0.0 으로 (흐린 글자도 최대한 박스로 잡음):
#       text_det_thresh=0.0, text_det_box_thresh=0.0, text_rec_score_thresh=0.0
#   · 읽은 글자 조각들 중 '하나라도' 정답(앞면 또는 뒷면 각인)과 같으면 성공
#
#  [점수 2가지]  ← 같은 사진·같은 결과에서 두 방식으로 셈
#   hit_B    : 팀원 코드 그대로 (라벨 문자열 그대로 비교, 앞뒷면 둘 중 하나면 성공)
#   hit_norm : 대문자·숫자만 남기고 비교, 빈 글자/'마크'는 정답에서 제외 (우리 A실험 규칙과 같은 정규화)
#   ※ 팀원 코드는 라벨이 빈 문자열이면 OCR 의 빈 결과('')와 '일치'로 셀 수 있어요. hit_norm 은 이걸 막습니다.
#   ※ 이 점수는 '약 이름 정확도'가 아니라 '글자 조각 중 정답이 있었나'예요. A실험 표의 정확도/F1 과 직접 비교하지 마세요.
#
#  [실행 예]  (A실험과 같은 manifest.csv 를 써서 같은 평가용 237장을 씁니다)
#     python run_exp300_paddle_b.py --manifest manifest.csv --split test --out pred_test_paddle_B.csv
#  끊기면 같은 명령을 다시 실행하세요 (이어하기).
# =====================================================================

import argparse
import csv
import os
import time
from collections import defaultdict

import run_exp300 as base
from run_exp300 import norm

COLS = ["file_name", "split", "truth_drug", "print_front", "print_back", "texts", "hit_B", "hit_norm",
        "n_boxes", "ms", "form", "drug_dir", "camera_la", "rot_deg"]


def make_ocr():
    """팀원 코드와 같은 설정(v3, 검출 기준 0.0). Colab CPU 오류 방지로 oneDNN 은 끔 (결과에는 영향 없음)"""
    from run_exp300_paddle import build_paddle
    return build_paddle(
        dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
             text_det_thresh=0.0, text_det_box_thresh=0.0, text_rec_score_thresh=0.0),
        dict(use_angle_cls=False, lang="en", det_db_thresh=0.0, det_db_box_thresh=0.0, drop_score=0.0, show_log=False))


def read_texts(ocr, path):
    """원본 사진 한 장 → 읽은 글자 조각 목록 (v3 .predict, v2 .ocr 둘 다 지원)"""
    if hasattr(ocr, "predict"):
        texts = []
        for res in ocr.predict(str(path)):
            texts.extend(res["rec_texts"])
        return texts
    lines = ocr.ocr(str(path), cls=False)
    return [t for _, (t, _) in (lines[0] if lines and lines[0] else [])]


def hits(texts, front, back):
    """(hit_B, hit_norm).  hit_B 는 팀원 코드와 똑같이: any(t in [front, back])"""
    prints = [front, back]
    hit_b = any(t in prints for t in texts)
    gold = {n for n in (norm(front), norm(back)) if n}          # 빈 값·'마크'는 제외
    hit_n = any(norm(t) in gold for t in texts if norm(t))
    return hit_b, hit_n


def summarize(rows):
    def rate(rs, key):
        return sum(r[key] == "1" for r in rs) / len(rs) * 100 if rs else 0.0
    print(f"\n[B 방식 점수]  n={len(rows)}")
    print(f"  hit_B    (팀원 코드 그대로)         : {rate(rows, 'hit_B'):5.1f}%")
    print(f"  hit_norm (정규화·빈값/마크 제외)    : {rate(rows, 'hit_norm'):5.1f}%")
    for field, name in (("drug_dir", "앞/뒷면"), ("camera_la", "카메라각도"), ("form", "제형")):
        g = defaultdict(list)
        for r in rows:
            g[r[field]].append(r)
        print(f"  {name}별 hit_norm: " + ", ".join(f"{k or '(빈값)'}={rate(v, 'hit_norm'):.0f}%({len(v)})"
                                                    for k, v in sorted(g.items())))
    empty_hit = sum(1 for r in rows if r["hit_B"] == "1" and r["hit_norm"] == "0")
    print(f"  ※ hit_B 만 성공(빈 글자·정규화 차이 등)인 사진: {empty_hit}장")


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR B 방식(원본 사진, 하나라도 맞으면 성공) 점수")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--out", default="pred_paddle_B.csv")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞에서 N장만 (0=전부)")
    a = ap.parse_args()

    items = base.load_manifest(a.manifest, a.split)         # A실험과 같은 채점 대상 ('마크'뿐인 사진은 제외)
    if a.limit:
        items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8-sig") as f:
            done = {r["file_name"] for r in csv.DictReader(f)}
        print(f"이어하기: 이미 {len(done)}장 완료")
    todo = [it for it in items if it["file_name"] not in done]
    print(f"사진 {len(items)}장 ({a.split}), 이번에 할 일 {len(todo)}장")

    new_file = not os.path.exists(a.out)
    fout = open(a.out, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        w.writeheader()
    if todo:
        print("\n[paddleocr] 불러오는 중... (B 설정: 검출 기준 0.0)")
        ocr = make_ocr()
        for k, it in enumerate(todo, 1):
            t0 = time.time()
            texts = read_texts(ocr, it["image_path"])
            hb, hn = hits(texts, it.get("print_front", ""), it.get("print_back", ""))
            w.writerow({"file_name": it["file_name"], "split": it["split"], "truth_drug": it["drug_N"],
                        "print_front": it.get("print_front", ""), "print_back": it.get("print_back", ""),
                        "texts": "|".join(texts), "hit_B": int(hb), "hit_norm": int(hn), "n_boxes": len(texts),
                        "ms": int((time.time() - t0) * 1000), "form": it.get("form", ""),
                        "drug_dir": it.get("drug_dir", ""), "camera_la": it.get("camera_la", ""),
                        "rot_deg": it.get("rot_deg", "")})
            fout.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  (최근: {it['file_name'][:24]}… 읽은 조각 {len(texts)}개)")
    fout.close()
    with open(a.out, encoding="utf-8-sig") as f:
        summarize(list(csv.DictReader(f)))
    print(f"\n저장 완료: {a.out}")


if __name__ == "__main__":
    main()
