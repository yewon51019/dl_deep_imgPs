# =====================================================================
#  paddle_boxes_cns.py  —  PaddleOCR(t0) 글자 상자만 미리 계산해 저장 (Paddle 전용 GPU 환경에서 실행)
# =====================================================================
#
#  [왜]  CPU 판 Paddle 은 12방향 검출에 사진 한 장당 약 20초 (개발용 20장 시험 운전) → 468장 × 전처리 3개면 약 8시간.
#        GPU 판(paddlepaddle-gpu 3.3.1, CUDA 11.8 빌드, GTX 1070 Ti 에서 동작 확인)을 본 환경(.venv)에 깔면
#        OpenCV·numpy·CUDA 라이브러리가 바뀌어 기존 결과가 달라질 수 있어서, 전용 환경 .venv_paddle_gpu 에서 상자만 계산한다.
#  [하는 일]  ocr_cns.py 와 같은 방법(ocr_common_cns)으로 사진을 고르고 전처리·12방향 돌리기 → 방향마다 Paddle t0 상자
#  [저장]  exp_cns\results\ocr_v1\<split>\paddle_t0\<prep>\boxes.json
#          { 파일명: { "각도": {"boxes": [[x0, y0, x1, y1], ...], "det_ms": 검출 시간} } , ... }
#          ocr_cns.py --det paddle_t0 는 이 파일이 있으면 Paddle 을 직접 돌리지 않고 여기 상자를 읽어 PARSeq 로 읽음
#  [주의]  개발용 final 전처리는 prep_v2\final_dev 에 저장된 것을 읽음 (없으면 같은 설정으로 만들어 저장 — ocr_cns 와 같은 결과)
#  [방향 수]  --angles 36 이면 10° 간격 36방향 → 결과 폴더 '<prep>_a36'.
#             30° 배수 12방향은 같은 사진·같은 설정이라 1라운드 boxes.json 이 있으면 그 상자를 그대로 가져옴 (새 24방향만 검출)
#  [실행 예]
#    .\.venv_paddle_gpu\Scripts\python.exe exp_cns\paddle_boxes_cns.py --split dev --prep orig --limit 20
#    .\.venv_paddle_gpu\Scripts\python.exe exp_cns\paddle_boxes_cns.py --split dev --prep orig --angles 36
#    .\.venv_paddle_gpu\Scripts\python.exe exp_cns\paddle_boxes_cns.py --split dev --prep orig --align long   (폴더 '<prep>_alignlong')
# =====================================================================

import argparse
import json
import os
import time

import align_long_cns as al                   # 긴 알약 축 정렬 (--align long)
import common_cns as cc
import ocr_common_cns as oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train500", "trainall", "exp300test", "hardset", "valdev"] + list(oc.CROP51),
                    help="train500 = 검출기 파인튜닝 라벨용 학습 500장 (make_det_labels_cns.py --step select 먼저)")
    ap.add_argument("--prep", default="orig", help="orig 또는 prep_v2\\final 안의 폴더 이름")
    ap.add_argument("--limit", type=int, default=0, help="시험 운전: 고르게 N 장만")
    ap.add_argument("--angles", type=int, default=12, choices=[12, 36], help="돌리는 방향 수 (12 = 30° 간격, 36 = 10° 간격)")
    ap.add_argument("--align", default="none", choices=["none", "long"],
                    help="long = 긴 알약은 축 각도·+180° 두 방향만 (align_long_cns v3), 나머지는 12방향")
    ap.add_argument("--det-model", default=None,
                    help="파인튜닝해서 내보낸 검출기 폴더 (예: exp_cns\\results\\det_ft1\\infer). 없으면 기본 모델(paddle_t0)")
    ap.add_argument("--unclip", type=float, default=None, help="상자 넓히기 비율 (없으면 모델 설정 1.4)")
    ap.add_argument("--name", default="paddle_t0",
                    help="결과 폴더의 검출기 이름 (예: paddle_ft1, paddle_ft1_u2.0). --det-model·--unclip 을 쓰면 꼭 새 이름으로")
    a = ap.parse_args()
    if (a.det_model or a.unclip) and a.name == "paddle_t0":
        ap.error("--det-model·--unclip 을 쓰면 --name 을 새 이름으로 (기존 paddle_t0 결과를 덮어쓰지 않게)")
    if a.split in ("test", "exp300test"):
        print("[주의] 평가용(test)입니다. 규칙상 마지막에 한 번만 돌립니다.")
    if a.align == "long" and a.angles != 12:
        ap.error("--align long 은 12방향과만 같이 씀")
    items = oc.load_split(a.split, a.limit)
    prep = oc.Prep(a.prep, a.split)
    out = oc.out_dir(a.split, a.name, a.prep, a.limit, a.angles, a.align)
    os.makedirs(out, exist_ok=True)
    # 36방향·축 정렬일 때: 1라운드(12방향) 상자 다시 쓰기 — 같은 사진·같은 돌리기·같은 검출 설정이라 다시 계산해도 같은 상자
    #   (축 정렬에서 원형으로 판정된 사진은 12방향 그대로라 전부 다시 쓰고, 긴 알약의 두 방향만 새로 검출)
    old = {}
    if a.angles != 12 or a.align == "long":
        f12 = oc.out_dir(a.split, a.name, a.prep, 0) / "boxes.json"     # 같은 검출기의 12방향 결과만 (다른 모델 상자 섞지 않게)
        if f12.exists():
            old = json.load(open(f12, encoding="utf-8"))
            print(f"  30° 배수 방향은 1라운드 상자 사용: {f12}")
    res = {}
    t_all = time.perf_counter()
    for k, it in enumerate(items, 1):
        src = cc.imread(it["image_path"])
        img, _ = prep(it, src)
        res[it["file_name"]] = {}
        angs = al.align_angles(src)[0] if a.align == "long" else oc.angles(a.angles)   # 축은 원본 사진에서 구함
        for ang in angs:
            if str(ang) in old.get(it["file_name"], {}):
                res[it["file_name"]][str(ang)] = old[it["file_name"]][str(ang)]   # 검출 시간도 1라운드에 잰 값 그대로
                continue
            t0 = time.perf_counter()
            rot = oc.rotate_color(img, ang)
            if a.det_model or a.unclip:
                boxes = oc.paddle_boxes(rot, None, None, a.det_model, a.unclip)   # 검출 기준은 모델 기본(0.2 / 0.45 = t0 와 같음)
            else:
                boxes = oc.paddle_t0_boxes(rot)
            res[it["file_name"]][str(ang)] = {"boxes": boxes, "det_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if k == 1:
            print(f"  Paddle 장치: {oc._PD.get('dev')}")
        if k % 50 == 0 or k == len(items):
            print(f"  {k}/{len(items)}  경과 {time.perf_counter() - t_all:.0f}초")
    json.dump(res, open(out / "boxes.json", "w", encoding="utf-8"))
    ms = [d["det_ms"] for f in res.values() for d in f.values()]
    first = list(res.values())[0]
    print(f"저장: {out / 'boxes.json'}  /  한 장당 검출 평균 {sum(ms) / len(res):.0f}ms "
          f"(첫 장 {sum(d['det_ms'] for d in first.values()):.0f}ms 포함 — 모델 준비 시간)")


if __name__ == "__main__":
    main()
