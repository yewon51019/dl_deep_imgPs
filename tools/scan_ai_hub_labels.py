#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Hub 경구약제 라벨링데이터(zip) 안에서 정신과 약물(기본값: "정신신경용제")에
해당하는 항목을 찾는 스크립트. 압축을 풀지 않고 zip 안의 JSON 파일을 하나씩
스트리밍으로 읽기 때문에, 전체 압축해제 없이 몇 분 안에 끝납니다.

라벨의 di_class_no 필드는 "[02340]제산제" 처럼 "[숫자코드]한글분류명" 형태라서,
숫자가 아니라 한글 분류명으로 매칭합니다 (엑셀의 "복지부 분류코드 117"과는
다른 번호체계이기 때문).

사용법:
  1) 먼저 JSON 구조를 확인 (필드 이름이 스크립트 예상과 맞는지 확인용)
       python scan_ai_hub_labels.py --peek TL_1_단일.zip

  2) 구조 확인 후 전체 스캔 (기본값: "정신신경용" 포함 항목 찾기)
       python scan_ai_hub_labels.py TL_1_단일.zip TL_2_단일.zip ...
       python scan_ai_hub_labels.py "TL_*_단일.zip"      (와일드카드도 가능)

  3) 다른 분류명을 찾고 싶으면
       python scan_ai_hub_labels.py --name 제산제 TL_1_단일.zip

출력:
  - 화면에 zip별 진행상황과 요약 출력
  - psychiatric_summary.csv 로 결과 저장 (zip 파일명, json 파일명, 품목명, di_class_no)
"""

import sys
import zipfile
import json
import csv
import glob
import argparse

# 실제 라벨 확인 결과, di_class_no 필드는 "[02340]제산제" 처럼
# "[숫자코드]한글분류명" 형태였습니다. 이 숫자는 엑셀에서 쓴
# "복지부 분류코드 117"과는 다른 번호체계라서, 숫자 대신
# 한글 분류명("정신신경용제")으로 매칭합니다 — 훨씬 안전합니다.
TARGET_NAME = "정신신경용"  # 기본값: 정신신경용제

CODE_KEYS = ["di_class_no", "DI_CLASS_NO", "dl_class_no", "class_no", "분류코드", "di_class_no_name"]
NAME_KEYS = ["dl_name", "item_name", "DL_NAME", "제품명", "drug_name"]
PRINT_KEYS = ["print_front", "print_back", "PRINT_FRONT", "PRINT_BACK"]


def find_key(d, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    return None


def flatten_records(data):
    """JSON이 dict 하나든, list든, 중첩된 dict든 최대한 레코드 단위로 풀어서 반환."""
    if isinstance(data, list):
        for item in data:
            yield from flatten_records(item)
    elif isinstance(data, dict):
        yield data
        # 흔히 쓰이는 중첩 키들도 한번 더 내려가봄
        for key in ("images", "annotations", "info", "data"):
            if key in data:
                yield from flatten_records(data[key])


def peek(zip_path, n=2, out_file="peek_output.txt"):
    lines = []
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [x for x in zf.namelist() if x.lower().endswith(".json")]
        header = f"[{zip_path}] JSON 파일 총 {len(json_names)}개 발견\n"
        print(header)
        lines.append(header)
        shown = 0
        for name in json_names:
            with zf.open(name) as f:
                try:
                    data = json.loads(f.read())
                except Exception as e:
                    msg = f"  파싱 실패: {name} ({e})"
                    print(msg)
                    lines.append(msg)
                    continue
            block = f"--- {name} ---\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            print(block)
            lines.append(block)
            shown += 1
            if shown >= n:
                break
        if shown == 0:
            msg = "JSON 파일을 찾지 못했습니다. zip 안 폴더 구조를 확인해주세요:"
            print(msg)
            lines.append(msg)
            for name in zf.namelist()[:20]:
                print(" ", name)
                lines.append(" " + name)

    # 터미널에서 길게 잘려 보일 수 있으니, 항상 파일로도 저장
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n(전체 내용은 {out_file} 파일에도 저장했습니다 — 터미널에서 잘려 보이면 이 파일을 열어서 확인하세요.)")


def scan_zip(zip_path, target_name, results):
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [x for x in zf.namelist() if x.lower().endswith(".json")]
        print(f"[{zip_path}] JSON {len(json_names)}개 스캔 중...")
        matched = 0
        for name in json_names:
            with zf.open(name) as f:
                try:
                    data = json.loads(f.read())
                except Exception:
                    continue
            for rec in flatten_records(data):
                code = find_key(rec, CODE_KEYS)
                if code is None:
                    continue
                # "[02340]제산제" 형태에서 대괄호 뒤 한글 분류명 부분에
                # target_name(예: "정신신경용")이 포함되는지 확인
                if target_name in str(code):
                    item_name = find_key(rec, NAME_KEYS)
                    front = find_key(rec, PRINT_KEYS)
                    results.append({
                        "zip": zip_path,
                        "json_file": name,
                        "item_name": item_name,
                        "print_sample": front,
                        "di_class_no": code,
                    })
                    matched += 1
        print(f"  -> 이 zip에서 \"{target_name}\" 포함 {matched}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+", help="zip 파일 경로 (와일드카드 가능)")
    ap.add_argument("--peek", action="store_true", help="JSON 구조만 미리 출력하고 종료")
    ap.add_argument("--name", default=TARGET_NAME, help="찾을 분류명 일부 (기본값: 정신신경용 = 정신신경용제)")
    ap.add_argument("--out", default="psychiatric_summary.csv", help="결과 저장 파일명")
    args = ap.parse_args()

    paths = []
    for pattern in args.zips:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    if not paths:
        print("zip 파일을 찾을 수 없습니다.")
        sys.exit(1)

    if args.peek:
        peek(paths[0])
        return

    results = []
    for p in paths:
        try:
            scan_zip(p, args.name, results)
        except FileNotFoundError:
            print(f"파일 없음: {p}")
        except zipfile.BadZipFile:
            print(f"zip 형식이 아니거나 손상됨: {p}")

    print(f"\n총 {len(results)}건이 \"{args.name}\" 분류로 확인됨")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["zip", "json_file", "item_name", "print_sample", "di_class_no"])
        writer.writeheader()
        writer.writerows(results)
    print(f"결과 저장: {args.out}")


if __name__ == "__main__":
    main()
