# =====================================================================
#  make_manifest.py  —  "이미지 + 정답" 목록표(manifest.csv) 만들기
# =====================================================================
#
#  [이 코드가 하는 일]  (모델은 안 쓰고, 파일 정리만 해요)
#    1. 이미지(.png)와 라벨(.json)을 이름이 같은 것끼리 짝지어요.
#    2. 라벨에서 필요한 정보(정답 각인, 제형, 모양, 색, 각도 ...)를 꺼내요.
#    3. 약제(약 종류)마다 일부는 "개발용(dev)", 나머지는 "평가용(test)"으로 나눠요.
#    4. 한 장짜리 표(manifest.csv)로 저장하고, 점검용 요약을 화면에 보여줘요.
#
#  [왜 필요해요?]
#    이후 실험(모델 돌리기, 채점)에서 "어떤 이미지의 정답이 뭔지"를
#    이 표 한 장만 보면 알 수 있게 하려는 거예요.
#
#  [폴더 구조]  zip을 풀면 이렇게 돼 있어야 해요.
#      약제각인인식_테스트셋/
#         ├─ images/  K-013392_0_2_0_2_90_000_200.png ...
#         └─ labels/  K-013392_0_2_0_2_90_000_200.json ...   (이름이 같은 쌍)
#
#  [실행 방법]  zip이 두 개면 --roots 에 폴더를 둘 다 적어요.
#      python make_manifest.py --roots 폴더1 폴더2 --out manifest.csv
# =====================================================================

import argparse            # 실행할 때 옵션(--roots 등)을 받는 도구
import csv                 # 표(csv)를 저장하는 도구
import json                # 라벨(json) 파일을 읽는 도구
import random              # 섞기(셔플)에 쓰는 도구
import re                  # 글자 패턴 처리 도구 (공백 지우기용)
import sys                 # 오류가 나면 프로그램을 멈추는 도구
import unicodedata         # 글자 모양을 통일하는 도구 (전각 문자 -> 일반 문자 등)
from collections import Counter, defaultdict   # 개수 세기 / 그룹 묶기 도구
from pathlib import Path   # 파일 경로를 편하게 다루는 도구


# ---------------------------------------------------------------------
# 표(manifest.csv)에 넣을 항목(열) 이름들. 이 순서대로 저장돼요.
# ---------------------------------------------------------------------
COLUMNS = [
    "file_name",    # 이미지 파일 이름
    "image_path",   # 이미지가 있는 전체 경로 (코드가 이미지를 열 때 사용)
    "drug_N",       # 약제 ID (예: K-013392)
    "dl_name",      # 약 이름
    "form",         # 제형 (정제, 연질캡슐제 ...)
    "shape",        # 모양 (원형, 장방형, 타원형 ...)
    "color1",       # 색 1 (예: 노랑, 투명)
    "color2",       # 색 2 (대부분 비어 있음)
    "drug_dir",     # 촬영한 면 (앞면 / 뒷면)
    "back_color",   # 배경색
    "light_color",  # 조명색
    "camera_la",    # 카메라 각도 (예: 90)
    "camera_lo",    # 라벨에 적힌 또 다른 각도값
    "rot_deg",      # 알약 회전 각도 (파일 이름에서 읽음, 예: 000, 060, 120)
    "size",         # 이미지 크기 코드 (예: 200)
    "print_front",  # 앞면 각인 정답
    "print_back",   # 뒷면 각인 정답
    "truth",        # ★ 이 사진의 최종 정답 (앞면 사진이면 앞면 각인, 뒷면이면 뒷면 각인)
    "truth_norm",   # ★ 정답을 채점하기 좋게 다듬은 것 (공백 제거 + 대문자)
    "has_mark",     # 각인이 아니라 "마크(그림)"가 있는지 (1이면 마크 있음)
    "split",        # dev(개발용) 또는 test(평가용)
]


# ---------------------------------------------------------------------
# 함수 1) 정답 글자 다듬기
#   예) " sp eed " -> "SPEED"
#   모델이 읽은 글자와 정답을 비교할 때, 공백/대소문자 때문에 틀리지 않게 하려는 거예요.
#   ※ 팀이 채점 규칙을 다르게 정하면 이 함수만 고치면 돼요.
# ---------------------------------------------------------------------
def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")   # 글자 모양 통일 (없으면 빈 글자 취급)
    text = re.sub(r"\s+", "", text)                    # 공백/줄바꿈 모두 삭제
    return text.upper()                                # 대문자로 통일


# ---------------------------------------------------------------------
# 함수 2) 라벨(json) 파일 한 개 읽기
#   라벨 파일 모양:  { "images": [ { "file_name": ..., "print_front": ..., ... } ] }
#   -> 안쪽 { ... } 한 덩어리(사전)만 꺼내서 돌려줘요.
# ---------------------------------------------------------------------
def read_label(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("images", data)      # "images" 칸이 있으면 그 안쪽을 사용
    if isinstance(info, list):           # 목록 형태라면
        info = info[0]                   # 첫 번째 것만 사용
    return info


# ---------------------------------------------------------------------
# 함수 3) 파일 이름에서 "알약 회전 각도" 꺼내기
#   예) K-013392_0_2_0_2_90_000_200.png
#       이름을 "_"로 자르면 맨 끝에서 둘째(000)가 알약 회전 각도예요.
#   (읽을 수 없으면 빈 칸으로 둬요)
# ---------------------------------------------------------------------
def rotation_from_name(file_name):
    parts = Path(file_name).stem.split("_")   # 확장자(.png) 빼고 "_"로 자르기
    try:
        return int(parts[-2])                 # 끝에서 둘째 조각을 숫자로
    except Exception:
        return ""


# ---------------------------------------------------------------------
# 함수 4) 이미지와 라벨을 짝지어서 "행(한 줄)" 목록 만들기
#   roots: 폴더 목록 (zip을 푼 폴더들)
#   돌려주는 것: (행 목록, 이미지 없는 라벨 목록, 정답이 빈 사진 목록)
# ---------------------------------------------------------------------
def collect_rows(roots):
    rows = []             # 완성된 행들을 모으는 상자
    seen = {}             # 이미 본 파일 이름 (두 폴더에 같은 이름이 있나 확인용)
    missing_images = []   # 라벨은 있는데 이미지가 없는 경우
    empty_truth = []      # 정답 각인이 비어 있는 사진 (채점에서 빼야 함)

    for root in roots:
        root = Path(root)
        label_files = sorted((root / "labels").glob("*.json"))   # 라벨 파일 전부

        if not label_files:   # 하나도 없으면 폴더 위치가 틀린 것 -> 멈추고 안내
            sys.exit(f"[오류] {root}/labels 안에 json 파일이 없어요. 폴더 이름/위치를 확인하세요.")

        for label_path in label_files:
            info = read_label(label_path)                              # 라벨 내용 읽기
            file_name = info.get("file_name") or info.get("imgfile") or (label_path.stem + ".png")

            # ---- (a) 짝이 되는 이미지 찾기 ----
            image_path = root / "images" / file_name
            if not image_path.exists():
                # 확장자만 다른 경우(.jpg 등)도 찾아봐요
                candidates = list((root / "images").glob(label_path.stem + ".*"))
                if candidates:
                    image_path = candidates[0]
                else:
                    missing_images.append(label_path.name)    # 못 찾음 -> 기록하고 건너뜀
                    continue

            # ---- (b) 두 폴더에 같은 이름이 있으면 한 번만 사용 ----
            if file_name in seen:
                print(f"[경고] 파일 이름이 겹쳐요: {file_name} ({seen[file_name]} 와 {root})")
                continue
            seen[file_name] = str(root)

            # ---- (c) 정답 각인 정하기 ----
            #   앞면 사진이면 print_front, 뒷면 사진이면 print_back 이 정답이에요.
            is_front = (info.get("drug_dir") == "앞면")
            print_front = info.get("print_front", "") or ""
            print_back = info.get("print_back", "") or ""
            truth = print_front if is_front else print_back

            if not truth.strip():                 # 정답이 비어 있으면 따로 기록
                empty_truth.append(file_name)

            # 마크(그림 표시) 여부: 마크 정보가 하나라도 있으면 1
            has_mark = 1 if (info.get("mark_code_front_anal") or info.get("mark_code_back_anal")) else 0

            # ---- (d) 한 줄(행) 만들어서 상자에 담기 ----
            rows.append({
                "file_name": file_name,
                "image_path": str(image_path),
                "drug_N": info.get("drug_N", ""),
                "dl_name": info.get("dl_name", ""),
                "form": info.get("dl_custom_shape", ""),    # 라벨 항목 이름이 dl_custom_shape
                "shape": info.get("drug_shape", ""),
                "color1": info.get("color_class1", ""),
                "color2": info.get("color_class2", ""),
                "drug_dir": info.get("drug_dir", ""),
                "back_color": info.get("back_color", ""),
                "light_color": info.get("light_color", ""),
                "camera_la": info.get("camera_la", ""),
                "camera_lo": info.get("camera_lo", ""),
                "rot_deg": rotation_from_name(file_name),
                "size": info.get("size", ""),
                "print_front": print_front,
                "print_back": print_back,
                "truth": truth,
                "truth_norm": normalize(truth),
                "has_mark": has_mark,
                "split": "",                                 # 나누기는 다음 단계에서 채워요
            })

    return rows, missing_images, empty_truth


# ---------------------------------------------------------------------
# 함수 5) 개발용(dev) / 평가용(test) 나누기
#   - 약제(약 종류)마다 따로 섞어서, 각 약제가 dev와 test에 모두 들어가게 해요.
#   - dev_ratio=0.2 이면 약제별로 약 20%가 dev, 나머지 80%가 test.
#   - dev_ratio=0 이면 나누지 않고 전부 test (방법을 고르지 않고 결과만 나란히 볼 때).
#   - seed(난수 시드)를 고정하므로 같은 코드를 돌리면 팀원도 똑같이 나뉘어요.
#
#   dev : 전처리/모델/설정을 고르고 조정할 때만 사용
#   test: 최종 점수를 낼 때 "한 번만" 사용 (여기서 설정을 고치면 안 돼요!)
# ---------------------------------------------------------------------
def assign_split(rows, dev_ratio, seed):
    # dev_ratio 가 0 이하이면 "나누지 않기" -> 전부 test 로 표시
    if dev_ratio <= 0:
        for row in rows:
            row["split"] = "test"
        return

    rng = random.Random(seed)            # 시드를 고정한 섞기 도구

    by_drug = defaultdict(list)          # 약제별로 행을 묶는 상자
    for row in rows:
        by_drug[row["drug_N"]].append(row)

    for drug, group in sorted(by_drug.items()):
        rng.shuffle(group)               # 같은 약제 안에서 순서를 섞고
        if len(group) > 1:
            n_dev = max(1, round(len(group) * dev_ratio))   # 최소 1장은 dev로
        else:
            n_dev = 0                                       # 1장뿐이면 전부 test
        for i, row in enumerate(group):
            row["split"] = "dev" if i < n_dev else "test"   # 앞쪽 n_dev장을 dev로


# ---------------------------------------------------------------------
# 함수 6) 점검용 요약 출력
#   "몇 장인지, 제형/약제/면/각도가 골고루 있는지, 문제 있는 사진은 없는지"를 보여줘요.
#   실행 후 이 요약을 꼭 한번 훑어보세요. (쏠림, 장수 부족을 여기서 발견해요)
# ---------------------------------------------------------------------
def print_summary(rows, missing_images, empty_truth, out_path):
    print(f"\n이미지+라벨 쌍: {len(rows)}장  "
          f"(이미지 없는 라벨 {len(missing_images)}개, 정답 각인이 빈 사진 {len(empty_truth)}장)")

    # (보여줄 이름, 표의 열 이름) 목록
    items = [("제형", "form"), ("약제", "drug_N"), ("앞/뒷면", "drug_dir"),
             ("카메라각도", "camera_la"), ("알약회전(도)", "rot_deg"), ("모양", "shape"),
             ("분할", "split"), ("마크 있음", "has_mark")]

    for title, key in items:
        counts = Counter(row[key] for row in rows)                   # 값별로 개수 세기
        text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))
        print(f"- {title}: {text}")

    if empty_truth:
        print("\n[주의] 정답 각인이 빈 사진 (채점에서 빼야 해요):", empty_truth[:5], "...")
    if missing_images:
        print("[주의] 이미지를 못 찾은 라벨:", missing_images[:5], "...")
    print(f"\n저장 완료: {out_path}")


# ---------------------------------------------------------------------
# 메인: 위의 함수들을 순서대로 실행
# ---------------------------------------------------------------------
def main():
    # 1) 실행할 때 받는 옵션 정의
    parser = argparse.ArgumentParser(description="이미지+정답 목록표(manifest.csv) 만들기")
    parser.add_argument("--roots", nargs="+", required=True, help="images/, labels/ 가 들어있는 폴더(여러 개 가능)")
    parser.add_argument("--out", default="manifest.csv", help="저장할 표 파일 이름")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="개발용 비율 (기본 0.2 = 20%%, 0이면 나누지 않음)")
    parser.add_argument("--seed", type=int, default=42, help="섞기 시드 (같으면 같은 결과)")
    args = parser.parse_args()

    # 2) 이미지와 라벨 짝짓기
    rows, missing_images, empty_truth = collect_rows(args.roots)

    # 3) 개발용/평가용 나누기
    assign_split(rows, args.dev_ratio, args.seed)

    # 4) 파일 이름 순으로 정렬해서 표로 저장 (utf-8-sig: 엑셀에서 한글이 깨지지 않게)
    rows.sort(key=lambda r: r["file_name"])
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 5) 점검용 요약 보여주기
    print_summary(rows, missing_images, empty_truth, args.out)


if __name__ == "__main__":
    main()
