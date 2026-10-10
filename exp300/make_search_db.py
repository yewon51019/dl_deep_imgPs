# =====================================================================
#  make_search_db.py  —  "검색용 약 DB(search_db.csv)" 만들기
# =====================================================================
#
#  [이 코드가 하는 일]
#    라벨(json) 파일들을 읽어서, "약 하나당 한 줄"짜리 표를 만들어요.
#    (같은 약의 사진이 24장이면 라벨도 24개지만, DB에는 1줄만 남겨요)
#
#  [어디에 쓰나요?]
#    실험에서 이미지로 각인을 읽은 뒤, "이 각인을 가진 약이 뭐지?" 하고
#    찾아볼 때 이 표를 검색해요.  (= 앱에서 식약처 DB가 할 역할)
#
#  [manifest.csv 와 차이]
#    manifest.csv  : 이미지 1장당 1줄  -> 채점표 (정답 확인용, 검색에 쓰면 안 됨)
#    search_db.csv : 약 1종당 1줄      -> 검색 대상
#
#  [주의]  테스트셋 300장의 라벨로만 만들면 후보가 27종뿐이라 너무 쉬워요.
#          가능하면 AI Hub 전체 라벨(정신·신경계 852종 등) 폴더를 --label-roots 에 넣으세요.
#          폴더 안의 하위 폴더까지 모든 .json 을 찾아서 읽어요.
#
#  [실행 방법]
#      python make_search_db.py --label-roots 라벨폴더1 라벨폴더2 --out search_db.csv
# =====================================================================

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# 표에 넣을 항목(열) 이름
COLUMNS = [
    "drug_N",        # 약제 ID (예: K-013392)  ← 채점할 때 정답과 비교하는 기준
    "dl_name",       # 약 이름
    "dl_name_en",    # 영어 이름
    "dl_material",   # 성분
    "form",          # 제형 (정제, 연질캡슐제 ...)
    "shape",         # 모양 (원형, 장방형 ...)
    "color1",        # 색 1
    "color2",        # 색 2
    "print_front",   # 앞면 각인 (원래 글자)
    "print_back",    # 뒷면 각인 (원래 글자)
    "front_norm",    # 앞면 각인 (검색용으로 다듬은 글자)
    "back_norm",     # 뒷면 각인 (검색용으로 다듬은 글자)
    "n_labels",      # 이 약의 라벨 파일 개수 (확인용)
]


# 각인 글자 다듬기: manifest.py 와 "똑같은 규칙"이어야 비교가 공정해요.
def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", "", text)
    return text.upper()


# 라벨 json 한 개 읽기 (형식: {"images": [ {...} ]})
def read_label(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("images", data)
    if isinstance(info, list):
        info = info[0] if info else {}
    return info if isinstance(info, dict) else {}


# 여러 라벨에서 같은 칸의 값이 다를 때, 가장 많이 나온 값을 고르기
def most_common(values):
    values = [v for v in values if v not in (None, "")]
    return Counter(values).most_common(1)[0][0] if values else ""


def main():
    parser = argparse.ArgumentParser(description="검색용 약 DB(search_db.csv) 만들기")
    parser.add_argument("--label-roots", nargs="+", required=True,
                        help="라벨 json 이 들어있는 폴더들 (하위 폴더까지 모두 찾음)")
    parser.add_argument("--out", default="search_db.csv")
    args = parser.parse_args()

    # 1) 약제 ID 별로 라벨 정보를 모으기
    by_drug = defaultdict(list)
    n_files, n_bad = 0, 0
    for root in args.label_roots:
        for path in Path(root).rglob("*.json"):       # 하위 폴더까지 전부
            n_files += 1
            try:
                info = read_label(path)
            except Exception:
                n_bad += 1                            # 깨진 파일은 건너뛰기
                continue
            if info.get("drug_N"):
                by_drug[info["drug_N"]].append(info)

    # 2) 약 하나당 한 줄로 합치기
    rows = []
    for drug, infos in sorted(by_drug.items()):
        pick = lambda key: most_common([i.get(key, "") for i in infos])
        pf, pb = pick("print_front"), pick("print_back")
        rows.append({
            "drug_N": drug,
            "dl_name": pick("dl_name"),
            "dl_name_en": pick("dl_name_en"),
            "dl_material": pick("dl_material"),
            "form": pick("dl_custom_shape"),
            "shape": pick("drug_shape"),
            "color1": pick("color_class1"),
            "color2": pick("color_class2"),
            "print_front": pf,
            "print_back": pb,
            "front_norm": normalize(pf),
            "back_norm": normalize(pb),
            "n_labels": len(infos),
        })

    # 3) 저장 (utf-8-sig: 엑셀에서 한글 안 깨짐)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 4) 점검용 요약
    no_print = sum(1 for r in rows if not r["front_norm"] and not r["back_norm"])
    same_print = Counter((r["front_norm"], r["back_norm"]) for r in rows if r["front_norm"] or r["back_norm"])
    dup = sum(c for c in same_print.values() if c > 1)
    print(f"읽은 라벨 파일: {n_files}개 (읽기 실패 {n_bad}개)")
    print(f"약 종류: {len(rows)}종  (앞/뒤 각인이 모두 빈 약 {no_print}종)")
    print(f"각인이 똑같은 약끼리 겹치는 경우: {dup}종  ← 각인만으로는 구분이 안 돼서 색/모양이 필요")
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
