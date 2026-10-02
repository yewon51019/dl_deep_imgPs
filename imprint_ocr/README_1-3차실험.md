# 알약 각인 추출 비교 실험 — 실행 방법

## 1. 설치 (처음 한 번)

```
pip install opencv-python numpy matplotlib easyocr
```

- EasyOCR은 처음 실행할 때 모델 파일을 인터넷에서 자동으로 내려받습니다.
- Tesseract로도 돌려보려면 Tesseract 프로그램을 따로 설치하고 `pip install pytesseract`도 해 주세요.

## 2. 폴더 구성

```
pill_exp/
  imprint_experiment.py   실험 코드
  labels.csv              이미지별 정답 각인
  images/                 촬영한 사진 (single_*.jpg = 알약 1개, multi_scene.jpg = 여러 개)
```

## 3. 실행 순서

```
# (1) 여러 알약 사진에서 알약을 하나씩 잘라내기
python imprint_experiment.py crop --input images/multi_scene.jpg --out crops

# (2) crops/crop_overview.png를 열어 몇 번이 어떤 알약인지 확인한 뒤 labels.csv에 정답 적기

# (3) 실험 실행
python imprint_experiment.py run --labels labels.csv --engine easyocr --out results_easyocr
#     GPU가 있으면 맨 뒤에 --gpu 추가
#     Tesseract와 비교하려면 --engine tesseract --out results_tesseract
```

## 4. 결과 파일

| 파일 | 내용 |
|---|---|
| results_*/results_summary.csv | 방식별 평균: 완전일치 정확도, 문자 단위 정확도, 방향을 맞혔다고 가정한 상한, 평균 시간 |
| results_*/results_detail.csv | 이미지 × 방식별 인식 결과 |
| results_*/summary_chart.png | 방식별 막대그래프 (보고서에 바로 넣을 수 있음) |
| results_*/preprocess_examples.png | 방식별 전처리 결과 이미지 예시 |

## 5. 사진을 더 추가할 때

- 알약 1개 사진은 `images/`에 넣고 labels.csv에 `images/파일명.jpg,정답` 한 줄을 추가하면 됩니다.
- 여러 알약 사진은 (1)번 crop 명령으로 잘라낸 뒤 같은 방식으로 정답을 적으면 됩니다.
- 정답은 대소문자와 공백을 무시하고 비교합니다 (`alza 18` = `ALZA18`).

---

# 2차 실험 (imprint_experiment_v2.py)

```
python imprint_experiment_v2.py crop --input images/multi_scene.jpg --out crops
python imprint_experiment_v2.py run --labels labels_v2.csv --db pill_db.csv --engine easyocr
```

- labels_v2.csv : `이미지경로,정답각인,각인유형(인쇄/음각)`
- pill_db.csv   : `약이름,각인,모양(원형/장방형),색(하양/회색/노랑...)` — 판별 후보 목록. 800종으로 늘릴 때 이 파일만 바꾸면 됨
- 결과: results_easyocr/ 폴더에 detail.csv, summary.csv, by_type.csv, errors.csv, chart_summary.png

---

# 3차 실험 (imprint_experiment_v3.py) — 전처리 조합 비교

```
python imprint_experiment_v3.py run --labels labels_v2.csv --db pill_db.csv --engine easyocr
```

- 전처리: A 배경제거만 / B +조명평탄화 / C +반사광제거 / D +반사광+평탄화 / AUTO(인쇄→A, 음각→D 자동 선택)
- detail.csv에 auto_type(자동 판정한 인쇄/음각), dark_ink_ratio(진한 잉크 비율), glare_ratio(반사광 비율) 기록
