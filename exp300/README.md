# exp300 — AI Hub 테스트셋(약 300장) A실험

| 파일 | 하는 일 |
|---|---|
| `make_manifest.py` | 이미지+라벨 짝짓기 → **채점표** `manifest.csv` (정답, 제형, 각도, dev/test 분할) |
| `make_search_db.py` | 라벨 → **검색 DB** `search_db.csv` (약 1종 = 1줄) |
| `run_exp300.py` | 알약 자르기 → 전처리 P0~P4 → 모델 5개로 회전하며 읽기 → 다수결 → DB 검색 → `predictions.csv` |
| `evaluate_exp300.py` | 정확도·macro 정밀도/재현율/F1, 각인 일치율, 상위 조합, 혼동행렬, 조건별 분석 |
| `run_exp300_prepare.ipynb` | (준비) 채점표·검색 DB만 만들어 확인 |
| `run_exp300.ipynb` | (실험) 설치 → 시험 운전 → 1단계(dev) → 상위 4개 → 2단계(test) → 최종 채점 |

흐름: `사진 → 자르기 → 전처리 → OCR(회전 12번) → 다수결 → 검색 DB 매칭(exact/edit/attr) → 채점`

- 모델: tesseract, easyocr, doctr, parseq(CRAFT+PARSeq), trocr(CRAFT+TrOCR)
- 전처리: P0 없음, P1 조명평탄화, P2 반사광제거+평탄화, P3 CLAHE, P4 2배확대+CLAHE
- 순위 기준: `edit` 매칭(편집거리) F1. `attr`(모양·색 필터)은 DB가 작으면 점수가 부풀 수 있어 참고용.
- 각인 정리 규칙: 대문자 + 글자·숫자만 남김 (`norm()`).

## 보정 실험 (correct_exp300.py)
OCR을 다시 돌리지 않고, 저장된 각도별 후보(`candidates`)로 매칭 방식만 바꿔 점수를 다시 계산합니다.
V0 기준선 → V1 부분일치 → V2 헷갈리는 글자(0/O 등) → V3 후보 전체(최고점) → V4 후보 합산 → V5 모델 합치기.
방법은 개발 세트로 고르고, 시험 세트는 한 번만 적용합니다.

## 분리 실험 + 오류 분석 (analyze_exp300.py)
A: 후보 전체만 / +부분일치 / +글자통일 / +둘 다 를 개발·시험 세트에서 비교.
B: V3로도 틀린 사진을 원인(읽은 글자 없음, DB와 너무 다름, 동점, DB상 각인 동일, 정답을 읽고도 다른 약 선택, 일부만 읽음, 잘못 읽음)별로 분류하고 `errors_*.csv` 저장.

## PaddleOCR 실험 (run_exp300_paddle.py + run_exp300_paddle.ipynb) — 팀원(B)용
Colab에서는 `run_exp300_paddle.ipynb` 를 열고 위에서부터 실행하면 됩니다 (3번 셀 zip 경로만 수정).
run_exp300.py 는 건드리지 않고 PaddleOCR 모델 하나만 끼워 넣어 같은 조건(자르기·전처리 P0~P4·회전 투표·매칭·CSV 형식)으로 실행합니다. 결과 CSV 는 evaluate_exp300.py 로 그대로 채점돼요.
설치: `pip install paddlepaddle paddleocr` (v2.x / v3.x 자동 판별)
1. 개발용: `python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv --split dev --out pred_dev_paddle.csv`
2. 전처리 고르기: `python evaluate_exp300.py --pred pred_dev_paddle.csv --top 1`
3. 평가용(한 번만): `python run_exp300_paddle.py --manifest manifest.csv --db search_db.csv --split test --combos paddleocr:P3 --out pred_test_paddle.csv` → `python evaluate_exp300.py --pred pred_test_paddle.csv --cm all`
※ manifest.csv 는 A실험과 같은 파일을 쓸 것(같은 개발/평가 분할). 평가용 점수를 보고 전처리를 다시 고르지 말 것.

### 추가 전처리 6종 (extra_preps.py, P5~P10)
PaddleOCR 실험(run_exp300_paddle.py)에서는 P0~P4 에 더해 6종이 기본으로 같이 실행됩니다 (총 11개).
P5 adaptive_threshold · P6 blackhat · P7 closing · P8 global_otsu · P9 localized_otsu(3×3 칸별 Otsu) · P10 tophat
(grayscale = P0, clahe = P3 과 같은 처리). 일부만 쓰려면 `--preps P0,P5,P9` 처럼 지정. 커널·블록 크기는 extra_preps.py 맨 위 상수로 조절.

### PaddleOCR B 방식 점수 (run_exp300_paddle_b.py)
팀원 val_paddle.py 와 같은 방식(원본 사진, 검출 기준 0.0, 글자 조각 중 하나라도 정답이면 성공)으로 같은 평가용 237장을 재서 팀원의 B 실험 결과와 이어 보기 위한 보조 점수입니다.
`hit_B`(팀원 코드 그대로) / `hit_norm`(정규화, 빈값·'마크' 제외) 두 가지로 셉니다. A 방식의 정확도/F1 과 직접 비교하지 마세요.
`python run_exp300_paddle_b.py --manifest manifest.csv --split test --out pred_test_paddle_B.csv`
