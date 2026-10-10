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
