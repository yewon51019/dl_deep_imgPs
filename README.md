# 정신건강의학과 약 분류 AI — 알약 각인 인식 실험

캡스톤디자인 팀 프로젝트 (김태린, 남예원, 장혜정, 조하은)
스마트폰으로 찍은 알약 사진에서 각인(글자)을 읽어 약을 판별하는 파이프라인을 실험하는 저장소입니다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `imprint_ocr/` | 각인 인식 실험 코드(1~3차), 정답 라벨, 판별 후보 목록, 테스트 사진, 결과 |
| `tools/scan_ai_hub_labels.py` | AI Hub 경구약제 라벨(zip)을 풀지 않고 정신과약 항목을 찾는 스크립트 |

## 빠른 실행

```
pip install -r requirements.txt
cd imprint_ocr
python imprint_experiment_v3.py run --labels labels_v2.csv --db pill_db.csv --engine easyocr
```

자세한 실행 방법은 `imprint_ocr/README.md` 참고.

## 실험 기록 (테스트 사진 12장: 인쇄 각인 alza 18·27·36 / 음각 DK·YJ)

| 차수 | 무엇을 비교했나 | 주요 결과 (OCR 엔진: Tesseract) |
|---|---|---|
| 1차 `imprint_experiment.py` | 전처리 4종: 원본 / CLAHE / 적응형 이진화 / Top-hat | 여러 알약 사진에서 알약 분리 6/6 성공. 글자 완전일치는 전부 0% → 이진화는 인쇄·음각 모두 손해 |
| 2차 `imprint_experiment_v2.py` | 원본 vs CLAHE × 매칭 3종(없음 / 편집거리 / 편집거리+색·모양) | 원본 + 편집거리 + 색·모양에서 약 판별 Top-1 33.3% (인쇄 66.7%, 음각 0%) |
| 3차 `imprint_experiment_v3.py` | 배경제거만 / +조명평탄화 / +반사광제거 / 둘 다 / 유형별 자동 | 배경제거만이 최고(33.3%). 조명평탄화는 인쇄 글자를 흐리게 해 손해. 인쇄/음각 자동 구분 12/12 |

- 판별 후보 목록(`pill_db.csv`)은 현재 사진 속 5종뿐이라 수치가 실제보다 높게 나올 수 있음. 정신과약+각성제 800종으로 확장 예정.
- 이 환경에서는 EasyOCR을 설치할 수 없어 Tesseract 결과만 있음. 다음 단계: EasyOCR로 동일 실험 재실행.
