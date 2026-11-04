# 기술 설계

최종 방식 **pft1**의 설계입니다. 순서도는 `exp_cns/results/정신과약_약판별_최종_9주차.docx` 부록에 있습니다. 앱 구조는 아직 정하지 않았습니다.

## 처리 흐름

```
사진 → 긴 변 400 정사각형 → 긴 알약 판정 → 방향별 돌리기
     → 방향마다 글자 상자 찾기 → 상자마다 읽기 → 글자 이어 붙이기 → 확신도 최고 방향 선택
     → 약 1,211종과 각인 점수 T → 겉모습 점수 A → 최종 점수 → 표시 결정
```

## 단계별 기준

| 단계 | 방법 | 값 | 코드 |
|---|---|---|---|
| 입력 | 비율 유지, 빈 곳은 모서리 색 | 400×400, 전처리 없음 | `common_cns.fit_side` |
| 긴 알약 | 색 거리 영역 + `minAreaRect` | 비율 > 1.3, 덮은 비율 ≥ 0.8 | `align_long_cns.py` (v3) |
| 방향 | 긴 알약 2방향, 그 외 12방향 | 축 각도·+180° / 30° 간격 | `align_long_cns.align_angles` |
| 글자 찾기 | PP-OCRv6 medium 파인튜닝 `det_ft1` | 0.2 / 0.45, 넓히기 2.0, 넓이 40 미만·사진 절반 초과 상자 제외 | `paddle_boxes_cns.py` |
| 읽기 | PARSeq 파인튜닝 `parseq_ft1_sure` | 입력 32×128, 상자 없으면 사진 전체 | `ocr_cns.py` |
| 이어 붙이기 | 긴 알약: 세로 절반 이상 겹치면 한 줄. 그 외: 위→아래, 왼→오른 | | `ocr_cns.join_line`, `run_exp300.join_boxes` |
| 방향 고르기 | 글자가 나온 방향 중 확신도 평균 최고 | | `ocr_cns.pick_best` |
| 각인 점수 | 편집거리 유사도 + 헷갈리는 글자 통일 + 부분일치 | 0.3 미만 0점, 후보 1~30자, 상자 2~4개 순서 바꾸기 | `correct_exp300.sim_v`, `match2_cns.py` |
| 방향 합치기 | T = max(고른 방향, λ × 다른 방향) | λ = 0.9 | `match2_cns.combine` |
| 겉모습 | ResNet18, 모양 4 + 제형 3 + 색 12칸 로그 확률 합 | 확률 0.02~0.98로 자름 | `attr_cnn_cns.py`, `match_cns.attr_score` |
| 최종 점수 | T + w × A | w = 0.02 | `display_cns.score` |
| 표시 | 확정 / 후보 / 요청 | 0.95, 0.1, 5개 | `display_cns.shown`, `final_wrap_cns.py` |

경로는 모두 `exp_cns/` 기준입니다 (`correct_exp300`, `run_exp300`은 `exp300/`).

## 모델과 데이터 파일

GitHub에 올리지 않으므로 로컬에만 있습니다.

| 파일 | 내용 |
|---|---|
| `exp_cns/results/det_ft1/infer/` | 글자 찾기 모델 |
| `exp_cns/results/parseq_ft1_sure/model/best.pt` | 읽기 모델 |
| `exp_cns/results/attr_cnn_v1/model/best.pt` | 겉모습 모델 |
| `exp_cns/results/match_v1/search_db_cns.csv` | 약 목록 1,211종 |

## 실행 환경

| 환경 | 쓰임 |
|---|---|
| `.venv` | PyTorch, PARSeq, 겉모습 모델 |
| `.venv_paddle_gpu` | 글자 상자 계산 (결과는 `boxes.json`) |
| `.venv_paddle_train` | 글자 찾기 모델 학습 |
| `.venv_label` | PPOCRLabel 라벨 도구 |

## 주의할 점

- Windows에서 Paddle을 torch보다 먼저 불러오면 DLL 오류가 납니다. `import torch`를 먼저 합니다.
- `cv2.imread`는 한글 경로에서 실패합니다. `run_exp300.imread_any`를 씁니다.
- paddleocr 3.7.0은 검출 기준 0.0을 기본값으로 처리합니다. 낮추려면 0.05처럼 0이 아닌 값을 씁니다.
- OpenCV·numpy 버전이 바뀌면 결과가 달라진 적이 있습니다.
- 저장되는 상자는 반듯한 사각형이라 글자 기울기 정보가 없습니다.
- `common_cns.ROOT`는 파일 위치 기준이고 `manifest.csv`, `det_ft*.yaml`에도 경로가 들어 있습니다. **실험 폴더를 옮기면 깨집니다.**

## 앱 설계에서 풀어야 할 문제 (미정)

- 지금은 상자를 다른 환경에서 미리 계산해 파일로 넘깁니다. 앱은 한 번에 처리해야 하므로 검출과 읽기 환경을 합치거나 분리된 서비스로 만들어야 합니다.
- 지금 코드는 저장된 결과를 읽는 실험용입니다. `src/`에는 사진 한 장을 받아 끝까지 처리하는 코드를 새로 만듭니다. `각인인식_코드_따라하기.docx`의 코드가 출발점이 될 수 있습니다.
