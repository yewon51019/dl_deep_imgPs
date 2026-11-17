# exp_cns/final — 알약 사진 → 약 이름 후보 (최종본)

정신과 약 실험(`exp_cns/`)에서 정한 최종 방식(**pft1**)을 **이 폴더만 있으면 실행되게** 정리한 최종본입니다.
실험 코드는 실험마다 파일이 하나씩 늘어나서(약 150개) 최종 방식이 여러 파일에 흩어져 있었습니다.
여기에는 **최종 방식에 필요한 계산만** 단계별 파일로 옮겼고, 계산 방법과 설정값은 바꾸지 않았습니다.

- 정리한 날: 2026-11-17
- 이 폴더의 코드는 다른 실험 파일을 import 하지 않습니다. 모든 경로가 이 폴더 기준이라, 폴더째 옮기거나 압축해 보내도 됩니다.
- **모델 파일(`models/`)은 GitHub 에 올리지 않습니다** (이유는 4장). GitHub 에서 받은 경우 모델 파일을 따로 받아 넣어야 실행됩니다.

## 1. 폴더 구성

```
exp_cns/final/
├─ README.md                 이 설명
├─ run_photo.py              실행 파일 (--step crop → boxes → read)
├─ requirements_main.txt     본 환경 패키지
├─ requirements_paddle.txt   Paddle 환경 패키지
├─ db/search_db_cns.csv      검색 DB 1,211종 (약 이름, 앞·뒷면 각인, 모양·제형·색)
├─ pill_id/                  단계별 코드
│   ├─ config.py             경로·설정값
│   ├─ imageio.py            한글 경로 읽기·저장, 크기 맞추기, 돌리기
│   ├─ s0_crop.py … s6_display.py
├─ models/                   (GitHub 에 없음) 모델 파일 3종, 약 194MB
│   ├─ det_ft1/inference.json · inference.pdiparams · inference.yml   글자 상자 검출기 (62MB)
│   ├─ parseq_ft1.pt                                                  글자 읽기 PARSeq (95MB)
│   └─ attr_cnn.pt                                                    겉모습 CNN (45MB)
└─ output/                   (GitHub 에 없음) 실행 결과
```

## 2. 전체 흐름

```
사진 1장 (알약 여러 개, 배경째)
  └─ 0단계 알약 찾기·자르기 ──── 알약마다 컬러 사진 1장
       └─ 1단계 방향 정하기 ──── 긴 알약 2방향 / 그 밖 12방향
            └─ 2단계 글자 상자 찾기 ──── 방향마다 상자 (Paddle 환경)
                 └─ 3단계 글자 읽기 + 방향 고르기
                      └─ 4단계 겉모습 CNN (모양·제형·색)
                           └─ 5단계 검색 DB 와 비교 → 약 순위
                                └─ 6단계 표시 (확실 1개 / 후보 / 다시 찍기)
```

| 파일 | 단계 | 하는 일 | 원래 실험 파일 |
|---|---|---|---|
| `s0_crop.py` | 0 | 배경색과 다른 덩어리를 찾아 알약마다 자름 | `photo_demo_cns` |
| `s1_orient.py` | 1 | 긴 알약이면 축 각도·+180°, 아니면 30° 간격 12방향 | `align_long_cns` (v3) |
| `s2_detect.py` | 2 | 파인튜닝 PaddleOCR 검출기(det_ft1, unclip 2.0)로 글자 상자 | `ocr_common_cns`, `paddle_boxes_cns` |
| `s3_read.py` | 3 | 파인튜닝 PARSeq 로 상자마다 읽고, 이어 붙이고, 확신도 최고 방향 고르기 | `ocr_cns`, `exp300/run_exp300` |
| `s4_appearance.py` | 4 | ResNet18 로 모양 4칸·제형 3칸·색 12칸 확률 | `attr_cnn_cns` |
| `s5_match.py` | 5 | 각인 점수 T + 0.02 × 겉모습 점수 A 로 DB 1,211종 순위 | `match_cns`, `match2_cns`, `exp300/correct_exp300` |
| `s6_display.py` | 6 | 몇 개를 보여 줄지 결정 | `display_cns`, `final_wrap_cns` |

### 단계별 설명

**0단계 알약 찾기·자르기** — 사진 테두리 색을 배경으로 보고, 배경과 색이 다른 덩어리를 알약 후보로 찾습니다.
너무 작거나(먼지) 큰 것, 사진 가장자리에 닿은 것(잘린 알약·탁자), 가늘고 긴 띠(빛 줄무늬), 모양이 엉성한 것은 버립니다.
남은 덩어리를 사방 15% 여유를 두고 컬러 그대로 자릅니다.
*한계:* 서로 닿은 알약은 한 덩어리로 나오고, 테두리가 배경이 아닌 사진(복잡한 배경, 강한 그림자)에서는 약합니다.

**1단계 방향 정하기** — 각인이 어느 쪽으로 서 있는지 모르므로 사진을 여러 각도로 돌려 읽습니다.
긴 알약(긴변/짧은변 > 1.3)은 긴 축을 가로로 눕히는 각도와 그 반대(+180°) 두 방향만 읽고, 원형 등은 30°씩 12방향을 모두 읽습니다.

**2단계 글자 상자 찾기** — 학습용 사진으로 파인튜닝한 PaddleOCR 검출기가 방향마다 글자 위치(상자)를 찾습니다.
알약 전체를 덮는 큰 상자(사진 넓이 절반 초과)와 아주 작은 상자는 뺍니다.
이 단계만 Paddle 환경에서 돌리고 결과를 `boxes.json` 으로 저장합니다 (한 환경에 Paddle GPU 판과 torch 를 같이 깔면 OpenCV·numpy·CUDA 라이브러리가 엉켜 결과가 달라질 수 있기 때문).

**3단계 글자 읽기** — 상자마다 파인튜닝 PARSeq 가 글자를 읽습니다. 상자를 하나도 못 찾으면 사진 전체를 한 줄로 읽습니다.
상자 글자를 이어 붙이는 순서는 긴 알약이면 '줄 묶기'(세로로 겹치는 상자끼리 한 줄), 그 밖은 '위→아래, 왼→오른'입니다.
방향마다 확신도(상자별 확신도 평균)를 내고 가장 높은 방향을 고릅니다. 정답(DB)을 보고 방향을 고르지 않습니다.

**4단계 겉모습 CNN** — 자른 알약 사진에서 모양·제형·색 확률을 냅니다. 약 이름을 맞히는 모델이 아니라, 5단계에서 각인이 비슷한 약끼리 가르는 데 씁니다.

**5단계 약 판별** — 읽은 글자를 DB 의 앞·뒷면 각인과 비교합니다.
- 비교 후보: 고른 방향의 글자 + 고른 방향 상자 글자들의 모든 순서(상자 2~4개), 그리고 다른 방향의 글자(점수 × 0.9)
- 유사도: 편집거리 기반 (L − 편집거리)/(L + 편집거리). 헷갈리는 글자(0·O·Q·D, 1·I·L, 5·S, 8·B, 2·Z, 6·G, U·V)는 같은 글자로 보고, 읽은 글자가 DB 각인 안에 통째로 있으면 부분 점수
- 최종 점수 = 각인 점수 + 0.02 × 겉모습 점수. 1등이 동점이면 '판정 없음'

**6단계 표시** — 정밀도(답을 낸 것 중 맞은 비율)를 우선합니다.
1등 점수 ≥ 0.95 이고 2등과 0.1 이상 차이 나면 약 1개만, 아니면 후보 여러 개를 보여 주고, 후보가 5개를 넘으면 다른 면이나 다시 찍기를 요청합니다.

## 3. 설치 (Windows, 파이썬 3.12)

가상환경을 두 개 만듭니다. 아래는 이 폴더(`exp_cns\final`)에서 실행하는 예입니다.

```powershell
# 본 환경: torch 는 GPU 에 맞는 판을 먼저 (GTX 10xx 는 cu126, GPU 가 없으면 --index-url 없이 CPU 판)
py -3.12 -m venv .venv_main
.\.venv_main\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv_main\Scripts\python.exe -m pip install -r requirements_main.txt

# Paddle 환경: paddlepaddle 은 컴퓨터에 맞는 판을 먼저 (GPU: paddlepaddle-gpu 3.3.1 CUDA 11.8 판, CPU: paddlepaddle 3.3.1)
py -3.12 -m venv .venv_paddle
.\.venv_paddle\Scripts\python.exe -m pip install paddlepaddle==3.3.1
.\.venv_paddle\Scripts\python.exe -m pip install -r requirements_paddle.txt
```

- 확인한 환경: GTX 1070 Ti, torch 2.14.1+cu126, paddlepaddle-gpu 3.3.1 (cu118), paddleocr 3.7.0, OpenCV 4.10, numpy 2.3.5.
- 처음 `read` 를 실행할 때 PARSeq 모델 구조 코드(약 1MB)를 GitHub(`baudm/parseq`)에서 받으므로 인터넷이 필요합니다. 가중치는 `models/parseq_ft1.pt` 를 씁니다.
- 가상환경 폴더(`.venv_*`)는 이 폴더 안에 만들어도 되지만 GitHub 에 올리지 마세요.

## 4. 모델 파일 받기

`models/` 의 세 가지(1장 표)가 있어야 실행됩니다. 빠진 파일이 있으면 `boxes`·`read` 단계가 빠진 파일 이름을 알려 주고 멈춥니다.

- **이 폴더를 압축해서 직접 받은 경우:** `models/` 가 들어 있으므로 그대로 실행합니다.
- **GitHub 에서 받은 경우:** `models/` 가 없습니다. 팀에서 공유한 `models` 폴더를 받아 이 폴더 안에 넣습니다.
- **원래 저장소(실험 결과가 있는 컴퓨터)에서 다시 만들기:** 저장소 맨 위에서
  ```powershell
  New-Item -ItemType Directory -Force exp_cns\final\models\det_ft1
  Copy-Item exp_cns\results\det_ft1\infer\* exp_cns\final\models\det_ft1\
  Copy-Item exp_cns\results\parseq_ft1_sure\model\best.pt exp_cns\final\models\parseq_ft1.pt
  Copy-Item exp_cns\results\attr_cnn_v1\model\best.pt exp_cns\final\models\attr_cnn.pt
  ```

**GitHub 에 모델을 올리지 않는 이유**
1. **용량:** 세 파일 합계 약 194MB. GitHub 는 50MB 넘는 파일에 경고, 100MB 넘는 파일은 거부합니다. PARSeq(95MB)는 한도 바로 아래이고, 한 번 올리면 기록에 영원히 남아 저장소를 받을 때마다 같이 내려받게 됩니다. 올린다면 Git LFS 나 GitHub Releases 가 맞는 방법입니다.
2. **데이터 조건:** 세 모델 모두 AI Hub 사진으로 학습했습니다. 저장소가 공개라서, AI Hub 이용 조건에서 학습한 모델의 공개 배포가 허용되는지 확인하기 전까지는 사진처럼 올리지 않습니다.

## 5. 실행 방법

가상환경이 둘이라 세 번에 나눠 실행합니다. 어느 폴더에서 실행해도 됩니다.

```powershell
.\.venv_main\Scripts\python.exe   run_photo.py --step crop  --photo 사진.jpg   # 0단계
.\.venv_paddle\Scripts\python.exe run_photo.py --step boxes --photo 사진.jpg   # 1~2단계
.\.venv_main\Scripts\python.exe   run_photo.py --step read  --photo 사진.jpg   # 3~6단계
```

결과: `output\<사진 이름>\`
- `crops\pill_N.png`, `crops.csv`, `check_crop.png` — 찾은 알약과 자른 영역
- `boxes.json` — 알약·방향마다 글자 상자
- `result.csv`, `result.png` — 읽은 글자, 겉모습, 답, 표시 방식, 1~5등

동작 규칙:
- 이미 있는 결과 파일은 덮어쓰지 않고 멈춥니다. 다시 하려면 `output\<사진 이름>\` 을 지우거나 `--out` 으로 다른 폴더를 줍니다.
- 단계마다 CSV·JSON 을 **맨 마지막에** 저장해서 완료 표시로 씁니다 (`crops.csv`, `boxes.json`, `result.csv`). 중간에 실패하면 이 파일이 생기지 않으므로 같은 명령으로 다시 실행하면 됩니다. 파일은 임시 이름으로 다 쓴 뒤 이름을 바꿔서 반쯤 쓴 파일이 남지 않습니다.
- 알약을 하나도 못 찾으면 `crop` 단계가 다시 촬영 안내를 내고, `boxes`·`read` 단계는 모델을 불러오기 전에 같은 안내로 멈춥니다.
- 검출기를 만들지 못하면(Paddle 판이 인자를 모르는 등) 모델 폴더와 마지막 오류를 담아 바로 알립니다.

## 6. 원래 코드와 결과가 같은지 확인한 것 (2026-11-17)

| 확인 | 범위 | 결과 |
|---|---|---|
| 사진 1장 전체 흐름 (`photo_demo_cns.py` 와 비교, 이 폴더의 모델 복사본으로 다른 폴더에서 실행) | 사용자 사진 1장, 알약 2개 | 자른 사진·좌표·확인 그림 같음, 상자 같음(검출 시간 값만 다름), `result.csv` 모든 칸 같음 |
| 1·3단계 방향·읽기 (저장된 방향별 읽기 `reread_v1` 와 비교) | 개발용 17장 (12방향 12 + 긴 알약 5) | 각도 목록·방향별 글자·확신도·고른 방향 모두 같음 |
| 5·6단계 약 판별·표시 (저장된 pft1 답과 비교) | 개발 468장 + 검증 828장 | 답 1,296/1,296 같음. 맞음 개발 409 · 검증 648, 약 1개 표시 274(틀림 1) · 403(틀림 0), 다시 찍기 요청 11 · 34 — 저장된 집계와 같음 |
| 예외 처리 | 알약 없는 사진, 검출기 초기화 실패, 그림 저장 실패 | 안내 후 멈춤 / 원인 담은 오류 / CSV 안 생기고 다시 실행하면 정상 |

2단계 검출은 사용자 사진의 알약 2개(긴 알약 2방향씩)로만 확인했습니다. 원형 알약의 12방향 검출은 같은 함수를 쓰지만 이번에 따로 다시 돌려 보지는 않았습니다.

## 7. 이 방식의 성능과 주의점

- 개발용 468장 약 정확도 87.4%, 검증 828장 78.3% (저장된 읽기로 계산).
- 평가 세트(`pill_image_testset` 432장)는 82.9%. 단, 이 방식은 평가 결과를 본 뒤 기본 방식으로 정한 것이라 깨끗한 평가 숫자가 아닙니다. 사전 기록한 공식 결과는 pft1 + TF3 83.1% 입니다 (CLAUDE.md 10장 3-22, 3-26).
- 학습·평가 사진은 모두 알약 1개가 이미 잘린 AI Hub 사진입니다. 0단계(배경째 찍은 사진에서 알약 찾기)는 사용자 사진 1장으로 시험만 했고 성능은 재지 않았습니다.
- 각인과 겉모습이 같은 약(예: 아리셉트정 5mg ↔ 아리셉트에비스정 5mg)은 이 방식으로 가를 수 없어 후보로만 보여 줍니다.
- 겉모습 CNN 과 PARSeq 는 CUDA GPU 가 있으면 GPU, 없으면 CPU 로 돌아갑니다 (CPU 에서는 느림).

## 8. 아직 안 한 것

- 앱 화면(입력·결과 표시)과 하나의 환경에서 도는 실행 방법 (지금은 가상환경 두 개를 오가야 함)
- 앞뒤 사진 2장을 받는 흐름 (실험에서는 2장일 때 정확도가 크게 오름, 약 6~7종으로만 확인)
- 실제 실행 환경에서 시간 측정, 서로 닿은 알약 나누기
- 새 컴퓨터에 처음부터 설치해서 실행해 보기 (지금은 기존 가상환경으로만 확인)
