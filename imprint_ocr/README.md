# pill_imprint.py 사용 방법

알약 사진에서 각인(글자)을 읽어 어떤 약인지 판별하고, 전처리 방식별로 점수를 매기는 실험 코드예요.
OCR 엔진은 **EasyOCR이 기본**이고, Tesseract는 비교용으로 남겨뒀어요.

---

## 0. 처음 한 번만: 설치

VS Code 아래쪽 터미널(PowerShell)에서 이 폴더로 이동한 뒤 실행해요.

```
pip install opencv-python numpy matplotlib easyocr
python pill_imprint.py check
```

`check` 결과에 `EasyOCR : 1.x.x` 처럼 버전이 나오면 준비 끝이에요.

- EasyOCR은 PyTorch를 같이 설치해서 용량이 커요(약 1~2GB). 시간이 좀 걸려요.
- 처음 실행할 때 글자 인식 모델(약 100MB)을 인터넷에서 자동으로 내려받아요.
- 랩실 GPU를 쓸 수 있으면 `GPU 사용 가능: True`로 나오고, 실행할 때 `--gpu`를 붙이면 훨씬 빨라요.

---

## 1. 촬영 방법

### 기본 촬영 (모든 알약)

| 항목 | 방법 | 이유 |
|---|---|---|
| 배경 | **검은 천이나 어두운 책상** | 알약을 자동으로 잘라낼 때 밝은 알약/어두운 배경으로 구분해요. 흰 배경은 실패해요 |
| 카메라 | 알약 **바로 위에서 수직으로** | 비스듬히 찍으면 글자가 찌그러져요 |
| 거리 | 각인 글자가 화면에서 또렷이 보일 만큼 가까이 | 글자가 작으면 OCR이 못 읽어요 |
| 알약 간격 | 서로 닿지 않게 | 붙어 있으면 한 덩어리로 잘려요 |

### 사광 촬영 (파인 각인 알약용: DK, YJ처럼 눌러 새긴 글자)

파인 각인은 잉크가 없어서, 정면 조명에서는 그림자가 생기지 않아 글자가 거의 안 보여요.
**옆에서 비스듬히 비추면** 홈 한쪽에 그림자가 생겨 글자가 진해져요.
방향을 바꿔 여러 장 찍고 합치면 글자 테두리 전체가 진해져요.
(금속 각인 연구에서 단일 조명 93~98% → 4방향 합성 99.6%로 향상)

1. **폰을 거치대에 고정**해요. 4장 모두 같은 위치에서 찍어야 해요. 조금 흔들리는 건 코드가 자동으로 보정해요.
2. **방 불을 끄고** 손전등(다른 폰의 플래시도 OK) 하나만 써요.
3. 손전등을 **알약 높이 가까이, 바닥에서 20~30도 정도 비스듬히** 들어요.
   ```
   손전등 ●
            \   ← 20~30도
             \
   ──────────[알약]────────── 바닥
   ```
4. 손전등 위치만 바꿔가며 **위 → 오른쪽 → 아래 → 왼쪽** 순서로 4장을 찍어요.
5. 파일 이름을 순서대로 붙이면 편해요. 예: `shots/1_up.jpg`, `2_right.jpg`, `3_down.jpg`, `4_left.jpg`

---

## 2. 실행 순서

### (선택) 2-1. 사광 사진 4장 합성 — `fuse`

```
python pill_imprint.py fuse --inputs shots/1_up.jpg shots/2_right.jpg shots/3_down.jpg shots/4_left.jpg --out fused
```

`fused/` 폴더에 3장이 생겨요.

| 파일 | 내용 | 용도 |
|---|---|---|
| `fused_min.png` | 4장에서 픽셀마다 가장 어두운 값을 모음 (모든 방향 그림자 합침) | **다음 단계 입력으로 추천** |
| `fused_diff.png` | 반대 방향 사진끼리의 밝기 차이 (각인 테두리만 하얗게) | 눈으로 확인용 |
| `fused_mean.png` | 단순 평균 | 비교용 |

### 2-2. 알약 1개씩 잘라내기 — `crop`

```
python pill_imprint.py crop --input images/multi_scene.jpg --out crops
```

- `crops/crop_00.png`, `crop_01.png` … 가 생겨요.
- `crops/crop_overview.png`를 열면 몇 번이 어떤 알약인지 번호가 표시돼 있어요.
- 사광 합성 사진이면 `--input fused/fused_min.png`로 넣으면 돼요.
- 알약 1개만 찍은 사진은 이 단계를 건너뛰고 바로 labels.csv에 넣어도 돼요.

### 2-3. 정답 적기 — `labels.csv`

메모장이나 VS Code로 열어서 한 줄에 사진 하나씩 적어요.

```
# 이미지경로,정답각인,각인유형(인쇄/음각)
crops/crop_00.png,alza18,인쇄
crops/crop_02.png,DK,음각
images/single_YJ.jpg,YJ,음각
```

- 정답은 대소문자·공백을 무시해요 (`alza 18` = `ALZA18`).
- 3번째 칸(유형)은 비워도 돼요. 비우면 코드가 자동으로 판정해요.

### 2-4. 판별 후보 목록 — `pill_db.csv`

```
약이름,각인,모양,색
콘서타OROS서방정 18mg,alza18,장방형,노랑
미확인 원형정(DK),DK,원형,하양
```

- 모양: `원형` 또는 `장방형`
- 색: 식약처 표기 그대로 (`하양`, `회색`, `노랑`, `주황`, `빨강`, `분홍`, `초록`, `파랑`, `보라` …)
- 지금은 5종이에요. **800종으로 늘릴 때 이 파일만 바꾸면** 코드는 그대로 써요.

### 2-5. 실험 실행 — `run`

```
python pill_imprint.py run
```

기본 설정은 엔진 EasyOCR, 전처리 A·D·AUTO, labels.csv, pill_db.csv예요. 자주 쓰는 옵션은 아래와 같아요.

| 옵션 | 뜻 | 예 |
|---|---|---|
| `--gpu` | GPU 사용 (훨씬 빠름) | `python pill_imprint.py run --gpu` |
| `--methods` | 비교할 전처리 고르기 | `--methods A B C D AUTO` (전부) |
| `--step` | 원형 알약 회전 간격(도). 크게 하면 빨라짐 | `--step 45` (8방향) |
| `--engine tesseract` | Tesseract로 비교 실행 | 윈도우는 `--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"` 도 필요 |
| `--out` | 결과 폴더 이름 | `--out results_easyocr_사광` |

전처리 종류:

| 기호 | 처리 | 1~3차 결과 |
|---|---|---|
| A | 배경 제거만 | 가장 좋았음 |
| B | A + 조명 평탄화 | 인쇄 각인에서 손해 |
| C | A + 반사광 제거 | 반사광이 거의 없어 판단 보류 |
| D | A + 반사광 제거 + 조명 평탄화 | 음각용 후보 |
| AUTO | 인쇄/음각 자동 구분 → 인쇄는 A, 음각은 D | 자동 구분 12/12 정확 |

---

## 3. 결과 보는 법

`results_easyocr/` 폴더에 생겨요. CSV는 엑셀로 열면 돼요.

| 파일 | 내용 |
|---|---|
| `chart_summary.png` | **먼저 볼 것.** 전처리 × 매칭별 약 판별 정확도와 처리 시간 그래프 |
| `summary.csv` | 위 그래프의 숫자 |
| `by_type.csv` | 인쇄/음각으로 나눠서 본 점수 → 어떤 유형에 어떤 전처리가 맞는지 |
| `errors.csv` | 틀린 이유별 개수 |
| `detail.csv` | 사진 한 장 한 장의 인식 결과 |

지표 뜻:

| 지표 | 뜻 |
|---|---|
| 글자완전일치% | 각인을 글자 하나까지 정확히 읽은 비율 |
| 문자정확도% | 글자 단위로 몇 % 맞았나 (예: 정답 ALZA27, 인식 ALZA27P → 83%) |
| 방향정답가정% | 돌려 읽은 결과 중 정답이 하나라도 있었나 (방향만 잘 골랐으면 맞았을까) |
| 약판별Top1% | 최종적으로 약을 맞혔나 ← **앱에서 가장 중요** |
| ①매칭없음 / ②편집거리 / ③편집거리+색모양 | 약을 고르는 방법 3가지 |

오류 원인:

| 원인 | 뜻 | 고칠 곳 |
|---|---|---|
| 빈 결과 | 아무 글자도 못 읽음 | 촬영(사광), 엔진 |
| 방향/선택 실패 | 돌려 읽은 결과엔 정답이 있었는데 다른 걸 고름 | 다수결 규칙 |
| 부분 인식 | 절반 이상 맞음 (l↔I, 0↔O 등) | 매칭으로 보완 가능 |
| 오인식 | 거의 다 틀림 | 엔진, 전처리 |

---

## 4. 추천 실험 순서

```
# ① 지금 사진으로 EasyOCR 기준선
python pill_imprint.py run --methods A B C D AUTO --out results_easyocr

# ② 파인 각인 알약을 사광으로 4장 찍어 합성 → 잘라내기 → labels.csv에 추가 → 실행
python pill_imprint.py fuse --inputs shots/*.jpg --out fused
python pill_imprint.py crop --input fused/fused_min.png --out crops_fused
python pill_imprint.py run --out results_easyocr_사광
```

①과 ②의 `by_type.csv`에서 **음각** 줄을 비교하면, 사광 촬영이 파인 각인을 얼마나 살리는지 숫자로 나와요.

---

## 5. 자주 나는 오류

| 메시지 | 해결 |
|---|---|
| `error: the following arguments are required` | 명령 뒤에 `crop`/`run` 같은 단어와 옵션을 같이 적어야 해요. ▶ 실행 버튼 말고 터미널에 직접 입력 |
| `No module named 'easyocr'` | `pip install easyocr` |
| `이미지 없음: ...` | labels.csv의 경로가 실제 파일 위치와 같은지 확인 |
| `[crop] 알약 0개 검출` | 배경이 알약보다 어두운지 확인 (흰 배경 X) |
| 너무 느림 | `--gpu` 사용, 또는 `--step 45`, `--methods A AUTO`로 줄이기 |
