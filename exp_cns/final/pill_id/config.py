"""경로와 설정값을 한곳에 모음. 숫자는 모두 exp_cns 에서 개발·검증 사진으로 정한 값 그대로 (CLAUDE.md 10장 3-21, 3-26).

모든 경로는 이 최종본 폴더(exp_cns/final) 기준이라, 폴더째 옮기거나 압축해 보내도 그대로 실행된다.
모델 파일(models/)은 용량이 커서 GitHub 에 올리지 않으므로 따로 받아 넣는다 (README 참고).
"""
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]                # 최종본 폴더 (exp_cns/final)

# ---- 모델·DB 위치 ----
MODELS = FINAL / "models"
DET_MODEL = MODELS / "det_ft1"                              # 파인튜닝 검출기 (PP-OCRv6_medium_det 구조, inference.* 3개)
PARSEQ_WEIGHTS = MODELS / "parseq_ft1.pt"                   # 파인튜닝 PARSeq
ATTR_WEIGHTS = MODELS / "attr_cnn.pt"                       # 겉모습 CNN (ResNet18)
DB_CSV = FINAL / "db" / "search_db_cns.csv"                 # 검색 DB 1,211종
OUT_ROOT = FINAL / "output"                                 # run_photo.py 결과 (.gitignore)

# ---- 0단계 알약 찾기 ----
WORK_SIDE = 1000            # 찾기 계산 크기 (긴 변 px)
MIN_AREA, MAX_AREA = 0.002, 0.30   # 덩어리 넓이 / 사진 넓이
MAX_ELONG = 4.0             # 긴변/짧은변이 이보다 크면 가는 띠(빛 줄무늬 등)로 보고 버림
MIN_SOLID = 0.8             # 덩어리 넓이 / 볼록 껍질 넓이 (엉성한 모양 버림)
EDGE = 0.01                 # 사진 가장자리에서 이 비율 안쪽에 닿으면 버림 (잘린 알약·배경)
MARGIN = 0.15               # 자를 때 사방 여유 (긴 변 × 0.15)

# ---- 1단계 방향 ----
SIDE = 400                  # 읽기 사진 크기 (긴 변 400 정사각형)
LONG_RATIO = 1.3            # 긴변/짧은변 > 1.3 → 긴 알약
MIN_COVER = 0.80            # 알약 영역의 감싸는 상자가 사진을 이만큼 덮어야 축 정렬 (안전장치)
N_ANGLES = 12               # 긴 알약이 아니면 30° 간격 12방향

# ---- 2단계 검출 ----
UNCLIP = 2.0                # 상자 넓히기 비율 (기본 1.4 보다 넉넉하게)
MAX_BOX_AREA = 0.5          # 사진 넓이의 절반 넘는 상자 = 알약 전체를 잡은 것 → 버림
MIN_BOX_AREA = 40           # 너무 작은 상자 버림

# ---- 5단계 약 판별 ----
MIN_SIM = 0.3               # 각인 유사도가 이보다 낮으면 후보에서 뺌
LAM = 0.9                   # 다른 방향 읽기의 점수 가중치 (고른 방향 = 1)
W_ATTR = 0.02               # 겉모습 점수 가중치
MAX_LEN = 30                # 비교할 읽기 후보의 최대 글자 수

# ---- 6단계 표시 ----
SURE_TOP, SURE_GAP = 0.95, 0.1   # 1등 ≥ 0.95 이고 1등−2등 ≥ 0.1 이면 '확실' (약 1개)
CAP = 5                     # 후보가 이보다 많으면 다른 면·다시 찍기 요청


def check_models():
    """필요한 모델 파일이 다 있는지 확인. 빠진 것이 있으면 이름을 모두 알려 주고 멈춤."""
    need = [DET_MODEL / "inference.json", DET_MODEL / "inference.pdiparams", DET_MODEL / "inference.yml",
            PARSEQ_WEIGHTS, ATTR_WEIGHTS, DB_CSV]
    miss = [str(p.relative_to(FINAL)) for p in need if not p.exists()]
    if miss:
        raise SystemExit("모델·DB 파일이 없습니다 (README 의 '모델 파일 받기' 참고): " + ", ".join(miss))
