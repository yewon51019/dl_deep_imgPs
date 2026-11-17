"""pill_id — 알약 사진 1장 → 알약별 약 이름 후보 (정신과 약 실험 exp_cns 의 최종 방식 pft1 을 단계별로 정리한 코드).

단계 (파일 하나 = 단계 하나):
    s0_crop        알약 찾기·자르기 (사진 1장에서 알약 여러 개)
    s1_orient      방향 정하기 (긴 알약은 축 2방향, 그 밖은 30° 간격 12방향)
    s2_detect      글자 상자 찾기 (파인튜닝 PaddleOCR 검출기 det_ft1) — Paddle 전용 환경, torch 를 import 하지 않음
    s3_read        글자 읽기 (파인튜닝 PARSeq) + 방향 고르기 (확신도 최고)
    s4_appearance  겉모습 CNN (모양·제형·색 확률)
    s5_match       검색 DB 와 비교해 약 순위 (각인 점수 T + 0.02 × 겉모습 점수 A)
    s6_display     결과 표시 (확실 1개 / 후보 / 다시 찍기 요청)
공통: config (경로·설정값), imageio (한글 경로 읽기·저장, 크기 맞추기, 돌리기)
실행: src/run_photo.py (--step crop → boxes → read)
"""
