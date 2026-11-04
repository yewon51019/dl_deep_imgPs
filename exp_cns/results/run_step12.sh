#!/bin/sh
# 1·2단계: 개발용·검증(trainval) 다시 읽기 (상자는 저장된 ft1_u2.0 그대로)
PY=.venv/Scripts/python.exe
W=exp_cns/results/parseq_ft1_sure/model/best.pt
C="exp_cns/ocr_cns.py --det paddle_ft1_u2.0 --align long --join line --no-dt"
for SP in dev trainval; do
  $PY $C --split $SP --parseq $W --suffix _line_pft1            > exp_cns/results/log_${SP}_pft1.txt 2>&1
  $PY $C --split $SP --pick orient --suffix _line               > exp_cns/results/log_${SP}_porient.txt 2>&1
  $PY $C --split $SP --parseq $W --pick orient --suffix _line_pft1 > exp_cns/results/log_${SP}_pft1_porient.txt 2>&1
done
# trainval 기준(사전학습 + 확신도)도 같은 코드로 (trainall 결과를 걸러 쓰는 것과 같은지 확인용)
$PY $C --split trainval --suffix _line > exp_cns/results/log_trainval_base.txt 2>&1
echo done
