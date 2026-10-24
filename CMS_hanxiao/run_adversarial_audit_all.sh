#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaona/CMS_hanxiao
PYTHON=/home/zhaona/anaconda3/envs/ljhtorch/bin/python
OUT="$ROOT/runs/adversarial_audit"
mkdir -p "$OUT"
MODELS=(densenet121 efficientnetv2_s convnext_tiny convnext_small swin_small dinov2_base_reg4 unet_mil_densenet121)
if [[ -f "$OUT/aggregate.done" ]]; then exit 0; fi
run_group() {
  local start=$1
  local end=$2
  local -a pids=()
  local i model gpu log
  for ((i=start; i<end; i++)); do
    model=${MODELS[$i]}; gpu=$((i-start))
    log="$OUT/${model}.log"
    if [[ -f "$OUT/$model/done.ok" ]]; then continue; fi
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$PYTHON" "$ROOT/adversarial_audit_all.py" --model "$model" --batch-size 64 --study-batch-size 4 --workers 8 >"$log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
run_group 0 4
run_group 4 7
"$PYTHON" "$ROOT/adversarial_audit_all.py" --aggregate >"$OUT/aggregate.log" 2>&1
