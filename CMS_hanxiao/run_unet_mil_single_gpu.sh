#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaona/CMS_hanxiao
GPU=${GPU:-0}
MODEL=${MODEL:-densenet121}
EPOCHS=${EPOCHS:-5}
OUT=${OUT:-$ROOT/runs/unet_mil/$MODEL}
CUDA_VISIBLE_DEVICES="$GPU" /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/train_unet_mil_single_gpu.py" \
  --model "$MODEL" --out "$OUT" --epochs "$EPOCHS" --study-batch-size "${STUDY_BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" --size "${SIZE:-224}" --topk-ratio "${TOPK_RATIO:-0.2}" \
  --lr "${LR:-1e-4}" --weight-decay "${WEIGHT_DECAY:-1e-4}"
