#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaona/CMS_hanxiao
GPU=${GPU:-0}
MODEL=${MODEL:-densenet121}
EPOCHS=${EPOCHS:-5}
POOL=${POOL:-topk}
STUDY_POOL=${STUDY_POOL:-mean}
OUT=${OUT:-$ROOT/runs/advanced/${MODEL}_${POOL}_${STUDY_POOL}}
EXTRA_ARGS=()
if [[ "${USE_POS_WEIGHT:-0}" == "1" ]]; then EXTRA_ARGS+=(--use-pos-weight); fi
if [[ "${FREEZE_ENCODER:-0}" == "1" ]]; then EXTRA_ARGS+=(--freeze-encoder); fi
if [[ "${NO_AMP:-0}" == "1" ]]; then EXTRA_ARGS+=(--no-amp); fi
exec env CUDA_VISIBLE_DEVICES="$GPU" /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/train_unet_mil_advanced.py" \
  --model "$MODEL" --out "$OUT" --epochs "$EPOCHS" \
  --study-batch-size "${STUDY_BATCH_SIZE:-4}" --workers "${WORKERS:-4}" --size "${SIZE:-224}" \
  --spatial-pool "$POOL" --topk-ratio "${TOPK_RATIO:-0.2}" --pool-temperature "${POOL_TEMPERATURE:-1.0}" \
  --study-pool "$STUDY_POOL" --study-temperature "${STUDY_TEMPERATURE:-1.0}" --study-attention-hidden "${STUDY_ATTENTION_HIDDEN:-32}" \
  --lr "${LR:-1e-4}" --encoder-lr-multiplier "${ENCODER_LR_MULTIPLIER:-1.0}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" --pos-weight-cap "${POS_WEIGHT_CAP:-50}" \
  --loss "${LOSS:-bce}" --asl-gamma-pos "${ASL_GAMMA_POS:-0}" --asl-gamma-neg "${ASL_GAMMA_NEG:-4}" --asl-clip "${ASL_CLIP:-0.05}" \
  --grad-clip "${GRAD_CLIP:-0}" --warmup-epochs "${WARMUP_EPOCHS:-0}" \
  --brightness "${BRIGHTNESS:-0}" --contrast "${CONTRAST:-0}" --noise "${NOISE:-0}" \
  "${EXTRA_ARGS[@]}"
