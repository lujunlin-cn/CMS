#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
OUT="$ROOT/runs/compare/dinov2_base_reg4"
rm -rf "$OUT"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/train_single_gpu.py" \
  --model dinov2_base_reg4 --out "$OUT" --epochs 5 --batch-size 32 --workers 12 --size 224 --seed 20261024
