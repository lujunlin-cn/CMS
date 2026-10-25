#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/CMS_hanxiao
DATA=/root/autodl-tmp/cxr_first
OUT="$ROOT/runs/research_v2_3080_remaining_01"
PY=/root/miniconda3/bin/python
exec env CMS_DATA_ROOT="$DATA" CMS_MODEL_ROOT="$ROOT/pretrain_model" CMS_IMAGE_CACHE=/dev/shm/cms_image_cache \
  "$PY" -u "$ROOT/run_verified_queue.py" --plan "$ROOT/research_plans/confirmation_3080_remaining.json" \
  --root "$ROOT" --out-root "$OUT" --python "$PY" --gpus 0 1 --max-parallel 2 --workers 12 --poll-seconds 30 --execute
