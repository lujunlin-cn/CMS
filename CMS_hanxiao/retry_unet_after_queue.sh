#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CMS_hanxiao
OUT="$ROOT/runs/research_v2_3080_remaining_01"
STATE="$OUT/queue_state.json"
PY=/root/miniconda3/bin/python

# Wait for the currently running queue to release both GPUs.  The first
# launch predated manifest generation and can contain failed U-Net entries;
# the retry launch below reuses completed checkpoints and reruns only entries
# without done.ok.
while true; do
  read -r status active < <("$PY" - "$STATE" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("status", "missing"), len(d.get("active_gpus", [])))
except Exception:
    print("missing", 0)
PY
  )
  if [[ "$active" == "0" && ( "$status" == "complete" || "$status" == "failed" ) ]]; then
    break
  fi
  sleep 60
done

cd "$ROOT"
exec env CMS_DATA_ROOT=/root/autodl-tmp/cxr_first \
  CMS_MODEL_ROOT=/root/autodl-tmp/CMS_hanxiao/pretrain_model \
  "$PY" -u run_verified_queue.py \
  --plan research_plans/confirmation_3080_remaining.json \
  --groups unet_structure \
  --root "$ROOT" \
  --out-root "$OUT" \
  --python "$PY" \
  --gpus 0 1 --max-parallel 2 --workers 8 --poll-seconds 30 --execute
