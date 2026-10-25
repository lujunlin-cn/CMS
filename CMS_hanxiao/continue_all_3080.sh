#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CMS_hanxiao
DATA=/root/autodl-tmp/cxr_first
MODEL=/root/autodl-tmp/CMS_hanxiao/pretrain_model
REMAINING="$ROOT/runs/research_v2_3080_remaining_01"
ALL="$ROOT/runs/research_v2_all_3080"
PY=/root/miniconda3/bin/python
LOG="$ALL/all_3080_supervisor.log"

mkdir -p "$ALL"
exec >>"$LOG" 2>&1
echo "$(date -Is) supervisor started"

# The current delegated queue plus its retry supervisor must finish before
# the imported first-nine jobs are scheduled on either 3080 Ti GPU.
while true; do
  done_count=$(find "$REMAINING" -mindepth 2 -maxdepth 2 -name done.ok -type f 2>/dev/null | wc -l)
  active_count=$(pgrep -f '/(run_verified_queue|research_train|train_unet_mil_advanced)\.py' | wc -l || true)
  retry_count=$(pgrep -f '^bash ./retry_unet_after_queue\.sh$' | wc -l || true)
  echo "$(date -Is) remaining_done=$done_count/12 active_train_processes=$active_count retry_processes=$retry_count"
  if [[ "$done_count" -ge 12 && "$active_count" -eq 0 && "$retry_count" -eq 0 ]]; then
    break
  fi
  sleep 120
done

echo "$(date -Is) copying completed remaining runs into unified root"
for job_dir in "$REMAINING"/*_s*; do
  [[ -d "$job_dir" ]] || continue
  job=$(basename "$job_dir")
  [[ -f "$job_dir/done.ok" ]] || { echo "incomplete $job"; exit 1; }
  rm -rf "$ALL/$job"
  cp -a "$job_dir" "$ALL/$job"
done

echo "$(date -Is) launching first-nine queue on 3080 Ti"
cd "$ROOT"
CMS_DATA_ROOT="$DATA" CMS_MODEL_ROOT="$MODEL" "$PY" -u run_verified_queue.py \
  --plan research_plans/confirmation_4090_first9.json \
  --root "$ROOT" --out-root "$ALL" --python "$PY" \
  --gpus 0 1 --max-parallel 2 --workers 12 --poll-seconds 30 --execute

echo "$(date -Is) first-nine queue complete; launching summary and Train+Valid inference"
"$PY" summarize_verified_runs.py --queue-root "$ALL" --out "${ALL}_summary"
CMS_DATA_ROOT="$DATA" CMS_MODEL_ROOT="$MODEL" "$PY" -u run_confirmed_inference.py \
  --queue-root "$ALL" --plan research_plans/confirmation_v2.json \
  --out "$ROOT/runs/full_inference_confirmed_3080" --root "$ROOT" --python "$PY" \
  --gpus 0 1 --max-parallel 2 --batch-size 64 --workers 8 --poll-seconds 60 --execute
echo "$(date -Is) all 3080 workflow complete"
