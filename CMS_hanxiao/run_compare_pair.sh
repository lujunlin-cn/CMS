#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
RUN_ROOT="$ROOT/runs/compare"
MODEL_A=${1:?usage: run_compare_pair.sh MODEL_GPU0 MODEL_GPU1}
MODEL_B=${2:?usage: run_compare_pair.sh MODEL_GPU0 MODEL_GPU1}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
WORKERS=${WORKERS:-8}
SEED=${SEED:-20261024}

if [ "${ALLOW_BUSY:-0}" != 1 ]; then
  mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1==0 || $1==1 {gsub(/ /,"",$2); print $2}')
  [ "${#used[@]}" -eq 2 ] || { echo 'GPU 0/1 query failed' >&2; exit 3; }
  for value in "${used[@]}"; do [ "$value" -lt 500 ] || { echo "GPU busy: ${used[*]} MiB" >&2; exit 3; }; done
fi

mkdir -p "$RUN_ROOT"
launch_one() {
  local model="$1" gpu="$2" out="$RUN_ROOT/$1"
  if [ -f "$out/done.ok" ]; then echo "$model already complete" >&2; LAUNCH_PID=; return 0; fi
  mkdir -p "$out"
  echo "$(date -Is) starting $model on GPU $gpu" >> "$out/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/train_single_gpu.py" \
    --model "$model" --out "$out" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" --seed "$SEED" > "$out/train.log" 2>&1 &
  LAUNCH_PID=$!
}
LAUNCH_PID=; launch_one "$MODEL_A" 0; PID_A=$LAUNCH_PID
LAUNCH_PID=; launch_one "$MODEL_B" 1; PID_B=$LAUNCH_PID
status=0
if [ -n "$PID_A" ]; then wait "$PID_A" || status=1; fi
if [ -n "$PID_B" ]; then wait "$PID_B" || status=1; fi
/home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/summarize_compare.py" --root "$RUN_ROOT" || status=1
exit "$status"
