#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
PIDFILE="$ROOT/train.pid"
LOG="$ROOT/train_baseline.log"
RUN="$ROOT/runs/densenet121_baseline"
# Do not interfere with unrelated jobs; launch only when both requested GPUs are idle.
if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    exit 0
  fi
  rm -f "$PIDFILE"
fi
if [ -f "$RUN/complete.ok" ]; then
  exit 0
fi
mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1==0 || $1==1 {gsub(/ /,"",$2); print $2}')
[ "${#used[@]}" -eq 2 ] || exit 0
for value in "${used[@]}"; do
  [ "$value" -lt 500 ] || exit 0
 done
cd "$ROOT"
printf '%s launch requested; GPUs 0,1 idle\n' "$(date -Is)" >> "$ROOT/launcher.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 /home/zhaona/anaconda3/envs/ljhtorch/bin/torchrun --standalone --nproc_per_node=2 \
  "$ROOT/train_baseline.py" --model densenet121 --epochs 5 --batch-size 32 --workers 8 \
  --out "$RUN" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
