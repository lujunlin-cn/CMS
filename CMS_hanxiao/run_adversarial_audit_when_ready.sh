#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
RUN="$ROOT/runs/densenet121_baseline"
PIDFILE="$ROOT/adv_audit.pid"
if [ -f "$RUN/robustness.ok" ]; then exit 0; fi
if [ ! -f "$RUN/complete.ok" ] || [ ! -f "$RUN/best.pt" ]; then exit 0; fi
if [ -f "$PIDFILE" ]; then pid=$(cat "$PIDFILE" 2>/dev/null || true); if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi; rm -f "$PIDFILE"; fi
mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1==0 || $1==1 {gsub(/ /,"",$2); print $2}')
[ "${#used[@]}" -eq 2 ] || exit 0
for value in "${used[@]}"; do [ "$value" -lt 500 ] || exit 0; done
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=0 /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/adversarial_audit.py" > "$ROOT/adversarial_audit.log" 2>&1 &
echo $! > "$PIDFILE"
