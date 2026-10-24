#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
PIDFILE="$ROOT/compare.pid"
RUN_ROOT="$ROOT/runs/compare"
if [ -f "$RUN_ROOT/compare.done" ]; then exit 0; fi
if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
  rm -f "$PIDFILE"
fi
if [ -f "$ROOT/train.pid" ]; then
  pid=$(cat "$ROOT/train.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
fi
mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$1==0 || $1==1 {gsub(/ /,"",$2); print $2}')
[ "${#used[@]}" -eq 2 ] || exit 0
for value in "${used[@]}"; do [ "$value" -lt 500 ] || exit 0; done
mkdir -p "$RUN_ROOT"
nohup "$ROOT/run_compare_all.sh" >> "$ROOT/compare_all.log" 2>&1 &
echo $! > "$PIDFILE"
