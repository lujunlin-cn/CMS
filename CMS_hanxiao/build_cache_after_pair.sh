#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CMS_hanxiao
DATA=/root/autodl-tmp/cxr_first
MODEL="$ROOT/pretrain_model"
OUT="$ROOT/runs/research_v2_3080_remaining_01"
PY=/root/miniconda3/bin/python
CACHE=/dev/shm/cms_image_cache
LOG="$ROOT/runs/research_v2_all_3080/cache_supervisor.log"

exec >>"$LOG" 2>&1
echo "$(date -Is) waiting for N1 seed20261026"
while [[ ! -f "$OUT/N1_eff_half_s20261026/done.ok" ]]; do
  sleep 30
done
echo "$(date -Is) N1 complete; pausing queue scheduling before cache build"

# Stop only the CMS remaining-queue process tree.  The unrelated jobs and the
# all-workflow supervisor are outside this tree and remain untouched.
queue_pid=$("$PY" - <<'PY'
import os
for line in os.popen("ps -eo pid=,args=").read().splitlines():
    fields=line.strip().split(None, 1)
    if len(fields) == 2 and 'run_verified_queue.py' in fields[1] and 'confirmation_3080_remaining.json' in fields[1]:
        if int(fields[0]) != os.getpid(): print(fields[0]); break
PY
)
if [[ -n "$queue_pid" ]]; then
  kill -STOP "$queue_pid"
  echo "$(date -Is) stopped scheduler pid=$queue_pid; active children may finish"
fi

echo "$(date -Is) building shared train-valid cache at $CACHE"
mkdir -p "$ROOT/runs/research_v2_all_3080"
env CMS_DATA_ROOT="$DATA" CMS_IMAGE_CACHE="$CACHE" \
  nice -n 10 ionice -c 3 "$PY" -u "$ROOT/build_image_cache.py"
test -s "$CACHE/meta.json"
echo "$(date -Is) cache ready; waiting for already-started children to finish"
while pgrep -f '/(research_train|train_unet_mil_advanced)\.py.*research_v2_3080_remaining_01' >/dev/null; do
  sleep 30
done
if [[ -n "${queue_pid:-}" ]]; then
  kill "$queue_pid" 2>/dev/null || true
fi
date > "$ROOT/runs/research_v2_all_3080/cache_ready_paused.ok"
echo "$(date -Is) queue remains paused for research review"
