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
echo "$(date -Is) waiting for current N1 and U0 jobs"
while [[ ! -f "$OUT/N1_eff_half_s20261026/done.ok" || ! -f "$OUT/U0_unet_decoder_s20261024/done.ok" ]]; do
  sleep 30
done
echo "$(date -Is) current pair complete; pausing queue before cache build"

# Stop only the CMS remaining-queue process tree.  The unrelated jobs and the
# all-workflow supervisor are outside this tree and remain untouched.
queue_pid=$("$PY" - <<'PY'
import os
for line in os.popen("ps -eo pid=,args=").read().splitlines():
    fields=line.strip().split(None, 1)
    if len(fields) == 2 and '/run_verified_queue.py' in fields[1] and 'confirmation_3080_remaining.json' in fields[1]:
        if int(fields[0]) != os.getpid(): print(fields[0]); break
PY
)
if [[ -n "$queue_pid" ]]; then
  "$PY" - "$queue_pid" <<'PY'
import os, signal, sys, time
from collections import defaultdict, deque
root=int(sys.argv[1]); children=defaultdict(list)
for line in os.popen('ps -eo pid=,ppid=').read().splitlines():
    try: p,pp=map(int,line.split()); children[pp].append(p)
    except ValueError: pass
seen=set(); q=deque([root])
while q:
    p=q.popleft()
    if p in seen: continue
    seen.add(p); q.extend(children.get(p, []))
for p in sorted(seen, reverse=True):
    try: os.kill(p, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(5)
for p in sorted(seen, reverse=True):
    try: os.kill(p, signal.SIGKILL)
    except ProcessLookupError: pass
print('stopped queue tree', sorted(seen), flush=True)
PY
fi

echo "$(date -Is) building shared train-valid cache at $CACHE"
mkdir -p "$ROOT/runs/research_v2_all_3080"
env CMS_DATA_ROOT="$DATA" CMS_IMAGE_CACHE="$CACHE" \
  nice -n 10 ionice -c 3 "$PY" -u "$ROOT/build_image_cache.py"
test -s "$CACHE/meta.json"
echo "$(date -Is) cache ready; restarting remaining queue with cache"
cd "$ROOT"
nohup env CMS_DATA_ROOT="$DATA" CMS_MODEL_ROOT="$MODEL" CMS_IMAGE_CACHE="$CACHE" \
  "$PY" -u run_verified_queue.py --plan research_plans/confirmation_3080_remaining.json \
  --root "$ROOT" --out-root "$OUT" --python "$PY" \
  --gpus 0 1 --max-parallel 2 --workers 12 --poll-seconds 30 --execute \
  > "$OUT/launcher_cache.log" 2>&1 < /dev/null &
echo "$(date -Is) restarted queue pid=$!"
