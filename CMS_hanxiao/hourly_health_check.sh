#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
LOG="$ROOT/hourly_health.log"
now=$(date -Is)
{
  echo "=== $now ==="
  echo "host=$(hostname)"
  echo "-- disk --"
  df -hT / | tail -1
  echo "-- unzip --"
  if pgrep -af 'unzip -q -o cxr_first.zip' > /tmp/cxr_unzip_ps.$$ 2>/dev/null; then
    cat /tmp/cxr_unzip_ps.$$; du -sh "$ROOT/cxr_first" 2>/dev/null || true
    echo 'status=RUNNING'
  elif [ -d "$ROOT/cxr_first" ]; then
    echo 'status=NOT_RUNNING_DIRECTORY_PRESENT'
    du -sh "$ROOT/cxr_first" 2>/dev/null || true
  else
    echo 'status=NOT_STARTED_OR_MISSING'
  fi
  rm -f /tmp/cxr_unzip_ps.$$ 2>/dev/null || true
  echo "-- archive cleanup --"
  archive_count=$(find "$ROOT" -maxdepth 2 -type f \( -name 'cxr_first.zip' -o -name 'cxr_first.tar' -o -name 'cxr_first.tar.gz' \) -print 2>/dev/null | wc -l)
  if [ "$archive_count" -eq 0 ]; then
    echo 'cxr_first_archive=DELETED_OR_NOT_PRESENT'
  else
    echo 'cxr_first_archive=FOUND_WARNING'
    find "$ROOT" -maxdepth 2 -type f \( -name 'cxr_first.zip' -o -name 'cxr_first.tar' -o -name 'cxr_first.tar.gz' \) -ls 2>/dev/null || true
  fi
  echo "-- training --"
  if [ -f "$ROOT/train.pid" ]; then
    pid=$(cat "$ROOT/train.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd || true
      echo 'status=RUNNING'
    else
      echo "pid=$pid status=STOPPED"
    fi
  else
    echo 'status=NO_PID_FILE'
  fi
  echo "-- compare training --"
  if [ -f "$ROOT/compare.pid" ]; then
    compare_pid=$(cat "$ROOT/compare.pid" 2>/dev/null || true)
    if [ -n "$compare_pid" ] && kill -0 "$compare_pid" 2>/dev/null; then
      ps -p "$compare_pid" -o pid,etime,%cpu,%mem,stat,cmd || true
      echo 'compare_status=RUNNING'
    else
      echo "compare_pid=$compare_pid compare_status=STOPPED"
    fi
  else
    echo 'compare_status=NO_PID_FILE'
  fi
  echo "-- seven-model queue --"
  queue_pid=$(pgrep -f 'run_experiment_queue.py' | head -1 || true)
  if [ -n "$queue_pid" ]; then
    ps -p "$queue_pid" -o pid,etime,%cpu,%mem,stat,cmd || true
    echo 'queue_status=RUNNING'
  elif [ -f "$ROOT/runs/experiment_queue.done" ]; then
    echo 'queue_status=COMPLETE'
  elif [ -f "$ROOT/runs/experiment_queue.failed" ]; then
    echo 'queue_status=FAILED'
  else
    echo 'queue_status=NOT_RUNNING'
  fi
  echo "-- research_v1 queue --"
  research_pid=$(pgrep -f 'run_research_suite.py' | head -1 || true)
  if [ -n "$research_pid" ]; then
    ps -p "$research_pid" -o pid,etime,%cpu,%mem,stat,cmd || true
    echo 'research_status=RUNNING'
  elif [ -f "$ROOT/runs/research_v1/queue_state.json" ]; then
    "$ROOT/../anaconda3/envs/ljhtorch/bin/python" - "$ROOT/runs/research_v1/queue_state.json" <<'PY' 2>/dev/null || true
import json, sys
state=json.load(open(sys.argv[1], encoding='utf-8'))
print('research_status=' + state.get('status','unknown'))
print('research_jobs=' + ' '.join(f"{k}:{v.get('status')}" for k,v in state.get('jobs',{}).items()))
PY
  else
    echo 'research_status=NOT_STARTED'
  fi
  echo "-- research_v2 confirmation queue --"
  research_v2_pid=$(pgrep -f '[r]un_verified_queue.py.*research_v2_confirm_03' | head -1 || true)
  if [ -n "$research_v2_pid" ]; then
    ps -p "$research_v2_pid" -o pid,etime,%cpu,%mem,stat,cmd || true
    echo 'research_v2_status=RUNNING_OR_WAITING'
  elif [ -f "$ROOT/runs/research_v2_confirm_03/queue_state.json" ]; then
    "$ROOT/../anaconda3/envs/ljhtorch/bin/python" - "$ROOT/runs/research_v2_confirm_03/queue_state.json" <<'PY' 2>/dev/null || true
import json, sys
state=json.load(open(sys.argv[1], encoding='utf-8'))
print('research_v2_status=' + state.get('status','unknown'))
print('research_v2_jobs=' + ' '.join(f"{k}:{v.get('status')}" for k,v in state.get('jobs',{}).items()))
PY
  else
    echo 'research_v2_status=NOT_STARTED'
  fi
  if [ -f "$ROOT/runs/experiment_queue_state.json" ]; then
    "$ROOT/../anaconda3/envs/ljhtorch/bin/python" - "$ROOT/runs/experiment_queue_state.json" <<'PY' 2>/dev/null || true
import json, sys
state = json.load(open(sys.argv[1], encoding='utf-8'))
print('queue_jobs=' + ' '.join(f"{j['name']}:{j['status']}" for j in state.get('jobs', [])))
PY
  fi
  if [ -d "$ROOT/runs/compare" ]; then
    find "$ROOT/runs/compare" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null | sort || true
    for run in "$ROOT"/runs/compare/*; do
      [ -d "$run" ] || continue
      printf '%s: ' "$(basename "$run")"
      if [ -f "$run/done.ok" ]; then echo done; elif [ -f "$run/FAILED" ]; then echo failed; else echo pending_or_running; fi
    done
  fi
  echo "-- gpu 0-3 --"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | sed -n '1,4p' || true
  echo "-- recent training logs --"
  find "$ROOT" -maxdepth 3 -type f \( -name '*.log' -o -name '*.jsonl' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8
  echo
} >> "$LOG" 2>&1
