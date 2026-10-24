#!/usr/bin/env bash
set -u
ROOT=/home/zhaona/CMS_hanxiao
RUN_ROOT="$ROOT/runs/compare"
mkdir -p "$RUN_ROOT"
status=0
for pair in "densenet121 efficientnetv2_s" "convnext_tiny convnext_small" "swin_small dinov2_base_reg4"; do
  set -- $pair
  echo "$(date -Is) starting comparison pair: $1 vs $2" | tee -a "$ROOT/compare_all.log"
  "$ROOT/run_compare_pair.sh" "$1" "$2" >> "$ROOT/compare_all.log" 2>&1 || status=1
done
/home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/summarize_compare.py" --root "$RUN_ROOT" >> "$ROOT/compare_all.log" 2>&1 || status=1
if [ "$status" -eq 0 ]; then echo completed > "$RUN_ROOT/compare.done"; else echo failed > "$RUN_ROOT/compare.failed"; fi
exit "$status"
