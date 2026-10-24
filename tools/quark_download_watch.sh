#!/usr/bin/env bash
set -u
log=/home/hajimi2025/CMS/quark_downloads/progress.log
while true; do
  pid=$(pgrep -f 'python3 tools/quark_direct_download.py' | head -1 || true)
  bytes=$(find /home/hajimi2025/CMS/quark_downloads/cxr_first.zip.chunks -type f -name '*.part' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  avail=$(df -B1 /home/hajimi2025/CMS | awk 'NR==2 {print $4}')
  printf '%s pid=%s bytes=%s avail=%s\n' "$(date -Is)" "${pid:-stopped}" "$bytes" "$avail" >> "$log"
  [ -n "$pid" ] || exit 0
  sleep 3600
done
