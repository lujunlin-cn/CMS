#!/usr/bin/env python3
"""Wait for research_v2 confirmation and publish a CPU summary automatically."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT = Path('/home/zhaona/CMS_hanxiao')
QUEUE = ROOT / 'runs' / 'research_v2_confirm_03'
PYTHON = '/home/zhaona/anaconda3/envs/ljhtorch/bin/python'

def main():
    while True:
        state_path = QUEUE / 'queue_state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            status = state.get('status', 'unknown')
            if status in {'complete', 'failed'}:
                out = ROOT / 'runs' / 'research_v2_confirm_03_summary'
                subprocess.run([PYTHON, str(ROOT / 'summarize_verified_runs.py'), '--queue-root', str(QUEUE), '--out', str(out)], check=False)
                (out / 'postprocess_status.json').write_text(json.dumps({'queue_status': status, 'finished_at': state.get('finished_at')}, indent=2) + '\n')
                return 0 if status == 'complete' else 1
        time.sleep(60)

if __name__ == '__main__':
    raise SystemExit(main())
