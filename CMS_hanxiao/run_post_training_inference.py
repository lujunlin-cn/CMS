#!/usr/bin/env python3
"""Wait for all seven best checkpoints, then run batched train+valid inference."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/zhaona/CMS_hanxiao")
PYTHON = "/home/zhaona/anaconda3/envs/ljhtorch/bin/python"
OUT = ROOT / "runs" / "full_inference"
JOBS = [
    ("densenet121", "compare"), ("efficientnetv2_s", "compare"),
    ("convnext_tiny", "compare"), ("convnext_small", "compare"),
    ("swin_small", "compare"), ("dinov2_base_reg4", "compare"),
    ("unet_mil_densenet121", "unet_mil"),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def ready(name, kind):
    d = ROOT / "runs" / kind / name
    return (d / "done.ok").exists() and (d / "best.pt").exists()


def save(state):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queue_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "queue.log"
    if (OUT / "aggregate.done").exists():
        return 0
    state = {"updated_at": now(), "status": "waiting", "jobs": {n: "pending" for n, _ in JOBS}}
    save(state)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} waiting for seven training checkpoints\n")
    while not all(ready(n, k) for n, k in JOBS):
        state["updated_at"] = now()
        state["status"] = "waiting"
        for n, k in JOBS:
            if ready(n, k): state["jobs"][n] = "ready"
        save(state)
        time.sleep(int(os.environ.get("INFER_WAIT_SECONDS", "60")))
    state["status"] = "running"; state["updated_at"] = now(); save(state)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} all seven checkpoints ready; starting batched inference\n")
    pending = list(JOBS)
    active = {}
    for gpu in range(4):
        if not pending: break
        name, _ = pending.pop(0)
        env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        logfile = OUT / f"{name}.log"
        command = [PYTHON, str(ROOT / "full_inference_analysis.py"), "--model", name, "--out", str(OUT)]
        handle = logfile.open("a", encoding="utf-8")
        handle.write(f"{now()} {' '.join(command)} gpu={gpu}\n"); handle.flush()
        active[gpu] = (name, subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT), handle)
        state["jobs"][name] = "running"
    save(state)
    while active or pending:
        for gpu, (name, proc, handle) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            handle.close()
            if code != 0:
                state["jobs"][name] = f"failed:{code}"
                with log.open("a", encoding="utf-8") as h: h.write(f"{now()} inference failed {name} code={code}\n")
                save(state)
                return 1
            state["jobs"][name] = "complete"
            del active[gpu]
            if pending:
                next_name, _ = pending.pop(0)
                env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                logfile = OUT / f"{next_name}.log"; handle = logfile.open("a", encoding="utf-8")
                command = [PYTHON, str(ROOT / "full_inference_analysis.py"), "--model", next_name, "--out", str(OUT)]
                handle.write(f"{now()} {' '.join(command)} gpu={gpu}\n"); handle.flush()
                active[gpu] = (next_name, subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT), handle)
                state["jobs"][next_name] = "running"
            save(state)
        time.sleep(15)
    subprocess.run([PYTHON, str(ROOT / "full_inference_analysis.py"), "--aggregate", "--out", str(OUT)], check=True)
    state["status"] = "complete"; state["updated_at"] = now(); save(state)
    with log.open("a", encoding="utf-8") as handle: handle.write(f"{now()} full train+valid inference and error analysis complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
