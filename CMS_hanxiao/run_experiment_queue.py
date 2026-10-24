#!/usr/bin/env python3
"""Run the six backbone baselines plus one U-Net+MIL experiment.

The queue keeps at most one process per physical GPU and immediately assigns
the next experiment when a GPU becomes free.  It is deliberately a fresh-run
queue: an unfinished output directory is removed at startup, while a directory
with ``done.ok`` is kept so the queue can be safely restarted after a host
reboot.  Failed jobs are retried once and the remaining jobs continue.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("CMS_ROOT", "/home/zhaona/CMS_hanxiao"))
PYTHON = os.environ.get("CMS_PYTHON", "/home/zhaona/anaconda3/envs/ljhtorch/bin/python")
EPOCHS = int(os.environ.get("EPOCHS", "5"))
WORKERS = int(os.environ.get("WORKERS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
STUDY_BATCH_SIZE = int(os.environ.get("STUDY_BATCH_SIZE", "4"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "2"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "15"))


@dataclass
class Job:
    name: str
    kind: str
    model: str
    gpu: int | None = None
    attempts: int = 0
    status: str = "pending"
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None


JOBS = [
    Job("densenet121", "baseline", "densenet121"),
    Job("efficientnetv2_s", "baseline", "efficientnetv2_s"),
    Job("convnext_tiny", "baseline", "convnext_tiny"),
    Job("convnext_small", "baseline", "convnext_small"),
    Job("swin_small", "baseline", "swin_small"),
    Job("dinov2_base_reg4", "baseline", "dinov2_base_reg4"),
    # The extra experiment is the approved U-Net decoder + spatial MIL head.
    Job("unet_mil_densenet121", "unet_mil", "densenet121"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(path: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_dir(job: Job) -> Path:
    return ROOT / "runs" / ("compare" if job.kind == "baseline" else "unet_mil") / job.name


def command_for(job: Job, output: Path) -> list[str]:
    if job.kind == "baseline":
        return [
            PYTHON, str(ROOT / "train_single_gpu.py"),
            "--model", job.model, "--out", str(output), "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE), "--workers", str(WORKERS),
            "--size", "224", "--seed", "20261024",
        ]
    return [
        PYTHON, str(ROOT / "train_unet_mil_advanced.py"),
        "--model", job.model, "--out", str(output), "--epochs", str(EPOCHS),
        "--study-batch-size", str(STUDY_BATCH_SIZE), "--workers", str(WORKERS),
        "--size", "224", "--spatial-pool", "topk", "--topk-ratio", "0.2",
        "--study-pool", "mean", "--lr", "1e-4", "--weight-decay", "1e-4",
        "--seed", "20261024",
    ]


def save_state(path: Path, jobs: list[Job]) -> None:
    payload = {
        "updated_at": now(),
        "epochs": EPOCHS,
        "workers": WORKERS,
        "jobs": [asdict(job) for job in jobs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "experiment_queue.log"
    state_path = ROOT / "runs" / "experiment_queue_state.json"
    for job in JOBS:
        output = run_dir(job)
        if (output / "done.ok").exists():
            job.status = "complete"
            continue
        # This queue intentionally starts incomplete jobs from scratch.
        if output.exists():
            shutil.rmtree(output)
    save_state(state_path, JOBS)
    append_log(log_path, f"queue start: {len(JOBS)} jobs, {EPOCHS} epochs, GPUs 0-3")

    pending = [job for job in JOBS if job.status != "complete"]
    active: dict[int, tuple[Job, subprocess.Popen, object]] = {}
    next_gpu = 0
    while pending or active:
        # Fill all currently free GPU slots.  Each process sees its assigned
        # card as cuda:0 because CUDA_VISIBLE_DEVICES is set per process.
        for gpu in range(4):
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job.gpu = gpu
            job.attempts += 1
            job.status = "running"
            job.started_at = now()
            output = run_dir(job)
            output.mkdir(parents=True, exist_ok=True)
            log_file = ROOT / "runs" / f"{job.name}.attempt{job.attempts}.log"
            job.log = str(log_file)
            command = command_for(job, output)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            append_log(log_path, f"start {job.name} gpu={gpu} attempt={job.attempts}: {' '.join(command)}")
            handle = log_file.open("a", encoding="utf-8")
            handle.write(f"{now()} command={' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (job, process, handle)
        save_state(state_path, JOBS)

        finished = []
        for gpu, (job, process, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            finished.append(gpu)
            job.returncode = int(code)
            job.finished_at = now()
            output = run_dir(job)
            if code == 0 and (output / "done.ok").exists():
                job.status = "complete"
                append_log(log_path, f"complete {job.name} gpu={gpu}")
            elif job.attempts < MAX_ATTEMPTS:
                job.status = "retry_pending"
                pending.append(job)
                append_log(log_path, f"retry {job.name} after returncode={code}")
                if output.exists():
                    shutil.rmtree(output)
            else:
                job.status = "failed"
                append_log(log_path, f"FAILED {job.name} after {job.attempts} attempts returncode={code}")
        for gpu in finished:
            del active[gpu]
        save_state(state_path, JOBS)
        if active:
            time.sleep(POLL_SECONDS)

    # Generate the comparison table after all jobs, even if one failed.
    summary = ROOT / "summarize_compare.py"
    subprocess.run([PYTHON, str(summary), "--root", str(ROOT / "runs" / "compare")], check=False)
    complete = all(job.status == "complete" for job in JOBS)
    (ROOT / "runs" / ("experiment_queue.done" if complete else "experiment_queue.failed")).write_text(
        now() + "\n", encoding="utf-8"
    )
    append_log(log_path, "queue finished status=" + ("complete" if complete else "failed"))
    save_state(state_path, JOBS)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
