#!/usr/bin/env python3
"""Run the preregistered research_v2 confirmation queue on independent GPUs.

The queue never stops unrelated GPU jobs, never overwrites a completed run,
and waits until a GPU is genuinely idle.  Each child process is bound to one
physical GPU and uses the existing single-GPU trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_gpu_rows() -> list[tuple[int, int, int]]:
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def idle_gpus(used: set[int], requested: list[int]) -> list[int]:
    # A free card has no meaningful allocation and no active utilization.
    # This conservative guard avoids sharing a card with an unrelated job.
    return [idx for idx, mem, util in read_gpu_rows()
            if idx in requested and idx not in used and mem <= 500 and util <= 5]


def command_for(root: Path, job: dict, seed: int, out: Path, protocol: dict, python: str) -> list[str]:
    args = job["args"]
    if job["kind"] == "backbone":
        cmd = [python, str(root / "research_train.py"), "--model", job["model"], "--out", str(out),
               "--epochs", str(protocol["epochs"]), "--batch-size", str(protocol["image_batch_size"]),
               "--workers", str(protocol["workers"]), "--size", str(protocol["size"]), "--seed", str(seed),
               "--weight-decay", str(protocol["weight_decay"]), "--deterministic"]
        for key in ("norm", "lr", "encoder_lr_multiplier", "train_unit", "loss", "eval_pool"):
            cmd += [f"--{key.replace('_', '-')}", str(args[key])]
        return cmd
    cmd = [python, str(root / "train_unet_mil_advanced.py"), "--model", job["model"], "--out", str(out),
           "--epochs", str(protocol["epochs"]), "--study-batch-size", str(protocol["study_batch_size"]),
           "--workers", str(protocol["workers"]), "--size", str(protocol["size"]), "--seed", str(seed),
           "--weight-decay", str(protocol["weight_decay"]), "--deterministic"]
    for key in ("fusion", "spatial_pool", "topk_ratio", "study_pool", "lr", "encoder_lr_multiplier", "freeze_epochs"):
        cmd += [f"--{key.replace('_', '-')}", str(args[key])]
    if args.get("freeze_encoder"):
        cmd.append("--freeze-encoder")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="research_plans/confirmation_v2.json")
    p.add_argument("--root", default="/home/zhaona/CMS_hanxiao")
    p.add_argument("--out-root", default="runs/research_v2_confirm_01")
    p.add_argument("--python", default="/home/zhaona/anaconda3/envs/ljhtorch/bin/python")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--execute", action="store_true", help="required to launch training")
    args = p.parse_args()
    root = Path(args.root); plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = plan_path if plan_path.exists() else root / plan_path
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [j for j in plan["jobs"] if not args.groups or j["group"] in args.groups]
    seeds = plan["protocol"]["seeds"]
    expanded = [(j, int(seed)) for j in jobs for seed in seeds]
    out_root = Path(args.out_root)
    if not out_root.is_absolute(): out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "queue_state.json"
    log_path = out_root / "queue.log"
    key_jobs = [f"{j['id']}_s{seed}" for j, seed in expanded]
    state = {"name": plan["name"], "started_at": now(), "status": "dry_run" if not args.execute else "waiting", "jobs": {k: {"status": "pending"} for k in key_jobs}}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    def emit(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh: fh.write(line + "\n")

    if not args.execute:
        emit(f"dry-run jobs={len(expanded)} groups={args.groups or 'all'}")
        return 0

    pending = list(expanded); active: dict[int, tuple[str, subprocess.Popen, object]] = {}
    while pending or active:
        for gpu in idle_gpus(set(active), args.gpus):
            if len(active) >= args.max_parallel or not pending: break
            job, seed = pending.pop(0); key = f"{job['id']}_s{seed}"; out = out_root / key
            if (out / "done.ok").exists():
                state["jobs"][key]["status"] = "complete_existing"; continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = command_for(root, job, seed, out, plan["protocol"], args.python)
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_handle = (out / "train.log").open("a", encoding="utf-8", buffering=1)
            log_handle.write(f"{now()} command={' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            active[gpu] = (key, proc, log_handle)
            state["jobs"][key] = {"status": "running", "gpu": gpu, "started_at": now(), "cmd": cmd}
            emit(f"start {key} gpu={gpu}")
        for gpu, (key, proc, handle) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            handle.close(); del active[gpu]
            ok = code == 0 and (out_root / key / "done.ok").exists()
            state["jobs"][key].update({"status": "complete" if ok else "failed", "returncode": code, "finished_at": now()})
            emit(f"finish {key} gpu={gpu} status={state['jobs'][key]['status']}")
        state["updated_at"] = now(); state["active_gpus"] = sorted(active)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        if pending or active: time.sleep(args.poll_seconds)
    state["status"] = "complete" if all(v["status"].startswith("complete") for v in state["jobs"].values()) else "failed"
    state["finished_at"] = now(); state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    emit("queue finished status=" + state["status"])
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
