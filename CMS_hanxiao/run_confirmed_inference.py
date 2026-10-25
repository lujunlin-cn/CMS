#!/usr/bin/env python3
"""Select confirmed best checkpoints and run batched Train/Valid inference.

The queue is the only source of checkpoints.  Selection is by validation
Macro-AUC, then Macro-F1 only for an exact AUC tie.  Test is never opened.
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


def metric_key(row: dict) -> tuple[float, float]:
    return float(row.get("macro_auc", -1.0)), float(row.get("macro_f1", -1.0))


def select_runs(queue_root: Path, plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = []
    for job in plan["jobs"]:
        candidates = []
        for seed in plan["protocol"]["seeds"]:
            run = queue_root / f"{job['id']}_s{seed}"
            metrics_path = run / "metrics.jsonl"
            if not (run / "done.ok").exists() or not (run / "best.pt").exists() or not metrics_path.exists():
                continue
            rows = []
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            if rows:
                best = max(rows, key=metric_key)
                candidates.append((metric_key(best), int(seed), run, best))
        if not candidates:
            raise RuntimeError(f"no completed checkpoint for arm {job['id']}")
        score, seed, run, best = max(candidates, key=lambda x: x[0])
        selected.append({"arm": job["id"], "job": job, "seed": seed,
                         "run": str(run), "best": best, "score": list(score)})
    payload = {"created_at": now(), "queue_root": str(queue_root), "selected": selected,
               "metric_order": "valid_macro_auc_then_exact_macro_f1", "test_used": False}
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def idle_gpus(requested: list[int]) -> list[int]:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    result = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3: continue
        idx, mem, util = map(int, parts)
        if idx in requested and mem <= 500 and util <= 5: result.append(idx)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    state_path = args.queue_root / "queue_state.json"
    if not args.execute:
        print("dry-run: no inference launched"); return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "aggregate.done").exists(): return 0
    while True:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "complete": break
            if state.get("status") == "failed": raise RuntimeError("training queue failed")
        time.sleep(args.poll_seconds)
    subprocess.run([args.python, str(args.root / "summarize_verified_runs.py"),
                    "--queue-root", str(args.queue_root),
                    "--out", str(args.queue_root) + "_summary"], check=True)
    selection = select_runs(args.queue_root, args.plan, args.out)
    jobs = list(selection["selected"])
    log = (args.out / "queue.log").open("a", encoding="utf-8", buffering=1)
    log.write(f"{now()} selected {len(jobs)} arms; starting batched Train/Valid inference\n")
    script = args.root / "full_inference_analysis.py"
    pending = list(jobs); active: dict[int, tuple[dict, subprocess.Popen, object]] = {}
    while pending or active:
        for gpu in idle_gpus(args.gpus):
            if len(active) >= args.max_parallel or not pending: break
            arm = pending.pop(0); job = arm["job"]; run = Path(arm["run"])
            out_dir = args.out / arm["arm"]
            if (out_dir / "done.ok").exists(): continue
            if job["kind"] == "unet":
                model = "unet_mil_densenet121"
                extra = ["--study-batch-size", "4", "--config", str(run / "config.json")]
            else:
                model = job["model"]
                norm = job["args"].get("norm", "imagenet")
                extra = ["--norm", str(norm), "--batch-size", str(args.batch_size)]
            command = [args.python, str(script), "--model", model, "--out", str(args.out),
                       "--output-name", arm["arm"], "--checkpoint", str(run / "best.pt"),
                       "--workers", str(args.workers), *extra]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = (args.out / f"{arm['arm']}.log").open("a", encoding="utf-8")
            handle.write(f"{now()} {' '.join(command)} gpu={gpu}\n"); handle.flush()
            proc = subprocess.Popen(command, cwd=str(args.root), env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (arm, proc, handle)
        for gpu, (arm, proc, handle) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            handle.close(); del active[gpu]
            log.write(f"{now()} finish {arm['arm']} gpu={gpu} code={code}\n")
            if code != 0: raise RuntimeError(f"inference failed for {arm['arm']} with code {code}")
        if pending or active: time.sleep(args.poll_seconds)
    subprocess.run([args.python, str(args.root / "aggregate_confirmed_inference.py"),
                    "--root", str(args.out), "--selection", str(args.out / "selection.json")], check=True)
    (args.out / "queue_state.json").write_text(json.dumps({"status": "complete", "finished_at": now()}, indent=2) + "\n", encoding="utf-8")
    log.write(f"{now()} confirmed Train/Valid inference and diagnostics complete\n"); log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
