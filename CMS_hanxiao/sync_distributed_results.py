#!/usr/bin/env python3
"""Merge the delegated 3080 Ti runs into the 4090 queue and start analysis."""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko


ROOT = Path("/home/hajimi2025/CMS")
PLAN = ROOT / "CMS_hanxiao/research_plans/confirmation_v2.json"
REMOTE_3080 = "/root/autodl-tmp/CMS_hanxiao"
REMOTE_4090 = "/home/zhaona/CMS_hanxiao"
QUEUE_3080 = REMOTE_3080 + "/runs/research_v2_3080_remaining_01"
QUEUE_4090 = REMOTE_4090 + "/runs/research_v2_confirm_03"
OUT_4090 = REMOTE_4090 + "/runs/full_inference_confirmed"
PY_4090 = "/home/zhaona/anaconda3/envs/ljhtorch/bin/python"
REMAINING = ["N1_eff_half", "U0_unet_decoder", "U1_unet_global", "U2_unet_residual"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_3080() -> paramiko.SSHClient:
    client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("connect.westb.seetacloud.com", port=13946, username="root",
                   timeout=15, allow_agent=True, look_for_keys=True)
    return client


def connect_4090() -> paramiko.SSHClient:
    password = os.environ.get("CMS_4090_PASSWORD")
    if not password:
        raise RuntimeError("CMS_4090_PASSWORD is required for result synchronization")
    client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.50.1.238", port=22, username="zhaona", password=password,
                   timeout=15, allow_agent=False, look_for_keys=False)
    return client


def read_json(sftp: paramiko.SFTPClient, path: str) -> dict:
    with sftp.open(path, "r") as handle:
        return json.loads(handle.read().decode())


def wait_for_queues() -> tuple[dict, dict]:
    while True:
        c3080 = connect_3080(); s3080 = c3080.open_sftp()
        c4090 = connect_4090(); s4090 = c4090.open_sftp()
        try:
            d3080 = read_json(s3080, QUEUE_3080 + "/queue_state.json")
            d4090 = read_json(s4090, QUEUE_4090 + "/queue_state.json")
            first9_done = all(
                _remote_exists(s4090, f"{QUEUE_4090}/{arm}_s{seed}/done.ok")
                for arm in ["C0_cs_p0", "C1_cs_lr01", "N0_eff_imagenet"]
                for seed in (20261024, 20261025, 20261026)
            )
        finally:
            s3080.close(); c3080.close(); s4090.close(); c4090.close()
        print(now(), "4090_first9", first9_done, "3080", d3080.get("status"), flush=True)
        if d3080.get("status") == "failed":
            # The 3080 supervisor may relaunch failed entries after repairing
            # an environment/data issue. Keep polling until the retry queue
            # reaches a clean terminal state.
            print(now(), "3080 queue reported failed; waiting for retry supervisor", flush=True)
        if d3080.get("status") == "complete" and first9_done:
            return d3080, d4090
        time.sleep(120)


def _remote_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path); return True
    except IOError:
        return False


def copy_file(src: paramiko.SFTPClient, dst: paramiko.SFTPClient, source: str, target: str) -> None:
    data = io.BytesIO(); src.getfo(source, data); data.seek(0)
    parent = str(Path(target).parent)
    try: dst.stat(parent)
    except IOError: dst.mkdir(parent)
    dst.putfo(data, target)


def merge_and_launch(d3080: dict, d4090: dict) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    c3080 = connect_3080(); s3080 = c3080.open_sftp()
    c4090 = connect_4090(); s4090 = c4090.open_sftp()
    try:
        for job in plan["jobs"]:
            if job["id"] not in REMAINING: continue
            for seed in plan["protocol"]["seeds"]:
                name = f"{job['id']}_s{seed}"
                source = f"{QUEUE_3080}/{name}"
                target = f"{QUEUE_4090}/{name}"
                for filename in ("best.pt", "last.pt", "config.json", "metrics.jsonl", "train.log", "done.ok"):
                    try: s3080.stat(f"{source}/{filename}")
                    except IOError: continue
                    copy_file(s3080, s4090, f"{source}/{filename}", f"{target}/{filename}")
        jobs = {}
        for job in plan["jobs"]:
            for seed in plan["protocol"]["seeds"]:
                key = f"{job['id']}_s{seed}"
                if job["id"] in REMAINING:
                    source_job = d3080.get("jobs", {}).get(key, {})
                    jobs[key] = {"status": "complete_external", "source_host": "3080ti",
                                 "gpu": source_job.get("gpu"), "started_at": source_job.get("started_at"),
                                 "finished_at": source_job.get("finished_at") or now()}
                else:
                    old = d4090.get("jobs", {}).get(key, {})
                    if _remote_exists(s4090, f"{QUEUE_4090}/{key}/done.ok"):
                        old = {**old, "status": "complete_existing", "source_host": "4090"}
                    jobs[key] = old
        merged = {"name": plan["name"], "started_at": d4090.get("started_at"),
                  "status": "complete", "finished_at": now(), "updated_at": now(),
                  "distributed": True, "jobs": jobs}
        s4090.file(QUEUE_4090 + "/queue_state.json", "w").write(
            json.dumps(merged, ensure_ascii=False, indent=2).encode())
        s4090.file(QUEUE_4090 + "/distributed_merge.json", "w").write(
            json.dumps({"merged_at": now(), "source_queue": QUEUE_3080,
                        "remaining_arms": REMAINING}, ensure_ascii=False, indent=2).encode())
    finally:
        s3080.close(); c3080.close(); s4090.close(); c4090.close()
    c4090 = connect_4090()
    try:
        commands = [
            f"cd {REMOTE_4090} && {PY_4090} summarize_verified_runs.py --queue-root {QUEUE_4090} --out {QUEUE_4090}_summary",
            f"cd {REMOTE_4090} && nohup {PY_4090} -u run_confirmed_inference.py --queue-root {QUEUE_4090} --plan research_plans/confirmation_v2.json --out {OUT_4090} --root {REMOTE_4090} --python {PY_4090} --gpus 0 1 2 3 --max-parallel 2 --batch-size 64 --workers 8 --poll-seconds 60 --execute > {OUT_4090}_launcher.log 2>&1 < /dev/null & echo $!",
        ]
        for command in commands:
            _, out, err = c4090.exec_command(command)
            print(out.read().decode(), err.read().decode(), flush=True)
    finally:
        c4090.close()


if __name__ == "__main__":
    state3080, state4090 = wait_for_queues()
    merge_and_launch(state3080, state4090)
