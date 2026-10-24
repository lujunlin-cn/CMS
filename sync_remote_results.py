#!/usr/bin/env python3
"""Pull completed remote training/inference artifacts and publish review logs.

The password is supplied through REMOTE_PASSWORD at runtime and is never
written to the repository.  Checkpoints, pretrained weights, and source images
are deliberately excluded; metrics, CSV/JSON predictions, and logs are kept in
``analysis_artifacts/`` for review.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko


REMOTE_HOST = os.environ.get("REMOTE_HOST", "10.50.1.238")
REMOTE_USER = os.environ.get("REMOTE_USER", "zhaona")
REMOTE_PASSWORD = os.environ.get("REMOTE_PASSWORD")
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/home/zhaona/CMS_hanxiao")
LOCAL_ROOT = Path(os.environ.get("LOCAL_ROOT", "/home/hajimi2025/CMS"))
TARGET = LOCAL_ROOT / "analysis_artifacts" / "remote_runs"
POLL_SECONDS = int(os.environ.get("SYNC_POLL_SECONDS", "120"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_file(sftp, path: str) -> bool:
    try:
        mode = sftp.stat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)


def collect_tree(sftp, root: str) -> list[str]:
    result: list[str] = []
    try:
        entries = sftp.listdir_attr(root)
    except OSError:
        return result
    for entry in entries:
        path = posixpath.join(root, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            result.extend(collect_tree(sftp, path))
        elif stat.S_ISREG(entry.st_mode):
            result.append(path)
    return result


def wanted(remote_path: str) -> bool:
    name = posixpath.basename(remote_path)
    if name.endswith((".pt", ".pth", ".safetensors", ".zip", ".part")):
        return False
    if "/full_inference/" in remote_path:
        return True
    if "/adversarial_audit/" in remote_path:
        return True
    if "/runs/compare/" in remote_path or "/runs/unet_mil/" in remote_path:
        return name in {"metrics.jsonl", "config.json", "done.ok", "FAILED"}
    if "/runs/" in remote_path and (".attempt" in name or name in {"experiment_queue_state.json", "experiment_queue.done", "experiment_queue.failed"}):
        return True
    if remote_path.endswith(("experiment_queue.log", "experiment_queue.stdout.log")):
        return True
    return False


def connect():
    if not REMOTE_PASSWORD:
        raise RuntimeError("REMOTE_PASSWORD is required in the watcher environment")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, 22, username=REMOTE_USER, password=REMOTE_PASSWORD, timeout=20)
    return client


def all_done(sftp) -> bool:
    required = [
        "runs/compare/densenet121/done.ok",
        "runs/compare/efficientnetv2_s/done.ok",
        "runs/compare/convnext_tiny/done.ok",
        "runs/compare/convnext_small/done.ok",
        "runs/compare/swin_small/done.ok",
        "runs/compare/dinov2_base_reg4/done.ok",
        "runs/unet_mil/unet_mil_densenet121/done.ok",
        "runs/full_inference/aggregate.done",
        "runs/adversarial_audit/aggregate.done",
    ]
    return all(remote_file(sftp, posixpath.join(REMOTE_ROOT, item)) for item in required)


def pull(sftp) -> list[str]:
    candidates = []
    for tree in ("runs/full_inference", "runs/compare", "runs/unet_mil", "runs"):
        candidates.extend(collect_tree(sftp, posixpath.join(REMOTE_ROOT, tree)))
    for name in ("experiment_queue.log", "experiment_queue.stdout.log"):
        candidates.append(posixpath.join(REMOTE_ROOT, name))
    paths = sorted({p for p in candidates if wanted(p)})
    TARGET.mkdir(parents=True, exist_ok=True)
    pulled = []
    for remote_path in paths:
        relative = remote_path[len(REMOTE_ROOT):].lstrip("/")
        local_path = TARGET / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local_path))
        pulled.append(relative)
    return pulled


def write_manifest(files: list[str]) -> None:
    records = []
    for relative in sorted(files):
        path = TARGET / relative
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": str(path.relative_to(LOCAL_ROOT)), "size_bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "created_at": utc_now(),
        "remote_host": REMOTE_HOST,
        "remote_root": REMOTE_ROOT,
        "scope": "seven-model train+valid inference and training logs; no checkpoints or source images",
        "files": records,
    }
    (LOCAL_ROOT / "analysis_artifacts" / "SYNC_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOCAL_ROOT / "analysis_artifacts" / "README.md").write_text(
        "# CMS 训练与推理复核产物\n\n"
        "本目录由 `sync_remote_results.py` 从训练机自动同步。包含七个模型的训练指标、日志、"
        "train/valid 全量 Study 推理结果和错误分析文件。权重、原始影像和测试集未同步。\n",
        encoding="utf-8",
    )


def publish() -> None:
    subprocess.run(["git", "add", "analysis_artifacts", ".gitignore"], cwd=LOCAL_ROOT, check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=LOCAL_ROOT)
    if status.returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", "Publish completed training and inference artifacts"], cwd=LOCAL_ROOT, check=True)
    subprocess.run(["git", "push"], cwd=LOCAL_ROOT, check=True)


def main() -> int:
    while True:
        client = None
        try:
            client = connect()
            sftp = client.open_sftp()
            if all_done(sftp):
                files = pull(sftp)
                sftp.close(); client.close()
                write_manifest(files)
                publish()
                print(f"{utc_now()} synced and published {len(files)} files", flush=True)
                return 0
            print(f"{utc_now()} waiting for aggregate.done", flush=True)
            sftp.close(); client.close()
        except Exception as exc:
            if client is not None:
                client.close()
            print(f"{utc_now()} sync retry after {type(exc).__name__}: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
