#!/usr/bin/env python3
"""Fit per-class temperature scaling on exported Valid Study logits only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(logits, targets, bins=10):
    probs = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64)).numpy()
    y = np.asarray(targets, dtype=np.float64)
    nll = float(np.mean(np.maximum(-30.0, np.minimum(30.0, np.logaddexp(0, logits) - y * logits))))
    brier = float(np.mean((probs - y) ** 2))
    ece = []
    for c in range(y.shape[1]):
        p, t = probs[:, c], y[:, c]
        value = 0.0
        for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
            mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if mask.any():
                value += float(mask.mean()) * abs(float(p[mask].mean()) - float(t[mask].mean()))
        ece.append(value)
    return {"nll": nll, "brier": brier, "macro_ece": float(np.mean(ece)), "ece_per_class": ece}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="NPZ from export_valid_logits.py")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--min-temperature", type=float, default=0.05)
    p.add_argument("--max-temperature", type=float, default=20.0)
    args = p.parse_args()
    data = np.load(args.input, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError(f"expected matching [study,class] arrays, got {logits.shape} and {targets.shape}")
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    theta = torch.zeros(logits.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    lo, hi = float(np.log(args.min_temperature)), float(np.log(args.max_temperature))
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        temp = theta.clamp(lo, hi).exp()
        loss = F.binary_cross_entropy_with_logits(x / temp, y)
        loss.backward(); opt.step()
    temp = theta.detach().clamp(lo, hi).exp().numpy()
    before = metrics(logits, targets)
    after = metrics(logits / temp[None, :], targets)
    result = {
        "input": str(args.input), "input_sha256": sha256(args.input),
        "study_count": int(logits.shape[0]), "num_classes": int(logits.shape[1]),
        "temperatures": temp.tolist(), "before": before, "after": after,
        "test_used": False, "fit_scope": "official Valid Study logits only",
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
