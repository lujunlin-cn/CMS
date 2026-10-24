#!/usr/bin/env python3
"""Export Study-level validation logits for post-hoc calibration.

This script is intentionally validation-only. It supports checkpoints produced by
train_unet_mil_advanced.py and writes logits/targets/Study IDs to NPZ. Test is
never opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_unet_mil_advanced import MANIFEST_ROOT, StudyDataset, collate_studies, aggregate_studies_extended
from unet_mil_model import MODEL_CONFIG
from unet_mil_extensions import AdvancedUNetMIL, StudyAttentionPool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=None)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt.get("args", {}))
    model_name = cfg.get("model", "densenet121")
    size = int(args.size or cfg.get("size", 224))
    spatial_pool = cfg.get("spatial_pool", "topk")
    study_pool = cfg.get("study_pool", "mean")
    study_temperature = float(cfg.get("study_temperature", 1.0))
    model = AdvancedUNetMIL(
        model_name, num_classes=10, topk_ratio=float(cfg.get("topk_ratio", 0.2)),
        spatial_pool=spatial_pool, pool_temperature=float(cfg.get("pool_temperature", 1.0)),
        attention_hidden=int(cfg.get("attention_hidden", 128)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    study_attention = None
    if study_pool == "attention":
        study_attention = StudyAttentionPool(10, int(cfg.get("study_attention_hidden", 32)))
        state = ckpt.get("study_attention")
        if state is None:
            raise RuntimeError("attention Study pool requested but checkpoint has no study_attention state")
        study_attention.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    if study_attention is not None:
        study_attention.to(device).eval()
    ds = StudyDataset(MANIFEST_ROOT / "valid_studies.jsonl", size, train=False)
    loader = DataLoader(ds, batch_size=int(cfg.get("study_batch_size", 4)), shuffle=False,
                        num_workers=args.workers, collate_fn=collate_studies, pin_memory=True,
                        persistent_workers=args.workers > 0)
    logits_all, targets_all, ids_all = [], [], []
    with torch.inference_mode():
        for images, masks, targets, study_ids, counts in loader:
            image_logits, _, _ = model(images.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            study_logits = aggregate_studies_extended(
                image_logits, counts, study_pool, study_temperature, study_attention
            )
            logits_all.append(study_logits.cpu().numpy())
            targets_all.append(targets.numpy())
            ids_all.extend(study_ids)
    logits = np.concatenate(logits_all, axis=0).astype(np.float32)
    targets = np.concatenate(targets_all, axis=0).astype(np.float32)
    if logits.shape != targets.shape or logits.shape[0] != len(ids_all):
        raise RuntimeError("exported arrays have inconsistent shapes")
    np.savez_compressed(args.out, logits=logits, targets=targets, study_ids=np.asarray(ids_all, dtype=str))
    print(json.dumps({"out": str(args.out), "study_count": int(logits.shape[0]),
                      "classes": int(logits.shape[1]), "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
