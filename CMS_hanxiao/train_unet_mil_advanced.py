#!/usr/bin/env python3
"""Train optional U-Net+MIL improvements on one GPU.

This is an explicit experiment entry point.  Defaults match the approved
baseline (top-k spatial MIL, mean Study aggregation, unweighted BCE).  Every
extra head starts from random initialization and receives only official
Study-level labels.
"""
import argparse
import json
import math
import os
import random
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from train_unet_mil_single_gpu import (
    MANIFEST_ROOT, StudyDataset as BaseStudyDataset, collate_studies,
    MEAN, STD, set_seed, cpu_state,
)
from unet_mil_model import MODEL_CONFIG, freeze_encoder
from unet_mil_extensions import AdvancedUNetMIL, StudyAttentionPool


class StudyDataset(BaseStudyDataset):
    """Baseline preprocessing plus optional train-only tensor perturbations.

    Perturbations are intensity-only and are disabled for validation. They do
    not use test statistics or external masks.
    """
    def __init__(self, manifest_path, size, train=False, hflip=False,
                 brightness=0.0, contrast=0.0, noise=0.0):
        super().__init__(manifest_path, size, train=train, hflip=hflip)
        self.brightness = max(0.0, float(brightness))
        self.contrast = max(0.0, float(contrast))
        self.noise = max(0.0, float(noise))

    def __getitem__(self, index):
        images, masks, labels, study_id = super().__getitem__(index)
        if self.transform.train:
            if self.contrast:
                scale = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * self.contrast
                mean = images.mean(dim=(1, 2, 3), keepdim=True)
                images = (images - mean) * scale + mean
            if self.brightness:
                shift = (2.0 * torch.rand(1).item() - 1.0) * self.brightness
                images = images + shift
            if self.noise:
                images = images + torch.randn_like(images) * self.noise
        return images, masks, labels, study_id


def aggregate_studies_extended(image_logits, counts, mode="mean", temperature=1.0, attention_pool=None):
    if mode not in {"mean", "max", "logsumexp", "attention"}:
        raise ValueError(mode)
    if mode == "attention" and attention_pool is None:
        raise ValueError("attention Study pooling requires attention_pool")
    outputs = []
    start = 0
    tau = float(temperature)
    for count in counts:
        if count <= 0:
            raise ValueError("Study with zero images")
        chunk = image_logits[start:start + count]
        if mode == "mean":
            pooled = chunk.mean(dim=0)
        elif mode == "max":
            pooled = chunk.max(dim=0).values
        elif mode == "logsumexp":
            pooled = tau * (torch.logsumexp(chunk / tau, dim=0) - math.log(count))
        else:
            pooled = attention_pool(chunk)
        outputs.append(pooled)
        start += count
    if start != image_logits.shape[0]:
        raise RuntimeError("Study/image count mismatch")
    return torch.stack(outputs, dim=0)


class AsymmetricLoss(nn.Module):
    """Asymmetric focal-style BCE for multi-label imbalance.

    This is an optional train-only objective. It uses no extra labels or
    statistics; ``gamma_neg`` suppresses the many easy negatives and ``clip``
    prevents extreme negative probabilities from dominating.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos); self.gamma_neg = float(gamma_neg)
        self.clip = float(clip); self.eps = float(eps)

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = xs_neg.clamp(min=self.eps, max=1.0 - self.eps)
        loss = targets * torch.log(xs_pos) + (1.0 - targets) * torch.log(xs_neg)
        if self.gamma_pos or self.gamma_neg:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * (1.0 - pt).pow(gamma)
        return -loss.mean()


def train_pos_weight(manifest_path, num_classes=10, cap=50.0):
    positives = np.zeros(num_classes, dtype=np.float64)
    total = 0
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        for label in row["labels"]:
            positives[int(label) - 1] += 1
    if total <= 0:
        raise ValueError("empty training manifest")
    negatives = total - positives
    # A missing class receives a neutral weight and is still reported.
    weights = np.divide(negatives, np.maximum(positives, 1.0))
    weights = np.clip(weights, 1.0, float(cap))
    return weights.astype(np.float32), positives.astype(np.int64), total


def evaluate(model, loader, device, study_pool, study_temperature, attention_pool=None):
    model.eval()
    if attention_pool is not None:
        attention_pool.eval()
    all_prob = []; all_target = []
    with torch.inference_mode():
        for images, masks, targets, _, counts in loader:
            logits, _, _ = model(images.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            study_logits = aggregate_studies_extended(logits, counts, study_pool, study_temperature, attention_pool)
            all_prob.append(torch.sigmoid(study_logits).cpu().numpy())
            all_target.append(targets.numpy())
    probabilities = np.concatenate(all_prob, axis=0)
    targets = np.concatenate(all_target, axis=0)
    auc_per_class = []
    for i in range(targets.shape[1]):
        try: auc_per_class.append(float(roc_auc_score(targets[:, i], probabilities[:, i])))
        except ValueError: auc_per_class.append(float("nan"))
    tail_auc = auc_per_class[:3]
    head_auc = auc_per_class[-3:]
    return {
        "study_count": int(targets.shape[0]),
        "macro_auc": float(np.nanmean(auc_per_class)),
        "tail_macro_auc_classes_1_3": float(np.nanmean(tail_auc)),
        "head_macro_auc_classes_8_10": float(np.nanmean(head_auc)),
        "macro_f1": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
        "auc_per_class": auc_per_class,
    }


def build_optimizer(model, args, extra_modules=()):
    enc, other = [], []
    for name, p in model.named_parameters():
        # Keep temporarily frozen encoder parameters in the optimizer.  The
        # first-epoch-freeze ablation unfreezes them later; excluding them here
        # would silently turn it into a head-only run forever.
        (enc if name.startswith("encoder.") else other).append(p)
    for module in extra_modules:
        other.extend(list(module.parameters()))
    groups = [{"params": other, "lr": args.lr}]
    if enc:
        groups.append({"params": enc, "lr": args.lr * args.encoder_lr_multiplier})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def run(args):
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    train_manifest = MANIFEST_ROOT / "train_studies.jsonl"
    valid_manifest = MANIFEST_ROOT / "valid_studies.jsonl"
    pos_weight, positive_counts, train_count = train_pos_weight(train_manifest, cap=args.pos_weight_cap)
    train_ds = StudyDataset(train_manifest, args.size, train=True, hflip=args.hflip,
                            brightness=args.brightness, contrast=args.contrast, noise=args.noise)
    valid_ds = StudyDataset(valid_manifest, args.size, train=False)
    kwargs = dict(batch_size=args.study_batch_size, num_workers=args.workers,
                  collate_fn=collate_studies, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kwargs)
    if args.loss == "asl" and args.label_smoothing > 0:
        raise ValueError("label smoothing and ASL are separate ablations; choose one")
    model = AdvancedUNetMIL(args.model, num_classes=10, topk_ratio=args.topk_ratio,
                            spatial_pool=args.spatial_pool, pool_temperature=args.pool_temperature,
                            attention_hidden=args.attention_hidden,
                            fusion=args.fusion).to(device)
    study_attention = StudyAttentionPool(10, args.study_attention_hidden).to(device) if args.study_pool == "attention" else None
    if args.freeze_encoder: freeze_encoder(model)
    optimizer = build_optimizer(model, args, [study_attention] if study_attention is not None else [])
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    if args.loss == "asl":
        if args.use_pos_weight:
            raise ValueError("--use-pos-weight cannot be combined with --loss asl; run a separate ablation")
        loss_fn = AsymmetricLoss(args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    elif args.use_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    config = vars(args).copy(); config.update({
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": str(device), "train_manifest": str(train_manifest), "valid_manifest": str(valid_manifest),
        "test_used": False, "load_info": model.load_info, "train_study_count": train_count,
        "positive_counts": positive_counts.tolist(), "pos_weight": pos_weight.tolist(),
    })
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n")
    (output / "metrics.jsonl").write_text("")
    best_key = (-float("inf"), -float("inf"))
    try:
        for epoch in range(1, args.epochs + 1):
            # Linear warmup followed by cosine decay, applied per epoch.
            if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                scale = epoch / args.warmup_epochs
            else:
                progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            for group in optimizer.param_groups:
                base = args.lr if len(optimizer.param_groups) == 1 or group is optimizer.param_groups[0] else args.lr * args.encoder_lr_multiplier
                group["lr"] = base * max(scale, 1e-3)
            started = time.time(); model.train()
            if study_attention is not None:
                study_attention.train()
            # ``requires_grad=False`` does not freeze BatchNorm running statistics.
            # Keep a fully frozen encoder in eval mode while decoder/MIL heads train.
            if args.freeze_encoder and (args.freeze_epochs == 0 or epoch <= args.freeze_epochs):
                model.encoder.eval()
            elif args.freeze_encoder and epoch == args.freeze_epochs + 1:
                for parameter in model.encoder.parameters():
                    parameter.requires_grad = True
                model.encoder.train()
            total_loss = 0.0; steps = 0
            for images, masks, targets, _, counts in train_loader:
                images = images.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
                if args.label_smoothing:
                    targets_loss = targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                else: targets_loss = targets
                optimizer.zero_grad(set_to_none=True)
                with (torch.autocast(device_type="cuda", dtype=torch.float16) if not args.no_amp else nullcontext()):
                    logits, _, _ = model(images, masks)
                    study_logits = aggregate_studies_extended(logits, counts, args.study_pool, args.study_temperature, study_attention)
                    loss = loss_fn(study_logits, targets_loss)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_params = list(model.parameters()) + (list(study_attention.parameters()) if study_attention is not None else [])
                    torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
                scaler.step(optimizer); scaler.update(); total_loss += float(loss.detach().cpu()); steps += 1
            validation = evaluate(model, valid_loader, device, args.study_pool, args.study_temperature, study_attention)
            result = {**validation, "epoch": epoch, "train_loss": total_loss / max(steps, 1),
                      "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - started}
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(result) + "\n")
            checkpoint = {"model": cpu_state(model), "epoch": epoch, "val": result, "args": vars(args), "load_info": model.load_info}
            if study_attention is not None:
                checkpoint["study_attention"] = cpu_state(study_attention)
            torch.save(checkpoint, output / "last.pt")
            candidate_key = (float(result["macro_auc"]), float(result.get("macro_f1", -float("inf"))))
            if candidate_key > best_key:
                best_key = candidate_key
                torch.save(checkpoint, output / "best.pt")
            print(json.dumps(result), flush=True)
        (output / "done.ok").write_text("completed\n")
    except Exception as exc:
        (output / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n"); raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODEL_CONFIG), default="densenet121"); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--study-batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--size", type=int, default=224)
    p.add_argument("--spatial-pool", choices=AdvancedUNetMIL.POOLS, default="topk"); p.add_argument("--topk-ratio", type=float, default=0.2)
    p.add_argument("--fusion", choices=("decoder", "global", "residual", "encoder_mil"), default="decoder")
    p.add_argument("--pool-temperature", type=float, default=1.0); p.add_argument("--attention-hidden", type=int, default=128)
    p.add_argument("--study-pool", choices=("mean", "max", "logsumexp", "attention"), default="mean"); p.add_argument("--study-temperature", type=float, default=1.0)
    p.add_argument("--study-attention-hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--encoder-lr-multiplier", type=float, default=1.0)
    p.add_argument("--loss", choices=("bce", "asl"), default="bce")
    p.add_argument("--asl-gamma-pos", type=float, default=0.0); p.add_argument("--asl-gamma-neg", type=float, default=4.0); p.add_argument("--asl-clip", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--pos-weight-cap", type=float, default=50.0)
    p.add_argument("--use-pos-weight", action="store_true", help="use Train-only neg/pos weights for BCE; off by default")
    p.add_argument("--label-smoothing", type=float, default=0.0); p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0); p.add_argument("--brightness", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=0.0); p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261024); p.add_argument("--hflip", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="disable cuDNN benchmarking for stricter ablation reproducibility")
    p.add_argument("--freeze-encoder", action="store_true"); p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    run(p.parse_args())

if __name__ == "__main__": main()
