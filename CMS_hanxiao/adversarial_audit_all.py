#!/usr/bin/env python3
"""Study-aware FGSM/PGD robustness audit for all seven frozen checkpoints.

The audit is diagnostic only. It reads official Train and Valid images, never
Test, and writes probabilities plus attack sensitivity summaries without
writing adversarial image files. Baseline models use their image-level logits
followed by the same max Study aggregation as the official inference path.
U-Net+MIL attacks the concatenated images of each Study against the aggregated
Study BCE objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from train_single_gpu import TensorTransform, build_model, read_records

ROOT = Path("/home/zhaona/CMS_hanxiao")
DATA_ROOT = ROOT / "cxr_first"
RUN_ROOT = ROOT / "runs" / "adversarial_audit"
LABEL_NAMES = [
    "enlarged_cardiomediastinum", "pneumothorax", "consolidation",
    "pneumonia", "edema", "cardiomegaly", "atelectasis",
    "lung_opacity", "pleural_effusion", "no_finding",
]
MODELS = (
    "densenet121", "efficientnetv2_s", "convnext_tiny", "convnext_small",
    "swin_small", "dinov2_base_reg4", "unet_mil_densenet121",
)
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


def metric_summary(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    aucs = []
    for i in range(targets.shape[1]):
        try:
            aucs.append(float(roc_auc_score(targets[:, i], probabilities[:, i])))
        except ValueError:
            aucs.append(float("nan"))
    return {
        "study_count": int(targets.shape[0]),
        "macro_auc": float(np.nanmean(aucs)),
        "macro_f1": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
        "auc_per_class": aucs,
    }


def bce_probability(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(target * np.log(p) + (1 - target) * np.log1p(-p)).mean())


def clamp_norm(value: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Clamp a normalized tensor to an epsilon pixel-space ball."""
    ref_pixel = reference * STD.to(reference.device) + MEAN.to(reference.device)
    pixel = value * STD.to(value.device) + MEAN.to(value.device)
    pixel = torch.maximum(torch.minimum(pixel, ref_pixel + epsilon), ref_pixel - epsilon)
    return (pixel.clamp(0.0, 1.0) - MEAN.to(value.device)) / STD.to(value.device)


def attack(model, images, targets, epsilon, step, steps, objective):
    with torch.no_grad():
        clean_logits = objective(images, targets, model)
    clean_prob = torch.sigmoid(clean_logits).detach()
    adv = images.detach().clone()
    # FGSM/PGD are intentionally run in float32 for reproducible gradients.
    for iteration in range(steps):
        adv.requires_grad_(True)
        logits = objective(adv, targets, model)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        gradient = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        magnitude = epsilon if steps == 1 else step
        adv = clamp_norm(adv + magnitude * gradient.sign(), images, epsilon).detach()
    with torch.no_grad():
        attack_logits = objective(adv, targets, model)
    return clean_prob, torch.sigmoid(attack_logits).detach()


class ImageDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = TensorTransform(224, train=False, hflip=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        item = self.records[index]
        with Image.open(item["path"]) as image:
            tensor = self.transform(image)
        return tensor, torch.from_numpy(item["target"]), item["study_id"], item["path"]


def aggregate_image_outputs(records, outputs):
    """Max-pool per-image probabilities into the competition Study unit."""
    studies = {}
    for i, study_id in enumerate(records["study_ids"]):
        item = studies.setdefault(study_id, {
            "subject_id": Path(records["paths"][i]).parts[-3],
            "target": records["targets"][i], "clean": [], "fgsm": [], "pgd": [],
        })
        for key in ("clean", "fgsm", "pgd"):
            item[key].append(outputs[key][i])
    rows = []
    for study_id in sorted(studies):
        item = studies[study_id]
        row = {"study_id": study_id, "subject_id": item["subject_id"], "image_count": len(item["clean"])}
        row["target"] = np.asarray(item["target"], dtype=np.float32)
        for key in ("clean", "fgsm", "pgd"):
            row[key] = np.max(np.stack(item[key]), axis=0)
        rows.append(row)
    return rows


def finish_rows(split, model_name, study_rows):
    targets = np.stack([r["target"] for r in study_rows])
    result = {"split": split, "model": model_name, "scope": "train+valid", "test_used": False, "metrics": {}}
    for key in ("clean", "fgsm", "pgd"):
        result["metrics"][key] = metric_summary(targets, np.stack([r[key] for r in study_rows]))
    csv_rows = []
    for r in study_rows:
        clean, fg, pg, target = r["clean"], r["fgsm"], r["pgd"], r["target"]
        row = {"split": split, "model": model_name, "subject_id": r["subject_id"], "study_id": r["study_id"], "image_count": r["image_count"]}
        for i in range(10):
            row[f"target_{i+1}"] = int(target[i])
            row[f"clean_prob_{i+1}"] = float(clean[i])
            row[f"fgsm_prob_{i+1}"] = float(fg[i])
            row[f"pgd_prob_{i+1}"] = float(pg[i])
        row["clean_loss"] = bce_probability(clean, target)
        row["fgsm_loss"] = bce_probability(fg, target)
        row["pgd_loss"] = bce_probability(pg, target)
        row["pgd_loss_increase"] = row["pgd_loss"] - row["clean_loss"]
        row["threshold_flips_fgsm"] = int(np.count_nonzero((clean >= .5) != (fg >= .5)))
        row["threshold_flips_pgd"] = int(np.count_nonzero((clean >= .5) != (pg >= .5)))
        row["pgd_prob_delta_l1"] = float(np.abs(pg - clean).mean())
        csv_rows.append(row)
    values = np.asarray([r["pgd_loss_increase"] for r in csv_rows], dtype=np.float64)
    cutoff = float(np.quantile(values, .99)) if len(values) else float("nan")
    for r in csv_rows:
        r["candidate_top1pct"] = bool(r["pgd_loss_increase"] >= cutoff)
    return result, csv_rows


def run_baseline(model_name, device, args):
    checkpoint = ROOT / "runs" / "compare" / model_name / "best.pt"
    model, _ = build_model(model_name)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    output = RUN_ROOT / model_name
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = {}, []
    for split in ("train", "valid"):
        records, _ = read_records(split, apply_exclusion=False)
        loader = DataLoader(ImageDataset(records), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        raw = {"study_ids": [], "paths": [], "targets": [], "clean": [], "fgsm": [], "pgd": []}
        for batch_index, (images, targets, study_ids, paths) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            objective = lambda x, y, m: m(x)
            # One-step FGSM and three-step PGD use the same epsilon ball.
            clean, fgsm = attack(model, images, targets, args.epsilon, args.epsilon, 1, objective)
            _, pgd = attack(model, images, targets, args.epsilon, args.step, args.pgd_steps, objective)
            raw["study_ids"].extend(study_ids); raw["paths"].extend(paths); raw["targets"].extend(targets.cpu().numpy())
            raw["clean"].extend(clean.cpu().numpy()); raw["fgsm"].extend(fgsm.cpu().numpy()); raw["pgd"].extend(pgd.cpu().numpy())
            if batch_index % args.progress_every == 0:
                print(json.dumps({"model": model_name, "split": split, "batches": batch_index, "images": len(raw["study_ids"])}, ensure_ascii=False), flush=True)
        study_rows = aggregate_image_outputs(raw, raw)
        summary, rows = finish_rows(split, model_name, study_rows)
        summaries[split] = summary; all_rows.extend(rows)
    write_outputs(output, model_name, summaries, all_rows, payload.get("epoch"))


def write_full_manifest(split, path):
    from collections import defaultdict
    with (DATA_ROOT / f"{split}.csv").open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    groups = defaultdict(list); labels = defaultdict(set)
    for row in source:
        groups[row["Study_id"]].append(row)
        for token in str(row.get("Predict_class", "")).split(";"):
            if token.strip(): labels[row["Study_id"]].add(int(token))
    rows = []
    for study_id in sorted(groups):
        group = groups[study_id]
        rows.append({"subject_id": group[0]["Subject_id"], "study_id": study_id,
                     "labels": sorted(labels[study_id]), "images": [str(Path(split) / r["Subject_id"] / r["Study_id"] / f"{r['Dicom_id']}.jpg") for r in group]})
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def aggregate_studies(image_logits, counts):
    outputs, start = [], 0
    for count in counts:
        outputs.append(image_logits[start:start + count].mean(dim=0)); start += count
    if start != image_logits.shape[0]: raise RuntimeError("Study/image count mismatch")
    return torch.stack(outputs)


def run_unet(device, args):
    from train_unet_mil_advanced import StudyDataset, collate_studies, aggregate_studies_extended
    from unet_mil_extensions import AdvancedUNetMIL
    checkpoint = ROOT / "runs" / "unet_mil" / "unet_mil_densenet121" / "best.pt"
    config = json.loads(checkpoint.with_name("config.json").read_text())
    model = AdvancedUNetMIL(config.get("model", "densenet121"), 10, float(config.get("topk_ratio", .2)),
                            config.get("spatial_pool", "topk"), float(config.get("pool_temperature", 1.0)),
                            int(config.get("attention_hidden", 128)))
    payload = torch.load(checkpoint, map_location="cpu"); model.load_state_dict(payload["model"], strict=True); model.to(device).eval()
    output = RUN_ROOT / "unet_mil_densenet121"; output.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = {}, []
    for split in ("train", "valid"):
        manifest = output / f"_{split}_full_studies.jsonl"
        write_full_manifest(split, manifest)
        dataset = StudyDataset(manifest, int(config.get("size", 224)), train=False)
        subject_by_study = {row["study_id"]: row["subject_id"] for row in dataset.rows}
        loader = DataLoader(dataset, batch_size=args.study_batch_size, shuffle=False, num_workers=min(args.workers, 4),
                            collate_fn=collate_studies, pin_memory=True, persistent_workers=args.workers > 0)
        study_rows = []
        for batch_index, (images, masks, targets, study_ids, counts) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True).float(); masks = masks.to(device, non_blocking=True).float(); targets = targets.to(device, non_blocking=True).float()
            def objective(x, y, m):
                image_logits, _, _ = m(x, masks)
                return aggregate_studies_extended(image_logits, counts, config.get("study_pool", "mean"), float(config.get("study_temperature", 1.0)), None)
            clean, fgsm = attack(model, images, targets, args.epsilon, args.epsilon, 1, objective)
            _, pgd = attack(model, images, targets, args.epsilon, args.step, args.pgd_steps, objective)
            for i, study_id in enumerate(study_ids):
                study_rows.append({"study_id": study_id, "subject_id": subject_by_study[study_id],
                                   "image_count": int(counts[i]), "target": targets[i].cpu().numpy(), "clean": clean[i].cpu().numpy(),
                                   "fgsm": fgsm[i].cpu().numpy(), "pgd": pgd[i].cpu().numpy()})
            if batch_index % args.progress_every == 0:
                print(json.dumps({"model": "unet_mil_densenet121", "split": split, "batches": batch_index, "studies": len(study_rows)}, ensure_ascii=False), flush=True)
        summary, rows = finish_rows(split, "unet_mil_densenet121", study_rows); summaries[split] = summary; all_rows.extend(rows)
    write_outputs(output, "unet_mil_densenet121", summaries, all_rows, payload.get("epoch"))


def write_outputs(output, model_name, summaries, rows, checkpoint_epoch):
    fieldnames = list(rows[0].keys()) if rows else []
    with (output / "study_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    for split in ("train", "valid"):
        candidates = [r for r in rows if r["split"] == split]
        candidates.sort(key=lambda r: r["pgd_loss_increase"], reverse=True)
        with (output / f"top_sensitive_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(candidates[: max(100, len(candidates) // 100)])
    summary = {"model": model_name, "checkpoint_epoch": checkpoint_epoch, "scope": "official train and valid; Study-level outputs", "test_used": False,
               "attack": {"fgsm_epsilon_pixel": 4 / 255, "pgd_epsilon_pixel": 4 / 255, "pgd_step_pixel": 2 / 255, "pgd_steps": 3},
               "splits": summaries, "interpretation": "High sensitivity is a robustness-review candidate; it is not proof of an adversarially modified source image."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
    (output / "done.ok").write_text("completed\n")


def aggregate_all():
    rows = []; summary_rows = []
    for model in MODELS:
        path = RUN_ROOT / model
        summary_path = path / "summary.json"
        if not summary_path.exists(): raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        for split, values in summary["splits"].items():
            for attack_name, metrics in values["metrics"].items():
                summary_rows.append({"model": model, "split": split, "attack": attack_name, **{k: v for k, v in metrics.items() if k != "auc_per_class"}, "auc_per_class": json.dumps(metrics["auc_per_class"])})
        with (path / "study_sensitivity.csv").open(encoding="utf-8") as handle: rows.extend(csv.DictReader(handle))
    with (RUN_ROOT / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys())); writer.writeheader(); writer.writerows(summary_rows)
    with (RUN_ROOT / "study_sensitivity_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    (RUN_ROOT / "aggregate.json").write_text(json.dumps({"models": list(MODELS), "splits": ["train", "valid"], "test_used": False, "rows": len(rows)}, ensure_ascii=False, indent=2) + "\n")
    (RUN_ROOT / "aggregate.done").write_text("completed\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ADV_BATCH_SIZE", "64")))
    parser.add_argument("--study-batch-size", type=int, default=int(os.environ.get("ADV_STUDY_BATCH_SIZE", "4")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("ADV_WORKERS", "8")))
    parser.add_argument("--epsilon", type=float, default=4 / 255)
    parser.add_argument("--step", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(); RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.aggregate: aggregate_all(); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    if args.model == "unet_mil_densenet121": run_unet(device, args)
    else: run_baseline(args.model, device, args)


if __name__ == "__main__": main()
