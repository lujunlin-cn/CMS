#!/usr/bin/env python3
"""Batched full inference and error analysis for all seven trained models.

Only the official train and valid splits are read.  Each worker writes
study-level probabilities (the competition scoring unit), per-class metrics,
and enough metadata to inspect model disagreement and class-specific errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from train_single_gpu import (
    CXRDataset, TensorTransform, build_model, read_records, set_seed,
)

ROOT = Path("/home/zhaona/CMS_hanxiao")
DATA_ROOT = ROOT / "cxr_first"
OUT_ROOT = ROOT / "runs" / "full_inference"
LABEL_NAMES = [
    "enlarged_cardiomediastinum", "pneumothorax", "consolidation",
    "pneumonia", "edema", "cardiomegaly", "atelectasis",
    "lung_opacity", "pleural_effusion", "no_finding",
]
BASELINE_MODELS = (
    "densenet121", "efficientnetv2_s", "convnext_tiny", "convnext_small",
    "swin_small", "dinov2_base_reg4",
)


def metric_summary(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    aucs, aps = [], []
    for index in range(targets.shape[1]):
        try:
            aucs.append(float(roc_auc_score(targets[:, index], probabilities[:, index])))
        except ValueError:
            aucs.append(float("nan"))
        try:
            aps.append(float(average_precision_score(targets[:, index], probabilities[:, index])))
        except ValueError:
            aps.append(float("nan"))
    pred = probabilities >= 0.5
    return {
        "study_count": int(targets.shape[0]),
        "macro_auc": float(np.nanmean(aucs)),
        "macro_ap": float(np.nanmean(aps)),
        "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, pred, average="micro", zero_division=0)),
        "auc_per_class": aucs,
        "ap_per_class": aps,
        "positive_count_per_class": targets.sum(axis=0).astype(int).tolist(),
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    fields = ["split", "subject_id", "study_id", "image_count"]
    fields += [f"target_{i}" for i in range(1, 11)]
    fields += [f"prob_{i}" for i in range(1, 11)]
    fields += [f"pred_{i}" for i in range(1, 11)]
    fields += ["error_count", "false_positive_count", "false_negative_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows_from_arrays(split: str, study_data: dict[str, dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows, targets, probabilities = [], [], []
    for study_id in sorted(study_data):
        item = study_data[study_id]
        target = np.asarray(item["target"], dtype=np.float32)
        probability = np.asarray(item["probability"], dtype=np.float32)
        pred = probability >= 0.5
        target_bool = target >= 0.5
        rows.append({
            "split": split,
            "subject_id": item["subject_id"],
            "study_id": study_id,
            "image_count": int(item.get("image_count", 1)),
            **{f"target_{i+1}": int(target[i]) for i in range(10)},
            **{f"prob_{i+1}": float(probability[i]) for i in range(10)},
            **{f"pred_{i+1}": int(pred[i]) for i in range(10)},
            "error_count": int(np.logical_xor(pred, target_bool).sum()),
            "false_positive_count": int(np.logical_and(pred, ~target_bool).sum()),
            "false_negative_count": int(np.logical_and(~pred, target_bool).sum()),
        })
        targets.append(target)
        probabilities.append(probability)
    return rows, np.stack(targets), np.stack(probabilities)


class ImageInferenceDataset(Dataset):
    def __init__(self, records: list[dict], size: int = 224, norm: str = "imagenet"):
        self.records = records
        self.transform = TensorTransform(size, train=False, hflip=False, norm=norm)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        from PIL import Image
        with Image.open(item["path"]) as image:
            tensor = self.transform(image)
        return tensor, torch.from_numpy(item["target"]), item["study_id"], item["path"]


def load_checkpoint_model(alias: str, checkpoint: Path, device: torch.device):
    model, load_info = build_model(alias)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload, load_info


def infer_baseline(alias: str, out: Path, device: torch.device, batch_size: int, workers: int,
                   checkpoint: Path | None = None, norm: str = "imagenet",
                   output_name: str | None = None) -> None:
    checkpoint = checkpoint or (ROOT / "runs" / "compare" / alias / "best.pt")
    model, payload, load_info = load_checkpoint_model(alias, checkpoint, device)
    model_out = out / (output_name or alias)
    model_out.mkdir(parents=True, exist_ok=True)
    all_metrics = {"model": output_name or alias, "architecture": alias, "kind": "baseline",
                   "checkpoint": str(checkpoint), "checkpoint_epoch": payload.get("epoch"),
                   "normalization": norm, "test_used": False, "splits": {}}
    for split in ("train", "valid"):
        # Full inference includes every official image, including the image
        # excluded from the baseline training manifest.
        records, _ = read_records(split, apply_exclusion=False)
        loader = DataLoader(
            ImageInferenceDataset(records, norm=norm), batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        )
        studies: dict[str, dict] = {}
        with torch.inference_mode():
            for images, targets, study_ids, paths in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    probabilities = torch.sigmoid(model(images.to(device, non_blocking=True)))
                probabilities = probabilities.float().cpu().numpy()
                targets_np = targets.numpy()
                for index, study_id in enumerate(study_ids):
                    item = studies.setdefault(study_id, {
                        "subject_id": Path(paths[index]).parts[-3],
                        "target": targets_np[index],
                        "probability": probabilities[index],
                        "image_count": 0,
                    })
                    item["probability"] = np.maximum(item["probability"], probabilities[index])
                    item["image_count"] += 1
        rows, targets_np, probabilities_np = rows_from_arrays(split, studies)
        write_predictions(model_out / f"{split}_study_predictions.csv", rows)
        metrics = metric_summary(targets_np, probabilities_np)
        all_metrics["splits"][split] = metrics
    all_metrics["load_info"] = load_info
    (model_out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n")
    (model_out / "done.ok").write_text("completed\n")
    print(json.dumps(all_metrics, ensure_ascii=False), flush=True)


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_all_image_study_manifest(split: str, path: Path) -> list[dict]:
    """Build an unfiltered Study manifest for full train/valid inference."""
    from collections import defaultdict
    with (DATA_ROOT / f"{split}.csv").open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    labels = defaultdict(set)
    groups = defaultdict(list)
    for row in source:
        for token in str(row.get("Predict_class", "")).split(";"):
            if token.strip():
                labels[row["Study_id"]].add(int(token))
        groups[row["Study_id"]].append(row)
    studies = []
    for study_id in sorted(groups):
        group = groups[study_id]
        studies.append({
            "subject_id": group[0]["Subject_id"],
            "study_id": study_id,
            "labels": sorted(labels[study_id]),
            "images": [str(Path(split) / row["Subject_id"] / row["Study_id"] / f"{row['Dicom_id']}.jpg") for row in group],
        })
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in studies), encoding="utf-8")
    return studies


def infer_unet(out: Path, device: torch.device, study_batch_size: int, workers: int,
               checkpoint: Path | None = None, config_path: Path | None = None,
               output_name: str | None = None) -> None:
    from train_unet_mil_advanced import StudyDataset, aggregate_studies_extended, collate_studies
    from unet_mil_extensions import AdvancedUNetMIL

    alias = "unet_mil_densenet121"
    checkpoint = checkpoint or (ROOT / "runs" / "unet_mil" / alias / "best.pt")
    config_path = config_path or checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = AdvancedUNetMIL(
        config.get("model", "densenet121"), num_classes=10,
        topk_ratio=float(config.get("topk_ratio", 0.2)),
        spatial_pool=config.get("spatial_pool", "topk"),
        pool_temperature=float(config.get("pool_temperature", 1.0)),
        attention_hidden=int(config.get("attention_hidden", 128)),
        fusion=config.get("fusion", "decoder"),
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model_out = out / (output_name or alias)
    model_out.mkdir(parents=True, exist_ok=True)
    all_metrics = {"model": output_name or alias, "architecture": alias, "kind": "unet_mil",
                   "checkpoint": str(checkpoint), "checkpoint_epoch": payload.get("epoch"),
                   "config": str(config_path), "test_used": False, "splits": {}}
    for split in ("train", "valid"):
        manifest = ROOT / "unet_mil_data" / f"{split}_studies.jsonl"
        if split == "train":
            manifest = out / "_all_train_studies.jsonl"
            write_all_image_study_manifest(split, manifest)
        rows_manifest = load_manifest(manifest)
        subject_by_study = {row["study_id"]: row["subject_id"] for row in rows_manifest}
        loader = DataLoader(
            StudyDataset(manifest, int(config.get("size", 224)), train=False),
            batch_size=study_batch_size, shuffle=False, num_workers=workers,
            collate_fn=collate_studies, pin_memory=True, persistent_workers=workers > 0,
        )
        studies: dict[str, dict] = {}
        with torch.inference_mode():
            for images, masks, targets, study_ids, counts in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    image_logits, _, _ = model(images.to(device, non_blocking=True), masks.to(device, non_blocking=True))
                    logits = aggregate_studies_extended(
                        image_logits, counts, config.get("study_pool", "mean"),
                        float(config.get("study_temperature", 1.0)), None,
                    )
                    probabilities = torch.sigmoid(logits).float().cpu().numpy()
                targets_np = targets.numpy()
                for index, study_id in enumerate(study_ids):
                    studies[study_id] = {
                        "subject_id": subject_by_study[study_id],
                        "target": targets_np[index],
                        "probability": probabilities[index],
                        "image_count": int(counts[index]),
                    }
        rows, targets_np, probabilities_np = rows_from_arrays(split, studies)
        write_predictions(model_out / f"{split}_study_predictions.csv", rows)
        all_metrics["splits"][split] = metric_summary(targets_np, probabilities_np)
    all_metrics["load_info"] = model.load_info
    (model_out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n")
    (model_out / "done.ok").write_text("completed\n")
    print(json.dumps(all_metrics, ensure_ascii=False), flush=True)


def aggregate(root: Path) -> None:
    model_names = list(BASELINE_MODELS) + ["unet_mil_densenet121"]
    all_metrics = []
    prediction_tables: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for model in model_names:
        model_dir = root / model
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        all_metrics.append(metrics)
        for split in ("train", "valid"):
            with (model_dir / f"{split}_study_predictions.csv").open(encoding="utf-8") as handle:
                prediction_tables[split][model] = list(csv.DictReader(handle))
    summary_rows = []
    for metrics in all_metrics:
        for split, values in metrics["splits"].items():
            summary_rows.append({"model": metrics["model"], "kind": metrics["kind"], "split": split, **{k: v for k, v in values.items() if k not in {"auc_per_class", "ap_per_class", "positive_count_per_class"}}, "auc_per_class": json.dumps(values["auc_per_class"]), "ap_per_class": json.dumps(values["ap_per_class"])})
    with (root / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sorted(summary_rows, key=lambda x: (x["split"], -float(x["macro_auc"]))))

    # Join the seven model outputs on Study_id and quantify disagreement.
    for split in ("train", "valid"):
        joined: dict[str, dict] = {}
        for model, rows in prediction_tables[split].items():
            for row in rows:
                item = joined.setdefault(row["study_id"], {"split": split, "study_id": row["study_id"], "subject_id": row["subject_id"]})
                for cls in range(1, 11):
                    item[f"target_{cls}"] = row[f"target_{cls}"]
                    item[f"{model}_prob_{cls}"] = row[f"prob_{cls}"]
                    item[f"{model}_pred_{cls}"] = row[f"pred_{cls}"]
        out_rows = []
        for study_id, item in joined.items():
            disagreements = 0
            for cls in range(1, 11):
                preds = [int(item[f"{model}_pred_{cls}"]) for model in model_names]
                disagreements += int(len(set(preds)) > 1)
                item[f"disagreement_{cls}"] = int(len(set(preds)) > 1)
            item["disagreement_count"] = disagreements
            out_rows.append(item)
        if out_rows:
            fields = list(out_rows[0].keys())
            with (root / f"model_disagreement_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sorted(out_rows, key=lambda x: int(x["disagreement_count"]), reverse=True))
        # Error frequency by class across all models, plus hard Study list.
        class_rows = []
        for cls in range(1, 11):
            error_counts = []
            for model in model_names:
                for row in prediction_tables[split][model]:
                    error_counts.append(int(row[f"pred_{cls}"]) != int(row[f"target_{cls}"]))
            class_rows.append({"split": split, "class_id": cls, "class_name": LABEL_NAMES[cls - 1], "model_count": len(model_names), "error_rate_across_model_studies": float(np.mean(error_counts)) if error_counts else float("nan")})
        with (root / f"class_error_summary_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(class_rows[0].keys())); writer.writeheader(); writer.writerows(class_rows)
        hard = sorted(joined.values(), key=lambda x: int(x["disagreement_count"]), reverse=True)[:1000]
        if hard:
            with (root / f"hard_studies_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(hard[0].keys())); writer.writeheader(); writer.writerows(hard)
    (root / "aggregate.done").write_text("completed\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(BASELINE_MODELS) + ["unet_mil_densenet121"])
    p.add_argument("--out", default=str(OUT_ROOT))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--output-name")
    p.add_argument("--norm", choices=("imagenet", "half"), default="imagenet")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("INFER_BATCH_SIZE", "64")))
    p.add_argument("--study-batch-size", type=int, default=int(os.environ.get("INFER_STUDY_BATCH_SIZE", "4")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("INFER_WORKERS", "8")))
    p.add_argument("--seed", type=int, default=20261024)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(out); return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full inference")
    set_seed(args.seed)
    device = torch.device("cuda:0")
    if args.model == "unet_mil_densenet121":
        infer_unet(out, device, args.study_batch_size, min(args.workers, 4),
                   checkpoint=args.checkpoint, config_path=args.config,
                   output_name=args.output_name)
    else:
        infer_baseline(args.model, out, device, args.batch_size, args.workers,
                       checkpoint=args.checkpoint, norm=args.norm,
                       output_name=args.output_name)


if __name__ == "__main__":
    main()
