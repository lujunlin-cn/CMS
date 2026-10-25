#!/usr/bin/env python3
"""Aggregate per-arm Train/Valid predictions from the confirmed queue.

This produces research diagnostics only.  It never reads Test and never builds
an ensemble submission.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABEL_NAMES = [
    "enlarged_cardiomediastinum", "pneumothorax", "consolidation",
    "pneumonia", "edema", "cardiomegaly", "atelectasis",
    "lung_opacity", "pleural_effusion", "no_finding",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    selected = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    models: list[str] = []
    tables: dict[str, dict[str, list[dict[str, str]]]] = {"train": {}, "valid": {}}
    metrics_rows: list[dict] = []
    for arm in selected:
        name = arm["arm"]
        directory = root / name
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        models.append(name)
        for split, values in metrics.get("splits", {}).items():
            metrics_rows.append({
                "arm": name,
                "architecture": metrics.get("architecture", ""),
                "kind": metrics.get("kind", ""),
                "split": split,
                "study_count": values.get("study_count"),
                "macro_auc": values.get("macro_auc"),
                "macro_ap": values.get("macro_ap"),
                "macro_f1": values.get("macro_f1"),
                "micro_f1": values.get("micro_f1"),
                "checkpoint": metrics.get("checkpoint", ""),
            })
            tables[split][name] = read_rows(directory / f"{split}_study_predictions.csv")

    summary_fields = ["arm", "architecture", "kind", "split", "study_count",
                      "macro_auc", "macro_ap", "macro_f1", "micro_f1", "checkpoint"]
    with (root / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(sorted(metrics_rows, key=lambda r: (r["split"], -float(r["macro_auc"]))))

    for split in ("train", "valid"):
        joined: dict[str, dict[str, str]] = {}
        for model in models:
            for row in tables[split].get(model, []):
                item = joined.setdefault(row["study_id"], {
                    "split": split, "study_id": row["study_id"], "subject_id": row["subject_id"]})
                for cls in range(1, 11):
                    item[f"target_{cls}"] = row[f"target_{cls}"]
                    item[f"{model}_prob_{cls}"] = row[f"prob_{cls}"]
                    item[f"{model}_pred_{cls}"] = row[f"pred_{cls}"]
        disagreement_rows = []
        for item in joined.values():
            count = 0
            for cls in range(1, 11):
                preds = [int(item[f"{model}_pred_{cls}"]) for model in models if f"{model}_pred_{cls}" in item]
                value = int(len(set(preds)) > 1) if preds else 0
                item[f"disagreement_{cls}"] = value; count += value
            item["disagreement_count"] = count
            disagreement_rows.append(item)
        if disagreement_rows:
            fields = list(disagreement_rows[0])
            with (root / f"model_disagreement_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerows(sorted(disagreement_rows, key=lambda r: int(r["disagreement_count"]), reverse=True))

            class_rows = []
            for cls in range(1, 11):
                errors = []
                for model in models:
                    errors.extend(int(row[f"pred_{cls}"]) != int(row[f"target_{cls}"])
                                  for row in tables[split].get(model, []))
                class_rows.append({"split": split, "class_id": cls,
                                   "class_name": LABEL_NAMES[cls - 1],
                                   "model_count": len(models),
                                   "error_rate_across_model_studies": float(np.mean(errors)) if errors else float("nan")})
            with (root / f"class_error_summary_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(class_rows[0])); writer.writeheader(); writer.writerows(class_rows)
            with (root / f"hard_studies_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                hard = sorted(disagreement_rows, key=lambda r: int(r["disagreement_count"]), reverse=True)[:1000]
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(hard)
    (root / "aggregate.done").write_text("completed\n", encoding="utf-8")
    print(json.dumps({"arms": models, "metrics_rows": len(metrics_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
