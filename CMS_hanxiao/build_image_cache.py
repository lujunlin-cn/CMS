#!/usr/bin/env python3
"""Build a shared train/valid RGB+mask cache for the CXR loaders.

Only the official train and valid CSV files are read.  The cache stores the
deterministic 224px letterboxed canvas, so workers avoid repeating JPEG decode
and resize while random train-time transforms remain in the Dataset classes.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image


DATA_ROOT = Path(os.environ.get("CMS_DATA_ROOT", "/home/zhaona/CMS_hanxiao/cxr_first"))
CACHE_ROOT = Path(os.environ.get("CMS_IMAGE_CACHE", "/dev/shm/cms_image_cache"))
SIZE = int(os.environ.get("CMS_CACHE_SIZE", "224"))


def collect_paths() -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for split in ("train", "valid"):
        with (DATA_ROOT / f"{split}.csv").open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rel = f"{split}/{row['Subject_id']}/{row['Study_id']}/{row['Dicom_id']}.jpg"
                if rel not in seen:
                    if not (DATA_ROOT / rel).exists():
                        raise FileNotFoundError(DATA_ROOT / rel)
                    seen.add(rel); paths.append(rel)
    return paths


def encode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        old_w, old_h = image.width, image.height
        scale = min(SIZE / old_w, SIZE / old_h)
        new_w = max(1, round(old_w * scale)); new_h = max(1, round(old_h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    valid = Image.new("L", (SIZE, SIZE), 0)
    offset_x = (SIZE - new_w) // 2; offset_y = (SIZE - new_h) // 2
    canvas.paste(image, (offset_x, offset_y))
    valid.paste(Image.new("L", (new_w, new_h), 255), (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8), np.asarray(valid, dtype=np.uint8)


def main() -> None:
    paths = collect_paths()
    CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_ROOT.parent / f".{CACHE_ROOT.name}.tmp.{os.getpid()}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    image_shape = (len(paths), SIZE, SIZE, 3); mask_shape = (len(paths), SIZE, SIZE)
    images = np.memmap(tmp / "images.uint8", mode="w+", dtype=np.uint8, shape=image_shape)
    masks = np.memmap(tmp / "masks.uint8", mode="w+", dtype=np.uint8, shape=mask_shape)
    started = time.time()
    for i, rel in enumerate(paths):
        images[i], masks[i] = encode(DATA_ROOT / rel)
        if (i + 1) % 1000 == 0:
            print(json.dumps({"processed": i + 1, "total": len(paths), "seconds": time.time() - started}), flush=True)
    images.flush(); masks.flush(); del images, masks
    (tmp / "index.json").write_text(json.dumps({p: i for i, p in enumerate(paths)}, ensure_ascii=False), encoding="utf-8")
    (tmp / "meta.json").write_text(json.dumps({
        "cache_version": 1, "data_root": str(DATA_ROOT), "splits": ["train", "valid"],
        "test_used": False, "size": SIZE, "count": len(paths),
        "image_shape": image_shape, "mask_shape": mask_shape,
        "created_at": time.time(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if CACHE_ROOT.exists(): shutil.rmtree(CACHE_ROOT)
    os.replace(tmp, CACHE_ROOT)
    print(json.dumps({"cache_root": str(CACHE_ROOT), "count": len(paths), "seconds": time.time() - started,
                      "image_bytes": int(np.prod(image_shape)), "mask_bytes": int(np.prod(mask_shape))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
