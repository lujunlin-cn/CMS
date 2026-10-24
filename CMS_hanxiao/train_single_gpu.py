#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import socket
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from safetensors.torch import load_file
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/home/zhaona/CMS_hanxiao/cxr_first')
MODEL_ROOT = Path('/home/zhaona/CMS_hanxiao/pretrain_model')
EXCLUSION_FILE = ROOT.parent / 'audit' / 'exclude_train_images.csv'
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
MODEL_CONFIG = {
    'densenet121': ('densenet121.tv_in1k', 'densenet121'),
    'efficientnetv2_s': ('tf_efficientnetv2_s.in21k_ft_in1k', 'efficientnetv2_s'),
    'convnext_tiny': ('convnext_tiny.fb_in22k_ft_in1k', 'convnext_tiny'),
    'convnext_small': ('convnext_small.fb_in22k_ft_in1k', 'convnext_small'),
    'swin_small': ('swin_small_patch4_window7_224.ms_in22k_ft_in1k', 'swin_small'),
    'dinov2_base_reg4': ('vit_base_patch14_reg4_dinov2.lvd142m', 'dinov2_base_reg4'),
}


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = ImageOps.contain(image, (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


class TensorTransform:
    def __init__(self, size, train=False, hflip=False):
        self.letterbox = Letterbox(size)
        self.train = train
        self.hflip = hflip

    def __call__(self, image):
        image = self.letterbox(image.convert('RGB'))
        if self.train and self.hflip and random.random() < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - MEAN) / STD


def parse_labels(value):
    labels = set()
    for token in str(value or '').split(';'):
        token = token.strip()
        if not token:
            continue
        label = int(token)
        if not 1 <= label <= 10:
            raise ValueError(f'Predict_class outside 1..10: {value!r}')
        labels.add(label)
    return labels


def load_excluded_keys():
    if not EXCLUSION_FILE.exists():
        return set()
    with EXCLUSION_FILE.open(newline='', encoding='utf-8') as f:
        return {
            (r['Subject_id'], r['Study_id'], r['Dicom_id'])
            for r in csv.DictReader(f)
            if r.get('action') == 'exclude_from_train_baseline_only'
        }


def read_records(split):
    with (ROOT / f'{split}.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    study_labels = defaultdict(set)
    for row in rows:
        study_labels[row['Study_id']].update(parse_labels(row.get('Predict_class', '')))
    excluded = load_excluded_keys() if split == 'train' else set()
    records = []
    seen = set()
    for row in rows:
        key = (row['Subject_id'], row['Study_id'], row['Dicom_id'])
        if key in seen or key in excluded:
            continue
        seen.add(key)
        target = np.zeros(10, dtype=np.float32)
        for label in study_labels[row['Study_id']]:
            target[label - 1] = 1.0
        records.append({
            'path': str(ROOT / split / row['Subject_id'] / row['Study_id'] / f"{row['Dicom_id']}.jpg"),
            'study_id': row['Study_id'],
            'target': target,
        })
    return records, study_labels


class CXRDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        with Image.open(item['path']) as image:
            tensor = self.transform(image)
        return tensor, torch.from_numpy(item['target']), item['study_id']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def build_model(alias):
    model_id, folder = MODEL_CONFIG[alias]
    checkpoint = MODEL_ROOT / folder / 'model.safetensors'
    model = timm.create_model(model_id, pretrained=False)
    state = load_file(str(checkpoint), device='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.reset_classifier(10)
    return model, {
        'model_id': model_id,
        'checkpoint': str(checkpoint),
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    }


def evaluate(model, loader, device):
    model.eval()
    probs_by_study = {}
    targets_by_study = {}
    with torch.inference_mode():
        for images, targets, study_ids in loader:
            probs = torch.sigmoid(model(images.to(device, non_blocking=True))).cpu().numpy()
            target_array = targets.numpy()
            for i, study_id in enumerate(study_ids):
                if study_id in probs_by_study:
                    probs_by_study[study_id] = np.maximum(probs_by_study[study_id], probs[i])
                else:
                    probs_by_study[study_id] = probs[i]
                    targets_by_study[study_id] = target_array[i]
    study_ids = sorted(probs_by_study)
    probabilities = np.stack([probs_by_study[x] for x in study_ids])
    targets = np.stack([targets_by_study[x] for x in study_ids])
    auc_per_class = []
    for i in range(10):
        try:
            auc_per_class.append(float(roc_auc_score(targets[:, i], probabilities[:, i])))
        except ValueError:
            auc_per_class.append(float('nan'))
    return {
        'study_count': len(study_ids),
        'macro_auc': float(np.nanmean(auc_per_class)),
        'macro_f1': float(f1_score(targets, probabilities >= 0.5, average='macro', zero_division=0)),
        'auc_per_class': auc_per_class,
    }


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def run(args):
    set_seed(args.seed)
    if args.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --cpu only for a smoke test')
        device = torch.device('cuda:0')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_records, train_studies = read_records('train')
    valid_records, valid_studies = read_records('valid')
    train_loader = DataLoader(
        CXRDataset(train_records, TensorTransform(args.size, train=True, hflip=args.hflip)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        CXRDataset(valid_records, TensorTransform(args.size, train=False)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0,
    )
    model, load_info = build_model(args.model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    amp_enabled = device.type == 'cuda' and not args.no_amp
    if hasattr(torch, 'amp'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    config = vars(args).copy()
    config.update({
        'hostname': socket.gethostname(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'device': str(device),
        'model_id': MODEL_CONFIG[args.model][0],
        'load_info': load_info,
        'train_images': len(train_records),
        'train_studies': len(train_studies),
        'valid_images': len(valid_records),
        'valid_studies': len(valid_studies),
        'test_used': False,
    })
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + '\n')
    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text('')
    best_auc = -float('inf')
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            model.train()
            total_loss = 0.0
            steps = 0
            for images, targets, _ in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
                with context:
                    loss = loss_fn(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu())
                steps += 1
            scheduler.step()
            validation = evaluate(model, valid_loader, device)
            result = {
                **validation,
                'epoch': epoch,
                'train_loss': total_loss / max(steps, 1),
                'lr': scheduler.get_last_lr()[0],
                'epoch_seconds': time.time() - epoch_started,
            }
            with metrics_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            torch.save({'model': cpu_state(model), 'epoch': epoch, 'val': result, 'args': vars(args)}, output / 'last.pt')
            if result['macro_auc'] > best_auc:
                best_auc = result['macro_auc']
                torch.save({'model': cpu_state(model), 'epoch': epoch, 'val': result, 'args': vars(args)}, output / 'best.pt')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        (output / 'done.ok').write_text('completed\n')
    except Exception as exc:
        (output / 'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


def main():
    parser = argparse.ArgumentParser(description='Single-GPU CXR transfer-learning experiment')
    parser.add_argument('--model', choices=sorted(MODEL_CONFIG), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
