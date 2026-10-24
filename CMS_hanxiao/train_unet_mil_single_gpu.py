#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from unet_mil_model import MODEL_CONFIG, UNetMIL, freeze_encoder

ROOT = Path('/home/zhaona/CMS_hanxiao/cxr_first')
MANIFEST_ROOT = ROOT.parent / 'unet_mil_data'
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class LetterboxWithMask:
    def __init__(self, size, train=False, hflip=False):
        self.size = size
        self.train = train
        self.hflip = hflip

    def __call__(self, image):
        image = image.convert('RGB')
        old_w, old_h = image.width, image.height
        scale = min(self.size / old_w, self.size / old_h)
        new_w = max(1, round(old_w * scale))
        new_h = max(1, round(old_h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        valid = Image.new('L', (self.size, self.size), 0)
        offset_x = (self.size - new_w) // 2
        offset_y = (self.size - new_h) // 2
        canvas.paste(image, (offset_x, offset_y))
        valid.paste(Image.new('L', (new_w, new_h), 255), (offset_x, offset_y))
        if self.train and self.hflip and random.random() < 0.5:
            canvas = ImageOps.mirror(canvas)
            valid = ImageOps.mirror(valid)
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD
        mask = torch.from_numpy(np.asarray(valid, dtype=np.float32) / 255.0).unsqueeze(0)
        return tensor, mask


class StudyDataset(Dataset):
    def __init__(self, manifest_path, size, train=False, hflip=False):
        self.rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        self.transform = LetterboxWithMask(size, train=train, hflip=hflip)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        masks = []
        for relative_path in row['images']:
            with Image.open(ROOT / relative_path) as image:
                tensor, mask = self.transform(image)
            images.append(tensor)
            masks.append(mask)
        labels = torch.zeros(10, dtype=torch.float32)
        for label in row['labels']:
            labels[label - 1] = 1.0
        return torch.stack(images), torch.stack(masks), labels, row['study_id']


def collate_studies(batch):
    images, masks, labels, study_ids = zip(*batch)
    counts = [x.shape[0] for x in images]
    return torch.cat(images, dim=0), torch.cat(masks, dim=0), torch.stack(labels), study_ids, counts


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def aggregate_studies(image_logits, counts):
    outputs = []
    start = 0
    for count in counts:
        if count <= 0:
            raise ValueError('Study with zero images')
        outputs.append(image_logits[start:start + count].mean(dim=0))
        start += count
    if start != image_logits.shape[0]:
        raise RuntimeError('Study/image count mismatch')
    return torch.stack(outputs, dim=0)


def evaluate(model, loader, device):
    model.eval()
    all_prob = []
    all_target = []
    with torch.inference_mode():
        for images, masks, targets, _, counts in loader:
            image_logits, _, _ = model(images.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            study_logits = aggregate_studies(image_logits, counts)
            all_prob.append(torch.sigmoid(study_logits).cpu().numpy())
            all_target.append(targets.numpy())
    probabilities = np.concatenate(all_prob, axis=0)
    targets = np.concatenate(all_target, axis=0)
    auc_per_class = []
    for i in range(10):
        try:
            auc_per_class.append(float(roc_auc_score(targets[:, i], probabilities[:, i])))
        except ValueError:
            auc_per_class.append(float('nan'))
    return {
        'study_count': int(targets.shape[0]),
        'macro_auc': float(np.nanmean(auc_per_class)),
        'macro_f1': float(f1_score(targets, probabilities >= 0.5, average='macro', zero_division=0)),
        'auc_per_class': auc_per_class,
    }


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def run(args):
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this script is intended for one GPU')
    device = torch.device('cuda:0')
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    train_manifest = MANIFEST_ROOT / 'train_studies.jsonl'
    valid_manifest = MANIFEST_ROOT / 'valid_studies.jsonl'
    if not train_manifest.exists() or not valid_manifest.exists():
        raise FileNotFoundError('run prepare_unet_mil_data.py before training')
    train_loader = DataLoader(StudyDataset(train_manifest, args.size, train=True, hflip=args.hflip), batch_size=args.study_batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_studies, pin_memory=True, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(StudyDataset(valid_manifest, args.size, train=False), batch_size=args.study_batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_studies, pin_memory=True, persistent_workers=args.workers > 0)
    model = UNetMIL(args.model, num_classes=10, topk_ratio=args.topk_ratio).to(device)
    if args.freeze_encoder:
        freeze_encoder(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    config = vars(args).copy()
    config.update({'hostname': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES',''), 'device': str(device), 'train_manifest': str(train_manifest), 'valid_manifest': str(valid_manifest), 'test_used': False, 'load_info': model.load_info})
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + '\n')
    (output / 'metrics.jsonl').write_text('')
    best_auc = -float('inf')
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.time(); model.train(); total_loss = 0.0; steps = 0
            for images, masks, targets, _, counts in train_loader:
                images = images.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
                with context:
                    image_logits, _, _ = model(images, masks)
                    study_logits = aggregate_studies(image_logits, counts)
                    loss = loss_fn(study_logits, targets)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total_loss += float(loss.detach().cpu()); steps += 1
            scheduler.step(); validation = evaluate(model, valid_loader, device)
            result = {**validation, 'epoch': epoch, 'train_loss': total_loss / max(steps, 1), 'lr': scheduler.get_last_lr()[0], 'epoch_seconds': time.time() - started}
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:f.write(json.dumps(result, ensure_ascii=False) + '\n')
            checkpoint = {'model': cpu_state(model), 'epoch': epoch, 'val': result, 'args': vars(args), 'load_info': model.load_info}
            torch.save(checkpoint, output / 'last.pt')
            if result['macro_auc'] > best_auc:
                best_auc = result['macro_auc']; torch.save(checkpoint, output / 'best.pt')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        (output / 'done.ok').write_text('completed\n')
    except Exception as exc:
        (output / 'FAILED').write_text(f'{type(exc).__name__}: {exc}\n'); raise


def main():
    p = argparse.ArgumentParser(description='Single-GPU U-Net + Study MIL training')
    p.add_argument('--model', choices=sorted(MODEL_CONFIG), default='densenet121')
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--study-batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--topk-ratio', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--hflip', action='store_true')
    p.add_argument('--freeze-encoder', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    args = p.parse_args(); run(args)

if __name__ == '__main__': main()
