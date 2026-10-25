#!/usr/bin/env python3
"""Controlled low-risk CXR research ablations.

The script keeps the official Train/Valid split and preprocessing contract,
but exposes one factor at a time: Study versus image supervision, fixed
Study pooling, ASL, encoder learning-rate multiplier, and normalization.
Evaluation is always full-precision and Study-level; Test is never read.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, time
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

ROOT = Path(os.environ.get("CMS_DATA_ROOT", "/home/zhaona/CMS_hanxiao/cxr_first"))
MODEL_ROOT = Path(os.environ.get("CMS_MODEL_ROOT", "/home/zhaona/CMS_hanxiao/pretrain_model"))
EXCLUSION_FILE = ROOT.parent / "audit" / "exclude_train_images.csv"
MODEL_CONFIG = {
    "densenet121": ("densenet121.tv_in1k", "densenet121"),
    "efficientnetv2_s": ("tf_efficientnetv2_s.in21k_ft_in1k", "efficientnetv2_s"),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", "convnext_tiny"),
    "convnext_small": ("convnext_small.fb_in22k_ft_in1k", "convnext_small"),
    "swin_small": ("swin_small_patch4_window7_224.ms_in22k_ft_in1k", "swin_small"),
    "dinov2_base_reg4": ("vit_base_patch14_reg4_dinov2.lvd142m", "dinov2_base_reg4"),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

def parse_labels(value):
    out = set()
    for token in str(value or "").split(";"):
        token = token.strip()
        if token:
            label = int(token)
            if not 1 <= label <= 10: raise ValueError(value)
            out.add(label)
    return out

def excluded_keys():
    if not EXCLUSION_FILE.exists(): return set()
    with EXCLUSION_FILE.open(newline="", encoding="utf-8") as f:
        return {(r["Subject_id"], r["Study_id"], r["Dicom_id"])
                for r in csv.DictReader(f) if r.get("action") == "exclude_from_train_baseline_only"}

def read_rows(split):
    with (ROOT / f"{split}.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups[r["Study_id"]].append(r)
    ex = excluded_keys() if split == "train" else set()
    records, studies = [], []
    for sid, rs in groups.items():
        images = []
        labels = set()
        for r in rs:
            labels.update(parse_labels(r.get("Predict_class", "")))
            key = (r["Subject_id"], r["Study_id"], r["Dicom_id"])
            if key in ex: continue
            images.append(str(ROOT / split / r["Subject_id"] / r["Study_id"] / f"{r['Dicom_id']}.jpg"))
        if not images: continue
        target = np.zeros(10, dtype=np.float32)
        for label in labels: target[label - 1] = 1.0
        target_tuple = tuple(float(x) for x in target)
        for path in images:
            records.append({"path": path, "study_id": sid, "target": target})
        studies.append({"study_id": sid, "paths": images, "target": target})
    return records, studies

class Transform:
    def __init__(self, size, norm, train=False, hflip=False, brightness=0.0, contrast=0.0, noise=0.0):
        self.size, self.train, self.hflip = int(size), bool(train), bool(hflip)
        self.mean = torch.tensor(IMAGENET_MEAN if norm == "imagenet" else HALF_MEAN).view(3,1,1)
        self.std = torch.tensor(IMAGENET_STD if norm == "imagenet" else HALF_STD).view(3,1,1)
        self.brightness, self.contrast, self.noise = float(brightness), float(contrast), float(noise)
    def __call__(self, image):
        image = ImageOps.contain(image.convert("RGB"), (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (0,0,0))
        canvas.paste(image, ((self.size-image.width)//2, (self.size-image.height)//2))
        return self.from_uint8(np.asarray(canvas, dtype=np.uint8))

    def from_uint8(self, array):
        """Apply train-time tensor transforms to a cached RGB canvas."""
        if array.shape != (self.size, self.size, 3):
            raise ValueError(f"cached image shape {array.shape} does not match {(self.size, self.size, 3)}")
        x = torch.from_numpy(np.asarray(array, dtype=np.uint8).copy()).permute(2,0,1).float() / 255.0
        if self.train and self.hflip and random.random() < .5:
            x = torch.flip(x, dims=(2,))
        if self.train:
            if self.contrast: x = (x - x.mean(dim=(1,2), keepdim=True)) * (1 + random.uniform(-self.contrast,self.contrast)) + x.mean(dim=(1,2), keepdim=True)
            if self.brightness: x = x + random.uniform(-self.brightness,self.brightness)
            if self.noise: x = x + torch.randn_like(x) * self.noise
        return ((x.clamp(0,1) - self.mean) / self.std).float()

class SharedImageCache:
    """Lazy read-only mmap cache shared by DataLoader workers."""
    def __init__(self, root):
        self.root = Path(root)
        self._index = None
        self._images = None

    def _open(self):
        if self._index is not None:
            return
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self._index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self._images = np.memmap(self.root / "images.uint8", mode="r", dtype=np.uint8,
                                 shape=tuple(meta["image_shape"]))

    def get(self, relative_path):
        self._open()
        try:
            return self._images[int(self._index[relative_path])]
        except KeyError as exc:
            raise KeyError(f"image is absent from cache: {relative_path}") from exc

class ImageDataset(Dataset):
    def __init__(self, rows, transform, cache_root=None):
        self.rows, self.transform = rows, transform
        self.cache = SharedImageCache(cache_root) if cache_root else None
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        if self.cache is not None:
            rel = str(Path(r["path"]).relative_to(ROOT))
            x = self.transform.from_uint8(self.cache.get(rel))
        else:
            with Image.open(r["path"]) as im: x = self.transform(im)
        return x, torch.from_numpy(r["target"]), r["study_id"]

class StudyDataset(Dataset):
    def __init__(self, studies, transform, cache_root=None):
        self.studies, self.transform = studies, transform
        self.cache = SharedImageCache(cache_root) if cache_root else None
    def __len__(self): return len(self.studies)
    def __getitem__(self, i):
        r = self.studies[i]; xs=[]
        for p in r["paths"]:
            if self.cache is not None:
                rel = str(Path(p).relative_to(ROOT))
                xs.append(self.transform.from_uint8(self.cache.get(rel)))
            else:
                with Image.open(p) as im: xs.append(self.transform(im))
        return torch.stack(xs), torch.from_numpy(r["target"]), r["study_id"]

def collate_study(batch):
    xs, ys, sids = zip(*batch)
    return torch.cat(xs), torch.stack(ys), list(sids), [x.shape[0] for x in xs]

def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.set_float32_matmul_precision("high")
    if deterministic and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def build_model(alias):
    model_id, folder = MODEL_CONFIG[alias]
    kwargs = {"img_size": 224} if alias == "dinov2_base_reg4" else {}
    model = timm.create_model(model_id, pretrained=False, **kwargs)
    state = load_file(str(MODEL_ROOT / folder / "model.safetensors"), device="cpu")
    if alias == "dinov2_base_reg4" and "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
        import torch.nn.functional as F
        src = state["pos_embed"]; old_extra = src.shape[1] - int((src.shape[1])**0.5)**2
        new_extra = model.pos_embed.shape[1] - int((model.pos_embed.shape[1])**0.5)**2
        if old_extra != new_extra: raise RuntimeError("DINO extra token mismatch")
        og = int((src.shape[1]-old_extra)**0.5); ng = int((model.pos_embed.shape[1]-new_extra)**0.5)
        extra, grid = src[:, :old_extra], src[:, old_extra:].reshape(1,og,og,-1).permute(0,3,1,2)
        grid = F.interpolate(grid.float(), size=(ng,ng), mode="bicubic", align_corners=False)
        state["pos_embed"] = torch.cat([extra, grid.permute(0,2,3,1).reshape(1,ng*ng,-1)],1).to(src.dtype)
    missing, unexpected = model.load_state_dict(state, strict=False); model.reset_classifier(10)
    return model, {"model_id": model_id, "checkpoint": str(MODEL_ROOT/folder/"model.safetensors"), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}

class ASL(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=.05): super().__init__(); self.gn,self.gp,self.clip=gamma_neg,gamma_pos,clip
    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float()); neg = (1-p+self.clip).clamp(max=1.0) if self.clip else 1-p
        pos = p.clamp(1e-8,1-1e-8); neg = neg.clamp(1e-8,1-1e-8)
        pt = pos*targets + neg*(1-targets); g=self.gp*targets+self.gn*(1-targets)
        return (-(targets*torch.log(pos)+(1-targets)*torch.log(neg)) * (1-pt).pow(g)).mean()

def aggregate(logits, counts, mode):
    out=[]; start=0
    for n in counts:
        z=logits[start:start+n];
        if mode == "max": q=z.max(0).values
        elif mode == "mean_prob": q=torch.logit(torch.sigmoid(z).mean(0).clamp(1e-6,1-1e-6))
        elif mode == "mean_logit": q=z.mean(0)
        elif mode == "lse": q=torch.logsumexp(z,0)-math.log(n)
        else: raise ValueError(mode)
        out.append(q); start += n
    if start != logits.shape[0]: raise RuntimeError("count mismatch")
    return torch.stack(out)

def evaluate(model, loader, device, mode, study_input=False):
    model.eval(); by_sid={}
    with torch.inference_mode():
        for batch in loader:
            if study_input:
                images, targets, sids, counts = batch
            else:
                images, targets, sids = batch; counts=None
            z = model(images.to(device, non_blocking=True)).float()
            if study_input: z = aggregate(z, counts, mode)
            else:
                # Preserve all images and apply the requested Study aggregation.
                for i,sid in enumerate(sids):
                    if sid not in by_sid: by_sid[sid]=[[],targets[i].numpy()]
                    by_sid[sid][0].append(z[i].cpu())
                continue
            for i,sid in enumerate(sids): by_sid[sid]=([z[i].cpu()],targets[i].numpy())
    if not study_input:
        probs=[]; t=[]
        for sid in sorted(by_sid):
            zz=torch.stack(by_sid[sid][0]); q=aggregate(zz,[len(zz)],mode)[0]
            probs.append(torch.sigmoid(q).numpy()); t.append(by_sid[sid][1])
    else:
        probs=[torch.sigmoid(by_sid[s][0][0]).numpy() for s in sorted(by_sid)]; t=[by_sid[s][1] for s in sorted(by_sid)]
    probs=np.asarray(probs); t=np.asarray(t); auc=[]
    for c in range(10): auc.append(float(roc_auc_score(t[:,c],probs[:,c])))
    return {"study_count": int(len(t)), "macro_auc": float(np.mean(auc)), "macro_f1": float(f1_score(t,probs>=.5,average="macro",zero_division=0)), "auc_per_class": auc}

def run(args):
    set_seed(args.seed, args.deterministic); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train_rows, train_studies=read_rows("train"); valid_rows, valid_studies=read_rows("valid")
    tr=Transform(args.size,args.norm,True,args.hflip,args.brightness,args.contrast,args.noise); va=Transform(args.size,args.norm,False)
    cache_root = os.environ.get("CMS_IMAGE_CACHE") or None
    if cache_root and not (Path(cache_root) / "meta.json").exists():
        raise FileNotFoundError(f"CMS_IMAGE_CACHE is missing meta.json: {cache_root}")
    study_mode=args.train_unit == "study"
    if study_mode:
        train_ds=StudyDataset(train_studies,tr,cache_root); valid_ds=StudyDataset(valid_studies,va,cache_root)
        train_loader=DataLoader(train_ds,batch_size=args.study_batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,collate_fn=collate_study)
        valid_loader=DataLoader(valid_ds,batch_size=args.study_batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,collate_fn=collate_study)
    else:
        train_ds=ImageDataset(train_rows,tr,cache_root); valid_ds=ImageDataset(valid_rows,va,cache_root)
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
        valid_loader=DataLoader(valid_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    model, load_info=build_model(args.model); model.to(device)
    enc, head=[],[]
    for name,p in model.named_parameters():
        (head if (name.startswith("head") or name.startswith("classifier") or name.startswith("fc")) else enc).append(p)
    if not head: head=enc; enc=[]
    optimizer=torch.optim.AdamW([{"params":head,"lr":args.lr},{"params":enc,"lr":args.lr*args.encoder_lr_multiplier}] if enc else [{"params":head,"lr":args.lr}],weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    loss_fn=ASL(args.asl_gamma_neg,args.asl_gamma_pos,args.asl_clip) if args.loss=="asl" else nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and not args.no_amp)
    config=vars(args).copy(); config.update({"device":str(device),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),"load_info":load_info,"train_images":len(train_rows),"train_studies":len(train_studies),"valid_images":len(valid_rows),"valid_studies":len(valid_studies),"test_used":False})
    (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str)+"\n"); (out/"metrics.jsonl").write_text("")
    best=(-1.0,-1.0)
    for epoch in range(1,args.epochs+1):
        model.train(); frozen=epoch<=args.freeze_epochs
        for p in enc: p.requires_grad=not frozen
        if frozen: model.eval();
        elif epoch>1: model.train()
        started=time.time(); total=0.; steps=0
        for batch in train_loader:
            if study_mode: images,targets,sids,counts=batch
            else: images,targets,sids=batch; counts=None
            images=images.to(device,non_blocking=True); targets=targets.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda",dtype=torch.float16) if scaler.is_enabled() else nullcontext()):
                z=model(images)
                if study_mode: z=aggregate(z,counts,args.train_pool)
                loss=loss_fn(z,targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=float(loss.detach().cpu()); steps+=1
        sched.step(); val=evaluate(model,valid_loader,device,args.eval_pool,study_mode)
        val.update({"epoch":epoch,"train_loss":total/max(steps,1),"lr":sched.get_last_lr()[0],"epoch_seconds":time.time()-started,"peak_memory_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0})
        with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(val,ensure_ascii=False)+"\n")
        ck={"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"epoch":epoch,"val":val,"args":vars(args),"load_info":load_info}; torch.save(ck,out/"last.pt")
        key=(val["macro_auc"],val["macro_f1"])
        if key>best: best=key; torch.save(ck,out/"best.pt")
        print(json.dumps(val,ensure_ascii=False),flush=True); 
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    (out/"done.ok").write_text("completed\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=sorted(MODEL_CONFIG),required=True); p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--study-batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=4); p.add_argument("--size",type=int,default=224)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--encoder-lr-multiplier",type=float,default=1.0); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--seed",type=int,default=20261024)
    p.add_argument("--norm",choices=("imagenet","half"),default="imagenet"); p.add_argument("--loss",choices=("bce","asl"),default="bce"); p.add_argument("--asl-gamma-neg",type=float,default=2.0); p.add_argument("--asl-gamma-pos",type=float,default=0.0); p.add_argument("--asl-clip",type=float,default=.05)
    p.add_argument("--train-unit",choices=("image","study"),default="image"); p.add_argument("--train-pool",choices=("max","mean_logit","mean_prob","lse"),default="mean_logit"); p.add_argument("--eval-pool",choices=("max","mean_logit","mean_prob","lse"),default="max")
    p.add_argument("--hflip",action="store_true"); p.add_argument("--brightness",type=float,default=0.0); p.add_argument("--contrast",type=float,default=0.0); p.add_argument("--noise",type=float,default=0.0); p.add_argument("--freeze-epochs",type=int,default=0); p.add_argument("--deterministic",action="store_true"); p.add_argument("--no-amp",action="store_true")
    run(p.parse_args())
if __name__=="__main__": main()
