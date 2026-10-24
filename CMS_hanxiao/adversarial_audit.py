#!/usr/bin/env python3
import csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from train_baseline import ROOT, TensorTransform, read_records

OUT=ROOT.parent/'runs'/'densenet121_baseline'
DEVICE=torch.device('cuda')
MEAN=torch.tensor((0.485,0.456,0.406),device=DEVICE).view(1,3,1,1)
STD=torch.tensor((0.229,0.224,0.225),device=DEVICE).view(1,3,1,1)
class ValidDS(Dataset):
    def __init__(self,records): self.records=records; self.tf=TensorTransform(224,False,False)
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(r['path']) as im: x=self.tf(im.convert('RGB'))
        return x,torch.from_numpy(r['target']),r['path'],r['study_id']
def clamp_norm(x): return (torch.clamp(x*STD+MEAN,0,1)-MEAN)/STD
def main():
    ck=OUT/'best.pt'
    if not ck.exists(): raise SystemExit('best.pt missing')
    payload=torch.load(ck,map_location='cpu')
    model=timm.create_model('densenet121.tv_in1k',pretrained=False)
    model.reset_classifier(10); model.load_state_dict(payload['model'],strict=True); model.to(DEVICE).eval()
    records,_=read_records('valid')
    loader=DataLoader(ValidDS(records),batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
    loss_fn=nn.BCEWithLogitsLoss(reduction='none')
    eps=4.0/255.0; step=2.0/255.0; steps=3
    epsn=torch.tensor((eps,eps,eps),device=DEVICE).view(1,3,1,1)/STD
    stepn=torch.tensor((step,step,step),device=DEVICE).view(1,3,1,1)/STD
    rows=[]; clean_losses=[]; fg_losses=[]; pg_losses=[]; flips=[]
    for x,y,paths,studies in loader:
        x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
        with torch.no_grad(): clean_logits=model(x); clean_prob=torch.sigmoid(clean_logits); clean_loss=loss_fn(clean_logits,y).mean(1)
        # Untargeted FGSM: maximize multilabel BCE loss.
        adv=x.detach().clone().requires_grad_(True); fg_obj=loss_fn(model(adv),y).mean(); grad=torch.autograd.grad(fg_obj,adv)[0]
        fg=clamp_norm(x + epsn*grad.sign()).detach()
        with torch.no_grad(): fg_logits=model(fg); fg_loss=loss_fn(fg_logits,y).mean(1); fg_prob=torch.sigmoid(fg_logits)
        # Untargeted 3-step PGD within the same epsilon ball.
        adv=x.detach().clone()
        for _ in range(steps):
            adv.requires_grad_(True); obj=loss_fn(model(adv),y).mean(); grad=torch.autograd.grad(obj,adv)[0]
            adv=adv + stepn*grad.sign(); delta=torch.clamp(adv-x,-epsn,epsn); adv=clamp_norm(x+delta).detach()
        with torch.no_grad(): pg_logits=model(adv); pg_loss=loss_fn(pg_logits,y).mean(1); pg_prob=torch.sigmoid(pg_logits)
        clean_losses.extend(clean_loss.cpu().tolist()); fg_losses.extend(fg_loss.cpu().tolist()); pg_losses.extend(pg_loss.cpu().tolist())
        flip=((clean_prob>=.5)!=(pg_prob>=.5)).sum(1).cpu().tolist(); flips.extend(flip)
        for i,p in enumerate(paths):
            pp=Path(p); rows.append({'split':'valid','subject_id':pp.parts[-3],'study_id':studies[i],'dicom_id':pp.stem,'clean_loss':float(clean_loss[i]),'fgsm_loss':float(fg_loss[i]),'pgd_loss':float(pg_loss[i]),'pgd_loss_increase':float(pg_loss[i]-clean_loss[i]),'threshold_flips':int(flip[i])})
    vals=np.asarray([r['pgd_loss_increase'] for r in rows]); cutoff=float(np.quantile(vals,.99)) if len(vals) else float('nan')
    for r in rows:r['candidate_top1pct']=bool(r['pgd_loss_increase']>=cutoff)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'adversarial_suspect_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    summary={'scope':'official train-derived model evaluated on valid only','test_accessed':False,'sample_count':len(rows),'attack':'untargeted FGSM and 3-step PGD','epsilon_pixel':eps,'pgd_step_pixel':step,'pgd_steps':steps,'mean_clean_loss':float(np.mean(clean_losses)),'mean_fgsm_loss':float(np.mean(fg_losses)),'mean_pgd_loss':float(np.mean(pg_losses)),'mean_pgd_loss_increase':float(np.mean(vals)),'median_pgd_loss_increase':float(np.median(vals)),'p99_pgd_loss_increase':cutoff,'mean_threshold_flips':float(np.mean(flips)),'candidate_top1pct_count':sum(r['candidate_top1pct'] for r in rows),'interpretation':'Candidates are robustness outliers requiring review; this audit does not label them as adversarial without provenance evidence.'}
    (OUT/'adversarial_robustness.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'robustness.ok').write_text('completed\\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
