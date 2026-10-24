#!/usr/bin/env python3
"""Summarize fresh research_v1 runs with Macro-AUC-first ranking."""
import argparse,csv,json
from pathlib import Path

BASE={"convnext_small":0.786510,"unet_mil":0.724071,"efficientnetv2_s":0.768963}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    root=Path(a.root); rows=[]
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        cfg={}
        if (d/"config.json").exists(): cfg=json.loads((d/"config.json").read_text())
        mets=[]
        if (d/"metrics.jsonl").exists():
            for line in (d/"metrics.jsonl").read_text().splitlines():
                if line.strip(): mets.append(json.loads(line))
        if not mets: rows.append({"run":d.name,"status":"no_metrics"}); continue
        best=max(mets,key=lambda x:(x.get("macro_auc",-1),x.get("macro_f1",-1)))
        model=cfg.get("model",cfg.get("model_id", "")); ref=BASE.get(model,BASE.get("unet_mil" if d.name.startswith("unet") else model))
        status = "invalid_head_only_freeze_bug" if d.name == "unet_global" else ("complete" if (d/"done.ok").exists() else "incomplete")
        rows.append({"run":d.name,"status":status,"model":model,"train_unit":cfg.get("train_unit", "study" if d.name.startswith("unet") else "image"),"loss":cfg.get("loss", "bce"),"fusion":cfg.get("fusion",""),"spatial_pool":cfg.get("spatial_pool",""),"best_epoch":best.get("epoch"),"macro_auc":best.get("macro_auc"),"macro_f1":best.get("macro_f1"),"delta_vs_reference":(best.get("macro_auc",0)-ref if ref is not None else None),"auc_per_class":json.dumps(best.get("auc_per_class",[]),ensure_ascii=False),"epoch_seconds":best.get("epoch_seconds")})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys()) if rows else ["run"]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    ordered=sorted([r for r in rows if r.get("macro_auc") is not None],key=lambda r:(r["macro_auc"],r.get("macro_f1",-1)),reverse=True)
    md=out.with_suffix(".md")
    lines=["# CMS 第一阶段研究实验结果", "", "排序规则：Valid Study Macro-AUC 降序；仅完全相等时比较 Macro-F1。Test 未读取。", "", "|排名|实验|模型/结构|best epoch|Macro-AUC|Macro-F1|相对固定参考|状态|", "|---:|---|---|---:|---:|---:|---:|---|"]
    for i,r in enumerate(ordered,1): lines.append(f"|{i}|{r['run']}|{r.get('model','')} {r.get('fusion','')} {r.get('spatial_pool','')}|{r.get('best_epoch','')}|{r.get('macro_auc',0):.6f}|{r.get('macro_f1',0):.6f}|{r.get('delta_vs_reference',0):+.6f}|{r.get('status','')}|")
    lines += ["", "## 解释", "", "只把 Macro-AUC 的增量视为主结果；Macro-F1 仅在 Macro-AUC 完全相等时用于比赛排序。单个 Valid 选择结果属于探索性证据，优胜者需用匹配对照和多 seed 复核。U-Net 的全局、残差和空间池化实验分别检验预训练全局语义、局部证据补充与平滑空间 MIL。"]
    md.write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__": main()
