#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for run in sorted(root.iterdir() if root.exists() else []):
        if not run.is_dir():
            continue
        config = json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
        metrics = []
        if (run/'metrics.jsonl').exists():
            metrics = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
        best = max(metrics, key=lambda x: x.get('macro_auc', float('-inf'))) if metrics else {}
        rows.append({'model':run.name,'status':'done' if (run/'done.ok').exists() else ('failed' if (run/'FAILED').exists() else 'running_or_pending'),'epoch':best.get('epoch',''),'macro_auc':best.get('macro_auc',''),'macro_f1':best.get('macro_f1',''),'train_loss':best.get('train_loss',''),'best_auc_per_class':json.dumps(best.get('auc_per_class',[]),ensure_ascii=False),'train_images':config.get('train_images',''),'valid_studies':config.get('valid_studies',''),'path':str(run)})
    out=root/'comparison_summary.csv'; fields=list(rows[0].keys()) if rows else ['model','status','epoch','macro_auc','macro_f1','train_loss','best_auc_per_class','train_images','valid_studies','path']
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r: float(r['macro_auc']) if r['macro_auc'] != '' else -1,reverse=True))
    print(json.dumps({'runs':len(rows),'summary':str(out)},ensure_ascii=False))
if __name__ == '__main__': main()
