#!/usr/bin/env python3
"""Summarize completed research_v2 runs without touching Test."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--queue-root", required=True); p.add_argument("--out", required=True)
    args=p.parse_args(); root=Path(args.queue_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in sorted(root.glob("*_s*")):
        if not d.is_dir(): continue
        rec={"run":d.name,"status":"complete" if (d/"done.ok").exists() else ("failed" if (d/"FAILED").exists() else "incomplete")}
        if (d/"config.json").exists():
            try: rec.update({"model":json.loads((d/"config.json").read_text()).get("model"),"seed":json.loads((d/"config.json").read_text()).get("seed")})
            except Exception as e: rec["config_error"]=type(e).__name__
        metrics=[]
        if (d/"metrics.jsonl").exists():
            for line in (d/"metrics.jsonl").read_text().splitlines():
                try: metrics.append(json.loads(line))
                except Exception: pass
        if metrics:
            best=max(metrics,key=lambda x:(float(x.get("macro_auc",-1)),float(x.get("macro_f1",-1))))
            rec.update({"best_epoch":best.get("epoch"),"macro_auc":best.get("macro_auc"),"macro_f1":best.get("macro_f1"),"auc_per_class":json.dumps(best.get("auc_per_class",[]),separators=(",",":"))})
        rows.append(rec)
    fields=["run","status","model","seed","best_epoch","macro_auc","macro_f1","auc_per_class","config_error"]
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
    print(f"runs={len(rows)} complete={sum(r['status']=='complete' for r in rows)}")
if __name__=="__main__": main()
