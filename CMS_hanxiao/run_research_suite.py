#!/usr/bin/env python3
"""Fresh, single-GPU research queue for the report's first-stage screen."""
import json, os, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/zhaona/CMS_hanxiao")
PY = "/home/zhaona/anaconda3/envs/ljhtorch/bin/python"
OUT = ROOT / "runs" / "research_v1"
GPU_IDS = [0]  # GPU1 is intentionally occupied by another user's job.
JOBS = [
    ("cs_p0", "backbone", ["--model","convnext_small","--train-unit","image","--loss","bce","--eval-pool","max"]),
    ("cs_study_mean", "backbone", ["--model","convnext_small","--train-unit","study","--train-pool","mean_logit","--eval-pool","mean_logit","--study-batch-size","8"]),
    ("cs_asl_g2", "backbone", ["--model","convnext_small","--train-unit","image","--loss","asl","--asl-gamma-neg","2","--eval-pool","max"]),
    ("cs_asl_g4", "backbone", ["--model","convnext_small","--train-unit","image","--loss","asl","--asl-gamma-neg","4","--eval-pool","max"]),
    ("cs_lr01", "backbone", ["--model","convnext_small","--train-unit","image","--encoder-lr-multiplier","0.1","--eval-pool","max"]),
    ("eff_norm_imagenet", "backbone", ["--model","efficientnetv2_s","--norm","imagenet","--train-unit","image","--eval-pool","max"]),
    ("eff_norm_half", "backbone", ["--model","efficientnetv2_s","--norm","half","--train-unit","image","--eval-pool","max"]),
    ("unet_global", "unet", ["--model","densenet121","--fusion","global","--spatial-pool","mean","--study-pool","mean","--lr","3e-4","--encoder-lr-multiplier","0.1","--freeze-encoder","--freeze-epochs","1"]),
    ("unet_residual", "unet", ["--model","densenet121","--fusion","residual","--spatial-pool","topk","--topk-ratio","0.2","--study-pool","mean","--lr","3e-4","--encoder-lr-multiplier","0.1","--freeze-encoder","--freeze-epochs","1"]),
    ("unet_lse", "unet", ["--model","densenet121","--fusion","decoder","--spatial-pool","logsumexp","--pool-temperature","1.0","--study-pool","mean","--lr","1e-4"]),
]

def now(): return datetime.now(timezone.utc).isoformat()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"started_at": now(), "gpus": GPU_IDS, "jobs": {name:{"status":"pending"} for name,_,_ in JOBS}}
    (OUT/"queue_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
    log = (OUT/"queue.log").open("w", buffering=1)
    def emit(msg):
        line=f"{now()} {msg}"; print(line,flush=True); log.write(line+"\n")
    for name,kind,extra in JOBS:
        d=OUT/name
        if d.exists(): shutil.rmtree(d)
    pending=list(JOBS); active={}
    while pending or active:
        for gpu in GPU_IDS:
            if gpu in active or not pending: continue
            name,kind,extra=pending.pop(0); d=OUT/name; d.mkdir(parents=True)
            if kind=="backbone":
                cmd=[PY,str(ROOT/"research_train.py"),"--out",str(d),"--epochs","5","--batch-size","32","--workers","8"]+extra
            else:
                cmd=[PY,str(ROOT/"train_unet_mil_advanced.py"),"--out",str(d),"--epochs","5","--study-batch-size","4","--workers","8","--size","224"]+extra
            env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            fh=(d/"train.log").open("w",buffering=1)
            emit(f"start {name} gpu={gpu}: {' '.join(cmd)}")
            p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=fh,stderr=subprocess.STDOUT)
            active[gpu]=(name,p,fh)
            state["jobs"][name]={"status":"running","gpu":gpu,"started_at":now(),"cmd":cmd}
        for gpu,(name,p,fh) in list(active.items()):
            code=p.poll()
            if code is None: continue
            fh.close(); del active[gpu]
            state["jobs"][name].update({"status":"complete" if code==0 and (OUT/name/"done.ok").exists() else "failed","returncode":code,"finished_at":now()})
            emit(f"finish {name} gpu={gpu} returncode={code}")
        state["updated_at"]=now(); (OUT/"queue_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
        if active: time.sleep(20)
    state["finished_at"]=now(); state["status"]="complete" if all(x["status"]=="complete" for x in state["jobs"].values()) else "failed"
    (OUT/"queue_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
    emit("queue finished status="+state["status"]); log.close()
    return 0 if state["status"]=="complete" else 1
if __name__=="__main__": raise SystemExit(main())
