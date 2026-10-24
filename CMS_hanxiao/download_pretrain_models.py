#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('/home/zhaona/pretrain_model')
BASE = 'https://hf-mirror.com'
MODELS = [
    {'alias':'densenet121','repo':'timm/densenet121.tv_in1k','revision':'f0d0f2698a02cb133b09d48396db6e1e46fe9f3b','size':32334434,'sha256':'c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358'},
    {'alias':'efficientnetv2_s','repo':'timm/tf_efficientnetv2_s.in21k_ft_in1k','revision':'ea9abc143ea2b9d8e1ec1de277bce02149b9cf0e','size':86523256,'sha256':'6f1933fb6c0d760eae03863ad0110570393e16c0610f8fe94dc9f978e52ec59c'},
    {'alias':'convnext_tiny','repo':'timm/convnext_tiny.fb_in22k_ft_in1k','revision':'bc48a87f119bae1b06418f5340bfa97ef75d4609','size':114374272,'sha256':'77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a'},
    {'alias':'convnext_small','repo':'timm/convnext_small.fb_in22k_ft_in1k','revision':'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b','size':200928946,'sha256':'2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9'},
    {'alias':'swin_small','repo':'timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k','revision':'221ee757828e4772df860bd42ef476de78dd5fc6','size':200037522,'sha256':'e17adf843764761f138cb8cab081d9a37824db4a948c0082c7af7f0092d19581'},
    {'alias':'dinov2_base_reg4','repo':'timm/vit_base_patch14_reg4_dinov2.lvd142m','revision':'3b06466a5548c52b8b98822e1390987695fcbf82','size':346344168,'sha256':'c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006'},
]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def run_curl(url, out):
    tmp = out.with_suffix(out.suffix + '.part')
    cmd=['curl','--fail','--location','--retry','8','--retry-delay','4','--retry-all-errors','--connect-timeout','30','--max-time','7200','-C','-','-o',str(tmp),url]
    print('DOWNLOAD', url, '->', tmp, flush=True)
    rc=subprocess.run(cmd).returncode
    if rc:
        raise RuntimeError(f'curl failed rc={rc}: {url}')
    tmp.replace(out)

def get_file(model, filename, expected_size=None, expected_sha=None):
    outdir=ROOT/model['alias']; outdir.mkdir(parents=True, exist_ok=True)
    out=outdir/filename
    if out.exists() and expected_size is not None:
        actual_size=out.stat().st_size
        if actual_size==expected_size:
            actual_sha=sha256(out)
            if actual_sha==expected_sha:
                print('SKIP_VALID', out, actual_size, actual_sha, flush=True)
                return
            print('BAD_SHA_REMOVE', out, actual_sha, flush=True)
            out.unlink()
    url=f"{BASE}/{model['repo']}/resolve/{model['revision']}/{filename}?download=true"
    run_curl(url,out)
    size=out.stat().st_size; digest=sha256(out)
    if expected_size is not None and (size != expected_size or digest != expected_sha):
        raise RuntimeError(f'CHECK_FAILED {out}: size={size} sha256={digest} expected_size={expected_size} expected_sha={expected_sha}')
    print('VALID', out, size, digest, flush=True)

ROOT.mkdir(parents=True, exist_ok=True)
for model in MODELS:
    print('MODEL_START', model['alias'], model['repo'], model['revision'], flush=True)
    get_file(model,'model.safetensors',model['size'],model['sha256'])
    get_file(model,'config.json')
    get_file(model,'README.md')
    record={
        'alias':model['alias'],'repo_id':model['repo'],'revision':model['revision'],
        'model_file':'model.safetensors','model_size_bytes':model['size'],'model_sha256':model['sha256'],
        'endpoint':BASE,'downloaded_at_epoch':time.time(),
    }
    (ROOT/model['alias']/'checkpoint.lock.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('MODEL_DONE', model['alias'], flush=True)
print('ALL_DONE', flush=True)
