import csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('/home/zhaona/CMS_hanxiao/cxr_first'); inp=Path('/home/zhaona/CMS_hanxiao/audit/perceptual_collision_groups.csv'); out=Path('/home/zhaona/CMS_hanxiao/audit/perceptual_cross_split_review.csv')
rows=[]
for g in csv.DictReader(inp.open()):
    members=json.loads(g['members'])
    train=[m for m in members if m[0]=='train']; valid=[m for m in members if m[0]=='valid']
    for a in train:
      for b in valid:
        pa=root/a[0]/a[1]/a[2]/(a[3]+'.jpg'); pb=root/b[0]/b[1]/b[2]/(b[3]+'.jpg')
        with Image.open(pa) as im: xa=np.asarray(im.convert('L').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        with Image.open(pb) as im: xb=np.asarray(im.convert('L').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        va=xa.ravel(); vb=xb.ravel(); corr=float(np.corrcoef(va,vb)[0,1]) if va.std()>1e-8 and vb.std()>1e-8 else 0.0
        mae=float(np.mean(np.abs(va-vb))); mse=float(np.mean((va-vb)**2))
        rows.append({'hash':g['hash'],'train_path':'/'.join(a),'valid_path':'/'.join(b),'correlation_256':corr,'mae_256':mae,'mse_256':mse,'near_exact_candidate':bool(corr>=0.995 and mae<=0.01)})
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print({'pairs':len(rows),'near_exact_candidates':sum(x['near_exact_candidate'] for x in rows),'max_corr':max(x['correlation_256'] for x in rows),'min_mae':min(x['mae_256'] for x in rows)})
for x in sorted(rows,key=lambda z:(-z['correlation_256'],z['mae_256']))[:10]: print(x)
