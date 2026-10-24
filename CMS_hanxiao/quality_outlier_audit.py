import csv, math
from pathlib import Path
root=Path('/home/zhaona/CMS_hanxiao')
qc=root/'audit/image_qc_train_valid.csv'; out=root/'audit/quality_outliers.csv'
rows=list(csv.DictReader(qc.open()))
flags=[]
for r in rows:
    try:
        w=float(r['width']); h=float(r['height']); mean=float(r['mean']); std=float(r['std']); z=float(r['zero_fraction']); m=float(r['max_fraction']); p01=float(r['p01']); p99=float(r['p99'])
    except Exception: continue
    reasons=[]
    if std < 5: reasons.append('very_low_contrast_std<5')
    if p99-p01 < 15: reasons.append('narrow_dynamic_range<15')
    if z > 0.80: reasons.append('mostly_black>0.80')
    if m > 0.80: reasons.append('mostly_white>0.80')
    ratio=max(w,h)/max(1,min(w,h))
    if ratio > 4.0: reasons.append('extreme_aspect_ratio>4')
    if reasons:
        rr=dict(r); rr['reasons']=';'.join(reasons); flags.append(rr)
fields=list(rows[0].keys())+['reasons'] if rows else ['reasons']
with out.open('w',newline='') as f:
    ww=csv.DictWriter(f,fieldnames=fields);ww.writeheader();ww.writerows(flags)
from collections import Counter
c=Counter()
for r in flags:
    for x in r['reasons'].split(';'): c[x]+=1
print({'images':len(rows),'flagged':len(flags),'reasons':dict(c),'by_split':Counter(r['split'] for r in flags)})
