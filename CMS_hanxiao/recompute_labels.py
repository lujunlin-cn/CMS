import csv, json, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path('/home/zhaona/CMS_hanxiao/cxr_first'); OUT=Path('/home/zhaona/CMS_hanxiao/audit')
def parse(value):
    labels=set(); invalid=[]
    for token in str(value or '').split(';'):
        token=token.strip()
        if not token: continue
        try: x=int(token)
        except (TypeError,ValueError): invalid.append(token); continue
        if 1<=x<=10: labels.add(x)
        else: invalid.append(token)
    return labels,invalid
summary=json.loads((OUT/'audit_summary.json').read_text())
summary['label_parser']={'format':'semicolon-separated multi-label Predict_class','valid_class_ids':list(range(1,11)),'recomputed_at_epoch':time.time()}
rows_out=[]
for split in ('train','valid'):
    labels_by_study=defaultdict(set); counts=Counter(); invalid=[]; multi=0; rows_count=0
    with (ROOT/f'{split}.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows_count += 1; labels_by_study[row['Study_id']]
            labels,bad=parse(row.get('Predict_class',''))
            if len(labels)>1: multi += 1
            for x in labels: counts[x]+=1; labels_by_study[row['Study_id']].add(x)
            for token in bad:
                counts['INVALID'] += 1
                if len(invalid)<20: invalid.append([row['Study_id'],row['Dicom_id'],token])
    labels_json={k:sorted(v) for k,v in sorted(labels_by_study.items())}
    (OUT/f'{split}_study_labels.json').write_text(json.dumps(labels_json,ensure_ascii=False,indent=2)+'\n')
    entry=summary['splits'][split]
    entry['label_row_counts']={str(k):int(v) for k,v in sorted(counts.items(),key=lambda z:str(z[0]))}
    entry['study_label_counts']={str(k):sum(k in v for v in labels_by_study.values()) for k in range(1,11)}
    entry['no_finding_cooccurrence_studies']=sum(10 in v and len(v)>1 for v in labels_by_study.values())
    entry['multi_label_rows']=multi; entry['invalid_label_tokens_examples']=invalid
    for x in range(1,11):
        rows_out.append({'split':split,'predict_class':x,'study_count':sum(x in v for v in labels_by_study.values()),'study_total':len(labels_by_study),'prevalence':sum(x in v for v in labels_by_study.values())/len(labels_by_study) if labels_by_study else None})
with (OUT/'label_distribution.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['split','predict_class','study_count','study_total','prevalence']);w.writeheader();w.writerows(rows_out)
summary['notes']=[x for x in summary.get('notes',[]) if 'Predict_class' not in x]
summary['notes'].append('Predict_class was recomputed as the official semicolon-separated multi-label field; image QC files were preserved from the completed prior audit.')
(OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'train_studies':summary['splits']['train']['unique_studies'],'valid_studies':summary['splits']['valid']['unique_studies'],'train_invalid':summary['splits']['train']['label_row_counts'].get('INVALID',0),'valid_invalid':summary['splits']['valid']['label_row_counts'].get('INVALID',0),'train_multi_rows':summary['splits']['train']['multi_label_rows'],'valid_multi_rows':summary['splits']['valid']['multi_label_rows']},ensure_ascii=False))
