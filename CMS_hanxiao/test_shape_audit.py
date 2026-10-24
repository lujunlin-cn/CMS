import csv,json,time
from collections import Counter
from pathlib import Path
from PIL import Image
ROOT=Path('/home/zhaona/CMS_hanxiao/cxr_first'); OUT=ROOT.parent/'audit'
rows=[]
with (ROOT/'test.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
records=[]; missing=[]; errors=[]; mismatches=[]; keys=set(); dup=[]; modes=Counter(); sizes=[]
for row in rows:
 key=(row['Subject_id'],row['Study_id'],row['Dicom_id'])
 if key in keys: dup.append(key)
 keys.add(key)
 p=ROOT/'test'/row['Subject_id']/row['Study_id']/(row['Dicom_id']+'.jpg')
 rec={'subject_id':row['Subject_id'],'study_id':row['Study_id'],'dicom_id':row['Dicom_id'],'relative_path':str(p.relative_to(ROOT)),'meta_rows':row.get('Rows',''),'meta_columns':row.get('Columns',''),'file_bytes':p.stat().st_size if p.exists() else '', 'header_ok':0,'width':'','height':'','mode':'','shape_matches_metadata':''}
 if not p.exists(): missing.append(key); records.append(rec); continue
 sizes.append(p.stat().st_size)
 try:
  with Image.open(p) as im:
   rec.update({'header_ok':1,'width':im.width,'height':im.height,'mode':im.mode,'shape_matches_metadata':bool(im.height==int(row['Rows']) and im.width==int(row['Columns']))})
   modes[im.mode]+=1
   if not rec['shape_matches_metadata']: mismatches.append([key,[int(row['Rows']),int(row['Columns'])],[im.height,im.width]])
 except Exception as e:
  rec['error']=f'{type(e).__name__}: {e}'; errors.append([key,rec['error']])
 records.append(rec)
fields=['subject_id','study_id','dicom_id','relative_path','meta_rows','meta_columns','file_bytes','header_ok','width','height','mode','shape_matches_metadata','error']
for r in records:r.setdefault('error','')
with (OUT/'test_shape_audit.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
summary={'scope':'test shape/header/byte-size audit only; no pixel statistics or full image decode','created_at_epoch':time.time(),'csv_rows':len(rows),'filesystem_images':len(list((ROOT/'test').rglob('*.jpg'))),'missing_files':len(missing),'duplicate_csv_keys':len(dup),'header_checked':len(records),'header_failures':len(errors),'metadata_dimension_mismatches':len(mismatches),'mode_counts':dict(modes),'file_bytes_total':sum(sizes),'file_bytes_min':min(sizes) if sizes else None,'file_bytes_max':max(sizes) if sizes else None,'missing_examples':missing[:20],'error_examples':errors[:20],'mismatch_examples':mismatches[:20],'test_data_used_for_training_or_model_selection':False}
(OUT/'test_shape_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
# update overall summary with shape-only result
p=OUT/'audit_summary.json'; overall=json.loads(p.read_text()); d=overall['splits']['test']; d.update({'shape_checked':True,'shape_check_failures':len(errors),'metadata_dimension_mismatches':len(mismatches),'file_size_bytes_total':sum(sizes),'file_size_bytes_min':min(sizes) if sizes else None,'file_size_bytes_max':max(sizes) if sizes else None,'mode_counts':dict(modes)}); overall['notes'].append('Test was checked for image header shape and file byte size only; no pixel statistics or full decode were performed.'); p.write_text(json.dumps(overall,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
