#!/usr/bin/env python3
import csv, hashlib, json, shutil, time
from pathlib import Path
ROOT=Path('/home/zhaona/CMS_hanxiao'); PRE=ROOT/'pretrain_model'; OUT=ROOT/'compliance'
MODELS={
'densenet121':('densenet121.tv_in1k','ImageNet-1k','Apache-2.0','https://huggingface.co/timm/densenet121.tv_in1k'),
'efficientnetv2_s':('tf_efficientnetv2_s.in21k_ft_in1k','ImageNet-21k, fine-tuned ImageNet-1k','Apache-2.0','https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k'),
'convnext_tiny':('convnext_tiny.fb_in22k_ft_in1k','ImageNet-22k, fine-tuned ImageNet-1k','Apache-2.0','https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k'),
'convnext_small':('convnext_small.fb_in22k_ft_in1k','ImageNet-22k, fine-tuned ImageNet-1k','Apache-2.0','https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k'),
'swin_small':('swin_small_patch4_window7_224.ms_in22k_ft_in1k','ImageNet-22k, fine-tuned ImageNet-1k','MIT','https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k'),
'dinov2_base_reg4':('vit_base_patch14_reg4_dinov2.lvd142m','LVD-142M self-supervised natural-image corpus','Apache-2.0','https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m'),
}
def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
OUT.mkdir(exist_ok=True); (OUT/'licenses').mkdir(exist_ok=True); (OUT/'model_cards').mkdir(exist_ok=True)
# Preserve local model-card evidence.
for alias in MODELS:
 src=PRE/alias/'README.md'
 if src.exists(): shutil.copy2(src,OUT/'model_cards'/f'{alias}.md')
apache=Path('/usr/share/common-licenses/Apache-2.0')
if apache.exists(): shutil.copy2(apache,OUT/'licenses'/'Apache-2.0.txt')
mit='''MIT License\n\nCopyright (c) Microsoft Corporation (Swin Transformer upstream)\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'''
(OUT/'licenses'/'MIT.txt').write_text(mit)
rows=[]; prov=[]
for alias,(mid,data,license,url) in MODELS.items():
 p=PRE/alias/'model.safetensors'; readme=PRE/alias/'README.md';
 rows.append({'asset':alias,'type':'encoder_pretrained_weight','path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p),'public_url':url,'license':license,'pretrain_data':data,'external_medical_data':'no','initialization':'approved_local_checkpoint','gated':'no','status':'local_model_card_reviewed_manual_submission_check_required'})
 prov.append({'component':alias,'model_id':mid,'local_path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'public_weight_url':url,'license':license,'upstream_training_data':data,'model_card':str(OUT/'model_cards'/f'{alias}.md'),'gated':False,'source_review_status':'local card and checksum recorded; verify public URL at submission'})
rows += [
 {'asset':'unet_decoder','type':'trainable_component','path':'unet_mil_model.py:UNetDecoder','size_bytes':'','sha256':'','public_url':'','license':'project code','pretrain_data':'none','external_medical_data':'no','initialization':'random','gated':'no','status':'prepared_not_trained'},
 {'asset':'spatial_evidence_head','type':'trainable_component','path':'unet_mil_model.py:UNetMIL.evidence_head','size_bytes':'','sha256':'','public_url':'','license':'project code','pretrain_data':'none','external_medical_data':'no','initialization':'random','gated':'no','status':'prepared_not_trained'},
 {'asset':'topk_mil_pool','type':'parameter_free_component','path':'unet_mil_model.py:UNetMIL.masked_topk_mean','size_bytes':'','sha256':'','public_url':'','license':'project code','pretrain_data':'none','external_medical_data':'no','initialization':'no_parameters','gated':'no','status':'prepared_not_trained'},
 {'asset':'official_train_valid_data','type':'competition_data','path':str(ROOT/'cxr_first'),'size_bytes':'','sha256':'','public_url':'','license':'competition controlled data','pretrain_data':'official competition data only','external_medical_data':'no','initialization':'not_applicable','gated':'not_applicable','status':'local_only'},
]
with (OUT/'asset_inventory.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'weight_provenance.json').write_text(json.dumps({'created_at_epoch':time.time(),'weights':prov,'policy':'No external medical data, medical pretraining, medical teacher, pseudomasks, or automatic downloads are allowed.'},ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((ROOT/'unet_mil_data/manifest_summary.json').read_text())
lineage={'created_at_epoch':time.time(),'task':'unet_mil_study_multilabel','train_manifest_sha256':manifest['train_studies.jsonl']['sha256'],'valid_manifest_sha256':manifest['valid_studies.jsonl']['sha256'],'test_used':False,'external_medical_data_used':False,'external_teacher_used':False,'pseudo_mask_training_used':False,'medical_self_supervised_stage_used':False,'model_code':'unet_mil_model.py','training_code':'train_unet_mil_single_gpu.py','preprocessing':'letterbox with valid-area mask; ImageNet normalization; no external medical preprocessor','study_aggregation':'mean image logits within Study','spatial_mil':'masked top-k mean logits','status':'prepared_not_trained'}
(OUT/'training_lineage.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2)+'\n')
(OUT/'migration_report.md').write_text('''# U-Net + MIL 合规改造记录\n\n- 依据：`unet_mil_compliance_guide.md`。\n- 已准备官方 Train/Valid Study manifest；Test 未进入 manifest。\n- 已实现经核验的通用编码器、随机初始化 U-Net 解码器、10 通道空间证据头、几何 padding mask 和 masked Top-K MIL。\n- Study 内聚合为图像 logits 的固定均值。\n- 未引入外部医学图像、医学权重、教师、分割器、伪掩码或医学自监督阶段。\n- 尚未启动正式训练；已完成 densenet、efficientnet、convnext、swin、DINOv2 的结构/梯度 smoke test。\n- Swin 的 relative_position_index/attn_mask 是输入窗口派生 buffer；DINOv2 features-only 路径明确记录未使用最终分类 norm 参数，需在正式实验记录中保留。\n''')
(OUT/'rules_revision_log.md').write_text('''# 规则来源修订记录\n\n## 2026-10-24\n\n`unet_mil_compliance_guide.md` 记录的官网当前 PDF（`aa80bececc2ee40e69cde5f8f2a0eaec.pdf`）将单图平均推理上限写为 100 ms，并更新 `cost.py` 参考命令增加 submission.csv 参数。历史 PDF `9c2194e6968d044c785e166df10c8255.pdf` 的 100 秒表述保留为历史记录；正式部署按当前官网/PDF 的 100 ms 口径执行。完整核验包提交阶段的其他差异仍待主办方书面确认。\n''')
print('wrote',OUT)
