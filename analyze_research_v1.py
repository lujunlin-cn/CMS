#!/usr/bin/env python3
"""Generate the evidence-bounded interpretation of the first-stage screen."""
import argparse, csv, json
from pathlib import Path

OLD = {
    "convnext_small": [0.695235,0.841553,0.791433,0.707016,0.844758,0.743787,0.791739,0.690440,0.894847,0.864293],
    "efficientnetv2_s": [0.681911,0.833383,0.723301,0.651056,0.853097,0.741712,0.765042,0.678657,0.895791,0.865679],
    "unet_old": [0.639721,0.652200,0.707324,0.633822,0.818139,0.694211,0.697817,0.669499,0.879899,0.848078],
}
NAMES=["纵隔心影增大","气胸","肺部实变","肺炎","肺水肿","心影扩大","肺不张","肺部阴影","胸腔积液","无异常发现"]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--out',required=True); a=p.parse_args(); root=Path(a.root)
    runs={}
    for d in root.iterdir():
        if not d.is_dir() or not (d/'metrics.jsonl').exists(): continue
        ms=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines() if x.strip()]
        if not ms: continue
        runs[d.name]=max(ms,key=lambda x:(x['macro_auc'],x['macro_f1']))
    ranked=sorted(runs.items(),key=lambda kv:(kv[1]['macro_auc'],kv[1]['macro_f1']),reverse=True)
    lines=['# CMS 第一阶段研究分析（实际训练结果）','',
           '数据边界：所有结果来自官方 Train 训练和官方 Valid 选择；Test 未读取。每个实验 5 轮，best 严格按未舍入 Valid Macro-AUC、同分再按 Macro-F1 保存。由于本轮是单次 seed 的筛选，下面是探索性证据，不是最终泛化结论。','',
           '## 结果排序','', '|排名|实验|best epoch|Macro-AUC|Macro-F1|每轮耗时 s|显存 MB|', '|---:|---|---:|---:|---:|---:|---:|']
    for i,(name,m) in enumerate(ranked,1): lines.append(f"|{i}|{name}|{m.get('epoch','')}|{m['macro_auc']:.6f}|{m['macro_f1']:.6f}|{m.get('epoch_seconds',0):.1f}|{m.get('peak_memory_mb',0):.0f}|")
    lines += ['', '## 关键结论', '',
      f"- ConvNeXt-Small 的 encoder 学习率倍率 0.1（`cs_lr01`）是本轮新基座筛选第一名，Macro-AUC={runs['cs_lr01']['macro_auc']:.6f}；相对本轮匹配 P0 为 {runs['cs_lr01']['macro_auc']-runs['cs_p0']['macro_auc']:+.6f}，相对历史固定 Small 参考 0.786510 仅 {runs['cs_lr01']['macro_auc']-0.786510:+.6f}，应视为接近持平，进入多 seed 复核而不是直接宣布提升。",
      f"- EfficientNetV2-S 的 `.5/.5` normalization（`eff_norm_half`）达到 {runs['eff_norm_half']['macro_auc']:.6f}，高于本轮 ImageNet normalization 对照 {runs['eff_norm_imagenet']['macro_auc']:.6f}，差值 {runs['eff_norm_half']['macro_auc']-runs['eff_norm_imagenet']['macro_auc']:+.6f}；这是本轮最清晰的单因素配置差异，但仍需重复 seed。",
      f"- ASL gamma-neg=2/4（{runs['cs_asl_g2']['macro_auc']:.6f}/{runs['cs_asl_g4']['macro_auc']:.6f}）均低于本轮 BCE P0（{runs['cs_p0']['macro_auc']:.6f}），本数据首轮不支持把 ASL 作为默认替换。它们的 F1 较高不能改变 Macro-AUC 排序。",
      f"- Study-level mean-logit 训练（`cs_study_mean`）为 {runs['cs_study_mean']['macro_auc']:.6f}，低于图像 BCE P0；当前 Study 监督与 batch/视图采样组合没有显示收益，不能单独归因于 mean-logit。",
      f"- 修正参数组后的 U-Net global（`unet_global_v2`）为 {runs['unet_global_v2']['macro_auc']:.6f}，残差融合（`unet_residual`）为 {runs['unet_residual']['macro_auc']:.6f}，均明显高于历史 U-Net 0.724071，但仍低于 Small 强基座。本轮最值得继续的是 residual 的互补性和多 seed稳定性；原 `unet_global` 是旧冻结 bug 的 head-only 结果，已标记为无效，不纳入结构结论。",
      f"- 空间 normalized LSE（`unet_lse`）为 {runs['unet_lse']['macro_auc']:.6f}，没有优于残差或历史 U-Net；平滑池化不能只凭理论先验保留。"]
    lines += ['', '## 类别级差异（best checkpoint）', '', '以下表格是每个实验 best 轮的十类 AUC，便于定位下一轮研究类别；类别级变化不能替代 Macro-AUC。','', '|类别|Small LR0.1|Small P0|差值|EffNet .5/.5|U-Net residual|', '|---|---:|---:|---:|---:|---:|']
    for i,n in enumerate(NAMES):
        a1=runs['cs_lr01']['auc_per_class'][i]; a0=runs['cs_p0']['auc_per_class'][i]; ae=runs['eff_norm_half']['auc_per_class'][i]; au=runs['unet_residual']['auc_per_class'][i]
        lines.append(f'|c{i+1} {n}|{a1:.6f}|{a0:.6f}|{a1-a0:+.6f}|{ae:.6f}|{au:.6f}|')
    lines += ['', '## 下一步建议', '',
      '1. 只对 `cs_lr01`、EfficientNet `.5/.5` 和 `unet_residual` 做匹配对照的 3 seed × 12 轮确认；保留 P0、EffNet ImageNet 和 U-Net decoder 作为同预算对照。',
      '2. 先比较冻结强模型的 Small+Dense 等权 ensemble 与 residual U-Net 的固定小权重增量；不在 228 位 Valid 患者上做高维逐类 stacking。',
      '3. 对确认候选再运行 Train+Valid 全量 FP32 推理与统一像素空间扰动审计；对抗敏感度仅作鲁棒性诊断，不把它当成真实攻击样本证据。',
      '4. 只有在确认阶段仍保持 Macro-AUC 优势，才冻结归一化、聚合、精度、权重和推理时延，最后再考虑 Test 的一次性固定推理。']
    Path(a.out).write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
