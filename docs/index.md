# CMS 实验路线总索引：供 GPT 网页版讨论

整理日期：2026-11-07。目的：回顾整个项目已经尝试的路线、辨别有效证据与历史误判，再讨论后续方向。这里不是新实验计划，也没有启动、停止或修改任何训练。

## 如何使用这组文件

共 16 个 Markdown 文件。建议一起上传；若上传额度有限，先上传本页、00、01、13，再按讨论主题补充路线文件。文件正文已写入关键事实和数字，离开本机后也能阅读；末尾的相对链接供本地追溯，GPT 网页版不需要访问它们才能理解主要结论。

建议给 GPT 的开场指令：

> 请先阅读 index、00、01 和 13，再综合各路线。不要把计划当结果，不要把图像 AUC 当 Study AUC，不要把错误 heavy9、训练集评估或患者重叠结果当突破。区分“该实现无效”“该配置负向”和“整条方法不可行”。请先指出现有证据中最大的未决问题，再提出最多三个能改变决策、具备匹配对照和患者独立评估的实验；不要继续追逐同一 Valid 上千分之一的 best。

## 文件地图

| 文件 | 总路线 / 用途 | 主要历史分支 |
|---|---|---|
| [00 项目背景与证据规则](00_project_context_and_evidence_rules.md) | 共用定义、官方提交、关键纠错 | 类别、数据、指标、证据等级 |
| [01 验证与错误审计](01_validation_metrics_and_error_audit.md) | 我们如何判断实验是否有用 | 患者隔离、heavy9、X49、v3 泄漏、证据审阅 |
| [02 骨干容量与分辨率](02_backbones_capacity_and_resolution.md) | 换模型、提高输入细节 | DenseNet/EfficientNet/ConvNeXt、YOLO、Swin、DINO、XL、448/512/640 |
| [03 Study 监督与多视图](03_study_supervision_and_multiview.md) | 监督单位与真实图像集合 | mean/max/LSE、class-token、view position、noisy-OR、PIV1 |
| [04 空间证据与区域模型](04_spatial_evidence_and_region_models.md) | 让模型使用局部病灶证据 | U-Net、局部辅助、FPN、A2/A3/A4、X89、MC residual |
| [05 分类头与结构解耦](05_heads_label_structure_and_disentanglement.md) | 改变疾病表示和头部几何 | cosine、逐类温度、hierarchy、GRL、decorr、additive head |
| [06 损失与困难阴性](06_losses_ranking_and_hard_negatives.md) | 从普通 BCE 到任务对齐排序 | ASL、PU、bootstrap、AbnNeg、rank、SAM/adversarial |
| [07 采样与分布偏移](07_sampling_patient_balance_and_domain_shift.md) | 谁被训练、以什么权重训练 | 1/m、患者降权、heavy/AP-heavy、DRO、专家路由、MC |
| [08 预处理与增强](08_preprocessing_and_augmentation.md) | 改善输入而非只改 loss | contain、mask、JPEG、CLAHE、裁剪、mixup、SSL |
| [09 优化正则与种子](09_optimization_regularization_and_seeds.md) | 训练动力学和可复现性 | encoder LR、EMA、SWA、drop path、热启动、扫参、多 seed |
| [10 蒸馏与软标签](10_distillation_soft_labels_and_teachers.md) | 借用教师、OOF 与标签不确定性 | OOF 权重/软目标、view teacher、X90/X91/X92/X93、异构蒸馏 |
| [11 集成校准与 TTA](11_ensembles_calibration_and_tta.md) | 利用既有模型的互补信息 | S1/S2、四模型包、X42+X41、Platt/stacking、MC20、DZ0–4 |
| [12 量化部署与实验执行](12_quantization_deployment_and_execution.md) | 模型成绩能否被正确交付 | weight-only INT8、W8A8 事故、CSV、缓存、CUDA EMA、断点与队列 |
| [13 实验编号总表](13_experiment_catalog.md) | 按编号反查路线与版本 | 早期族系、X1–X93、A/T/E、PIV、R、MC、DZ |
| [14 原始运行证据清单](14_run_evidence_inventory.md) | 可追溯的日志覆盖与缺口 | 343 份 metrics 文件、内容去重、来源路径与摘要 |

## 项目的一条主线

```text
通用预训练分类器
  → Study 标签/图像监督及聚合方式
  → 患者频率与病例构成差异
  → 全量微调 ConvNeXt-L + X5 正则配方
  → X21 余弦头与 Swin 异构成员
  → heavy / AP-single×heavy 定向优化
  → v3“突破”的指标与泄漏复核
  → PIV 患者独立验证及其信号放大
  → 中等难度类别的真实空间 residual / 低容量融合
  → DINO 固定标准化融合及纠错—破坏分解
```

主线不是单调涨分史。大量变体失败；一些早期负结论由混杂实验推导，另一些后期高分由错误评估制造。下列共同认识比“哪次 best 更高”更重要。

1. **当前有正式提交证据的最好组合为 X42 ep1 与 X41 ep6 各 0.5，Test 77.66。** X41 是 SwinV2-L；不是 X21+X42。后者官方 Valid heavy9 只有 0.76603，前者为约 0.77277。
2. **困难格子和有效监督是瓶颈之一，但不是唯一可优化区域。** Train AP-single×heavy 604 Study，占 3.26%；其纵隔心影增大正例仅 25。中等类别 3/5/6/7 和跨视角排序仍有提升空间；MC 与 DZ 的实测收益目前很小。
3. **换架构带来的低相关不等于净收益。** Swin 已贡献真实互补；PIV3、DINO 也纠正部分排序，却引入新错序。需要净纠错证据，不能仅看相关系数。
4. **评估与实现问题曾反复压过模型信号。** 图像/Study 混用、heavy 分桶错误、X92 泄漏、Freeze 初始化错误、X35 loss 无效、量化/导出错误，都必须先剔除。
5. **目前没有证据保证还能上涨 3–5 个百分点。** 存在未充分验证的方法，但不能把理论预期、oracle 或同一 Valid 的挑选收益当成可迁移增益。

## 阅读时保留的边界

- 本套文档基于本机已有报告、配置、日志、预测复算报告与源码；没有重新训练或为每个历史 run 重新推理。实时任务状态只是本机同步快照。
- v2/v3 原设计文档曾出现在会话中，但本次在本地未找到原文件；按现存实现报告、运行脚本和后续审计还原，没有假装逐行核对不存在的文件。
- 完整规则以 [比赛规则](../比赛规则.md) 及最新官方信息为准。本套文件不会把历史计划中有合规争议的做法推荐为可提交方案。
- 每个路线的“讨论问题”属于尚待验证的问题，不是已批准、已实施的新实验。
