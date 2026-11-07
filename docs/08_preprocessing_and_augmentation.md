# 路线八：预处理、几何保真、图像质量与增强

核心假设：模型在不同设备、投照、强度和图像尺寸上的泛化，可能被输入过程限制。最重要的历史修正是：**X21 实际使用 contain，不是 stretch。**

## 1. normalization、contain、padding 与 mask

早期对比过 ImageNet normalization、half normalization，ConvNeXt、EfficientNet 与 YOLO 的输入契约不同。YOLO26X曾误用ImageNet norm；后续raw[0,1]修正结果更合理，但仍未成为强模型。

早期U-Net/subject研究加入mask-aware photometric与padding处理；C3 masked spatial pooling .80504，相对C1只有+.00037。X21的Transform采用 `ImageOps.contain`，保持比例后黑边填充。旧报告将它写成stretch、据此解释心胸比失真，是已被源码推翻的诊断。

X47“保持比例+mask感知增强”并非从X21 stretch改contain的干净对照；有首版invalid_contain失败后重跑，峰值.81289。高分辨率+contain+masked_gap仍可以研究，但不能宣称解决了一个X21不存在的问题。

## 2. 标准几何与光度增强

从早期gamma结构配对，到R3/E3的photo/noise/geometry，再到X5完整强正则包，试过hflip、brightness、contrast、rotate、scale、translate、erasing。

structural gamma三seed treatment均值约.793355，配对均值+.00305，协议是image-train/eval-max；不能说它已在后续Study协议中确认。subject_v3 rotate配对约.75973→.76535，masked photo约.76382→.76325，说明不同增强不等价。

E3在E1上加增强，训练日志峰值.80423、hflip等独立复算约.8048；S1通过与E1集成提供收益。X5的成功混合EMA/WD/增强/目标，不能把其全部收益归因于“erasing有效”。

## 3. JPEG、曝光与噪声

- X9 JPEG40–95+noise .01+hflip，Valid .80584；q50损坏下相对小幅改善约.0011，clean中性。没有达到大幅损坏鲁棒性收益预期。
- X23 gamma_jitter .10 +noise .01，在X5完整配方下八轮.80514，低于X21。
- 计划中的完整11类良性损坏与FGSM/PGD审计没有全部接入，不能用X9一项宣称做完鲁棒性全审计。

## 4. 裁剪路线分别做过什么

| 实验 | 裁剪发生在哪里 | 结果 |
|---|---|---|
| S2 | 推理时AP全图+2×2 | Valid.79641，Test.7586，负向 |
| X4 | 训练额外心影中心crop作为同Study视图 | .73248 vs无crop MIL .74117，负向 |
| X57 | contain+上下15%/左右10%心影区域crop | .80929 /heavy .75569，低于X21 |
| X60 | 原图/hflip/上75%/中心85%多crop推理 | .81034 /heavy .75635，未改善 |
| X71 | 底部20%+下方偏置erasing | 代码已实现，未确认有效完成结果 |

X57失败不能解释为“contain无效”；X21本来contain，变化主要在crop。也不能把每个crop负结果外推成任何定位/区域学习都无效。

## 5. CLAHE 与高分辨率组合

X67 448+contain+masked_gap，X69 384+CLAHE，X70叠加三者。CLAHE默认clip2、8×8网格，内容mask保黑边。早期5090计划8ep并有X68/X70/X72/X73门控；后来本地夜间方案明确最多六轮，是不同预算版本。

DGX六轮结果：X67 macro .80089；X69 .80168 /heavy .74959；X70 .79858 /heavy .74897。相对X21无正向证据。X68仅在X67≥基线+.004运行；X72/X73还依赖足够强的新配置，不能把未运行分支写成失败。

## 6. Mixup / 对比学习

- X32 feature mixup把0/7视作稀有类，索引7误认；.80925。X38组合也继承错误，不能证明正确稀有类0/1/2的mixup失败。
- T3只对heavy feature mixup，旧链被取消，没有完整受控完成证据。
- v2 X1 contrastive-heavy full从原始公开权重训练12ep，peak ep6 macro .80610 /heavy .75745，后期回落。
- X33 heavy oversampling+within-heavy contrastive混合，无法分解各自贡献。
- X75 Train-only对比SSL曾编写并进入队列计划，未找到完成监督跟进的结果。规则对医学数据自监督续训有限制，即便只用赛方Train，也不能默认可提交；此处仅保留历史。

## 讨论问题

是否应优先验证“模型是否使用内容mask、缓存是否对应真实预处理、推理是否复用同一流程”，再谈复杂增强？若要改图像质量，是否能提前量化它改变哪些器官/病灶边缘，避免一次同时改分辨率、比例、crop和优化预算？

来源：[输入合约复核](../report/CMS_YOLO_experiment_analysis_20261028.md)、[subject指南](../report/CMS_subject_balance_v3_guide.md)、[X1–10](../report/CMS_X1_X10_implementation_audit_20261030.md)、[X56–60](../report/CMS_X56_X60_codex_impl_20261101.md)、[输入质量计划](../report/CMS_X67_X73_input_quality_plan_20261102.md)、[本地队列](../report/CMS_DGXSPARK_night_queue_X67_X75_20261103.md)、[规则](../比赛规则.md)。
