# U-Net + MIL 改良研究与实现方案

**检索日期：2026-10-24；对象：官方 Train/Valid、Study 级 10 类多标签胸部 X 光任务。**

本研究只讨论可以接入现有合规路径的改良：已经审计的通用视觉编码器、随机初始化 U-Net 解码器和新增 MIL 头，监督只来自官方 Study 级标签。没有引入外部医学图像、医学预训练权重、外部教师、肺部/病灶伪掩码或医学自监督阶段。Test 不参与统计、调参或模型选择。

## 证据与结论

| 改良 | 依据 | 对当前任务的判断 | 首轮状态 |
|---|---|---|---|
| 类别独立 gated attention 空间 MIL | Ilse, Tomczak & Welling, *Attention-based Deep Multiple Instance Learning*, ICML 2018；[arXiv:1802.04712](https://arxiv.org/abs/1802.04712) | 用 `tanh(Vh) * sigmoid(Uh)` 产生每类空间权重，能替代固定 Top-K，并保留可审计的分类证据图 | 已实现，可选 |
| 胸片弱监督空间证据 | *Localization of Critical Findings in Chest X-Ray Without Local Annotations Using Multi-Instance Learning*, ISBI 2020；[DOI](https://doi.org/10.1109/ISBI45749.2020.9098551) | 图像级标签可以端到端训练分类和定位证据，但证据图不能当作分割真值 | 现有 U-Net 路径保持 |
| Attention U-Net 式解码/池化 | *Attention U-Net for Interpretable Classification on Chest X-ray Image*, BIBM 2020；[DOI](https://doi.org/10.1109/BIBM49941.2020.9313354) | 与当前“解码器→空间证据→MIL”结构同源；优先做池化消融，不额外引入外部解剖先验 | 已实现对应池化 |
| ASL 多标签损失 | Ridnik et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021；[arXiv:2009.14119](https://arxiv.org/abs/2009.14119) | 胸片正负标签高度不平衡时，分别处理正/负样本并抑制简单负样本；需与 BCE/`pos_weight` 单因素比较 | 已实现，可选 |
| Train-only `pos_weight` | 多标签长尾训练的通用做法 | 仅用 Train Study 标签计算 `neg/pos`，不读取 Valid/Test 标签统计；适合作为 BCE 对照 | 已实现，可选 |
| Study 内聚合对照 | 多图 Study 的标签只在 Study 级给出 | `mean` 保持基线；`max` 和归一化 `logsumexp` 可作固定聚合消融，不能跨 Study 聚合 | 已实现，可选 |
| 可选冻结、分层学习率、warmup/cosine、梯度裁剪 | 迁移学习的稳定训练实践 | 冻结编码器或使用较低编码器学习率；解码器/新头使用基础学习率；首轮先固定单因素 | 已实现 |
| 轻度强度扰动 | 医学图像增强经验，但任务依赖数据分布 | 仅 Train 使用亮度/对比度/小噪声，默认关闭；不默认水平翻转，不改变几何 mask | 已实现，可选 |
| EMA、证据 TV/一致性、对抗训练 | 常见鲁棒化方向 | 需要单独合规和消融；EMA 可能被误读为 teacher，TV/一致性可能压制小病灶，PGD 先做 Valid 审计 | 暂不作为默认正式路线 |
| 外部肺部分割、医学教师、伪掩码、医学自监督 | 与本项目保守合规约束冲突 | 不接入正式代码路径 | 禁用 |

近期胸片工作还显示显著性图在小病灶和复杂病灶上可能不可靠（Nature Machine Intelligence 2022，[DOI](https://doi.org/10.1038/s42256-022-00536-x)）。因此本项目将 attention/evidence 图作为分类证据诊断，只检查其稳定性、有效区域覆盖率和数值异常，不宣称像素级病灶定位准确。

## 已写入代码

- `unet_mil_extensions.py`：在原模型之外提供 `AdvancedUNetMIL`。支持 `topk`（默认基线）、`mean`、`max`、`logsumexp`、类别独立 `attention` 和 `gated_attention`。所有新增层随机初始化；padding 几何 mask 在池化前屏蔽。
- `train_unet_mil_advanced.py`：单 GPU 实验入口。支持 Study 聚合、Train-only `pos_weight`、分层学习率、warmup/cosine、梯度裁剪、可选 label smoothing 和轻度强度增强；验证指标为 Study Macro-AUC/Macro-F1。
- `run_unet_mil_advanced.sh`：便于 GPU 0/1 分别启动。
- `UNET_MIL_IMPROVEMENTS.md`：运行参数和消融顺序。

## 建议消融顺序

每次只改一个因素，固定 seed、输入尺寸、manifest、验证集和训练轮数；只用 Valid Study Macro-AUC 选项：

1. E0：`topk + mean + BCE`，与旧基线复现。
2. E1：只把空间池化改为 `gated_attention`。
3. E2：回到 E0，仅启用 Train-only `pos_weight`，或单独换 ASL。
4. E3：在最佳单项上启用编码器低学习率倍率、warmup、梯度裁剪。
5. E4：单独比较 Study `max`/`logsumexp`；若没有稳定提升，保留 mean。
6. E5：最后才评估轻度强度扰动。

正式 Test 推理前，冻结空间池化、Study 聚合、预处理、阈值、精度和 checkpoint。不要根据 Test 结果回调参数。

## 风险边界

- attention/evidence 权重是弱监督分类证据，不是分割掩码。
- `pos_weight`、ASL、label smoothing 不能同时叠加；比较时记录完整配置。
- 水平翻转默认关闭，避免改变胸片左右侧别语义。
- 外部医学先验、教师模型、伪标签和医学自监督不进入训练。
- 任何后续 EMA、TV、consistency 或 FGSM/PGD 训练都应先在 Valid 上做独立审计并记录是否改变正式路线。

## 参考文献

1. Ilse M, Tomczak JM, Welling M. Attention-based Deep Multiple Instance Learning. ICML 2018. https://arxiv.org/abs/1802.04712
2. Ridnik T, et al. Asymmetric Loss For Multi-Label Classification. ICCV 2021. https://arxiv.org/abs/2009.14119
3. Localization of Critical Findings in Chest X-Ray Without Local Annotations Using Multi-Instance Learning. ISBI 2020. https://doi.org/10.1109/ISBI45749.2020.9098551
4. Attention U-Net for Interpretable Classification on Chest X-ray Image. BIBM 2020. https://doi.org/10.1109/BIBM49941.2020.9313354
5. Saliency benchmarking for chest radiographs. Nature Machine Intelligence 2022. https://doi.org/10.1038/s42256-022-00536-x
6. Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065v2 (2026). https://doi.org/10.48550/arXiv.2609.00065

本报告采用 literature-review skill 的可复现检索记录方式；检索摘要和 DOI/URL 需在正式论文或提交材料前再次人工核对。
