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

## 第二轮研究：Study 多图聚合

CheXFusion（ICCV CVAMD 2023，[DOI](https://doi.org/10.1109/ICCVW60793.2023.00285)，预印本 [arXiv:2308.03968](https://arxiv.org/abs/2308.03968)）把同一 Study 的多视图作为集合做 self/cross-attention，并使用真实图像数 mask。它证明多视图融合是独立于单图骨干的研究变量，但其完整 Transformer、位置编码和外部比赛流程超出当前首轮范围。

本数据的 Study 平均只有约 1.6 张图，Train/Valid/Test 分布分别约为 1.639、1.655、1.569 张；Test 的 AP 比例高于 Train。因此不使用手工 AP/PA 权重或 ViewPosition 先验，也不把图像顺序当作位置。新增的 `StudyAttentionPool` 是一层置换不变、类别独立的 attention，只读取每张图的 logits，并按真实 Study 图像数 softmax；参数随机初始化，监督仍是官方 Study 标签。默认仍使用 mean。

Study attention 应按 Study 图像数 1、2、≥3 分层报告 Macro-AUC，确认收益来自多图融合而不是单图分布变化。若 attention 只在少量多图 Study 上提升，正式方案可以继续使用 mean 以降低过拟合风险。

第二轮实验顺序：

- E6：固定最佳空间池化和损失，只将 Study `mean` 改为 `attention`。
- E7：固定其它配置比较 `max` 与归一化 `logsumexp`。
- E8：若 E6 稳定提升，再考虑低层数 Set Transformer；深层 Transformer、ViewPosition 编码和标签图残差暂不进入正式路线。

Train 标签正例比例约为 `[3.36%, 4.86%, 5.09%, 7.76%, 12.99%, 20.99%, 21.43%, 24.35%, 25.14%, 36.10%]`，长尾确实存在，这支持 ASL 或 Train-only `pos_weight` 的单因素实验。标签共现 lift 可作为后续 CheXGCN 风格残差的研究依据，但不能把共现关系写成硬规则，也不能使用 Valid/Test 估计图结构。

近期噪声标签方法（GLANCE、BoMD）常使用 temporal ensembling、伪标签或外部语义描述；这些会改变监督来源，当前合规路径不接入。校准可以在最终模型固定后仅用 Valid 拟合 temperature，并报告 ECE/Brier；它属于提交后处理，不参与训练或 Test 参数选择。

Train Study 约 46% 为单图、45% 为双图、9% 为三图以上；Test 约 55%、35%、10%。因此 `max`/`logsumexp` 可能把图像数量差异放大，所有非均值 Study 聚合都要按 Study 图像数分层评估；mean 继续作为稳健默认。Query2Label/ML-Decoder 的类别 query 交叉注意力可作为更高成本的后续对照，首轮不替换现有证据头。

## 近期 2025–2026 方向的取舍

CXR-LT 相关工作继续集中在长尾损失、query decoder 和多视图融合。2026 年的 LDAM-DRW、query decoder/MoE 和 momentum anchor 方向对低频类别有启发，但它们依赖新的训练阶段、教师式 EMA 或更复杂的分类器重训，尚未在本赛数据和指标上验证。当前先保留 ASL、Train-only `pos_weight` 和轻量 Study attention；复杂 query decoder、LDAM-DRW、对比学习、TTA 和 momentum teacher 作为后续研究，不进入首轮正式路线。

2026 CXR-LT 长尾研究提示 ASL 的收益依赖长尾程度和 `gamma`；在某些极端长尾设置中 ASL 反而压低稀有阳性梯度。本数据类 1/类 10 正例比例约 10.7 倍，低于极端长尾场景，因此 ASL 不能默认优于 BCE，必须同时记录每类 AUC，特别是尾部类别。
