# 路线二：骨干、模型容量与输入分辨率

核心假设：缺少细节或归纳偏置不合适，可能比损失技巧更限制病灶判别。这里区分“单模型更强”和“弱一点但能贡献互补排序”。

## 1. 通用骨干筛选与小模型阶段

早期试过 DenseNet121、EfficientNetV2-S、ConvNeXt-Tiny/Small、Swin-Small、DINOv2-Base，以及 U-Net 变体。原始日志中代表峰值为 DenseNet121 0.78029、ConvNeXt-Tiny 0.78116、Small 0.78654、EfficientNetV2-S 0.76900。它们的训练配方不完全一致，只能用于描述筛选历史，不能当作公平架构排行榜。

Small@640 的 e6/e7/e8 变体峰值约 0.80060/0.80474/0.80471，说明“大输入”早已试过，不能将后来的 448/512 提案称为首次高分辨率。预处理、预算、损失和池化需逐一对齐。

## 2. ConvNeXt-L 变成主力

encoder-budget 修复从 Small/encoder LR×.1 转向 Large@384、全量 encoder LR、warmup2、layer decay .8。E1 峰值 0.80382，正式固定 ep5 0.78945；E3 增强版复算约 0.8048。之后 X5 正则/EMA 达 0.81128，X21 cosine 在同一完整配方下达 0.81457。

这是一条“容量与配方共同改善”的过程，不能把 E1→X21 的全部增益归给单个部件。

## 3. 更大模型与异构成员

| 实验 | 骨干 / 输入 | 评价与结果 | 能说明什么 |
|---|---|---|---|
| X7 Swin | Swin-S@384 | Valid 0.79343，和 E1 秩相关历史约 .856 | 单模弱但进入四模型包，提供异构性 |
| X7 DINO | DINOv2-Base@384 | 0.76646 | 该配置弱，不能外推到所有 DINO |
| X16 | ConvNeXt-XL@384 | 7ep，峰值 0.80762 ep4 | 容量升级未超过 X5/X21 |
| X18 | DINOv2-L-reg4@518 | 7ep，0.80638 ep5；近期官方 Valid heavy9 0.75751 | 比 Base 强，可作固定融合审计，但低于当前组合 |
| X24 | SwinV2-L@256 | 8ep，0.80763 ep5 | 值得继续作为异构成员 |
| X29 | ConvNeXt-XL + cosine | 8ep，0.80723 ep4 | 不能把 Large 的 cosine 收益直接迁移为 XL 收益 |
| X41 | SwinV2-L + per-class cosine | 8ep，0.81072 ep6；heavy9 .76529；AP-heavy .73806 | 对 X42 集成有效，是 77.66 提交成员 |
| PIV3 | 患者隔离的 SwinV2 AP 专家 | X49 AP .67481 ep5；和 PIV0 相关约 .767 | 低相关存在，但单模和净增益未稳定超过对照 |
| MC01 | 干净 SwinV2-L@256，C 选 ep4 | 官方 macro .80271 /heavy .75496 | 训练边界和预算不同，不与 X41 直接归因比较 |
| MC02 | Small@224 弱配方互补探针 | 官方 .78027 /.72413 | 检验逐类互补，不是追求最强单模型 |

ConvNeXt-XXLarge、ViT-H 等曾因体积预算排除，并无本地完成训练证据。XL / DINO-L INT8 约 335 /294 MB 是部署可行性估算，不是有效性证据。

## 4. 分辨率分支

| 阶段 | 分支 | 结果 / 限制 |
|---|---|---|
| 早期 HR512 | BCE mean /hybrid AUC /AUC+EMA /LSE /BCE+EMA | 日志峰值约 .80210/.80112/.79328/.80869/.81279；image-train，聚合各异 |
| X3 | AP-family@512 对 X1@384 | .74568 vs .74794，未提高 |
| X17 | Large@512 | 10ep 老版 .81081 ep5；7ep 5090 版 .80712 ep5；必须分版本 |
| X59 | 448 + stretch | .81255 /heavy .76441；heavy 对 X21 仅 +.00011；分辨率与几何同时改变 |
| X67 | 448 + contain + masked GAP | DGX 六轮记录 .80089 ep4，未突破 |
| X68 | 512 + contain + masked GAP | 条件门控方案；本次无确认有效完成结果 |
| X70 | 448 + contain + masked GAP + CLAHE | DGX 六轮 .79858 /heavy .74897；远端迁移计划不能替代实际记录 |

X21 实际已经 contain。不能以“纠正 X21 stretch”为 X67 的理论起点；X59 某些小病灶类改善而心脏类下降，是值得检验的现象，不是已完成的因果分解。

## 5. YOLO 作为分类 / 弱监督骨干

试过 YOLO26X global、单阶段 WSL、修正 raw[0,1] 输入与官方几何、真实多尺度 WSL H1/H2/H3。早版错误 ImageNet normalization 与结构实现偏离需单独剔除。后续修正后仍明显落后主力。

历史报告列 H3 末轮约 .72417；原日志机械最大值为 .74529。这里明确区分末轮和峰值，不把该差异写成模型互相矛盾，更不把“报告低值”当整条架构路线的数学上限。无证据支持继续在该实现上大规模投入。

## 讨论问题

新骨干应先证明：在固定人群和匹配训练下，是否提供比当前 X42+X41 更多的净纠错？分辨率实验应否固定 contain、mask、有效 batch 和优化步数，先排除训练动力学变化？目前“更大”缺少收益，“异构”已有收益但接近抵消，二者不能混作同一个论证。

来源：[encoder 复盘](../report/CMS_encoder_budget_v2_retrospective_and_next_stage_20261029.md)、[X1–10 实现](../report/CMS_X1_X10_implementation_audit_20261030.md)、[X16–20](../report/CMS_X16_X20_codex_impl_20261030.md)、[YOLO 分析](../report/CMS_YOLO_experiment_analysis_20261028.md)、[512 协议](../report/CMS_512_method_decision_update_20261027.md)、[X67–73](../report/CMS_X67_X73_codex_impl_20261102.md)、[原始指标清单](14_run_evidence_inventory.md)。
