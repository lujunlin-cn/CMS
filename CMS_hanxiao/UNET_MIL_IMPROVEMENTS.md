# U-Net + MIL 可选改良（研究代码说明）

`unet_mil_extensions.py` 与 `train_unet_mil_advanced.py` 是基线之外的显式实验入口。它们沿用已审计的通用编码器、随机初始化 U-Net 解码器和官方 Study 标签监督，不读取 Test，不生成伪掩码，不接入外部医学数据。

默认参数保持基线语义：空间 `topk`、Study `mean`、普通未加权 BCE。`pos_weight`、ASL、注意力池化、增强等均是可选单因素实验。

## 可用改良

- `--spatial-pool topk|mean|max|logsumexp|attention|gated_attention`：空间证据聚合。`gated_attention` 使用每类独立的 `tanh(Vh) * sigmoid(Uh)` 权重，权重只在几何有效区域内 softmax。
- `--study-pool mean|max|logsumexp`：同一 Study 内固定多图聚合；默认 `mean`。
- `--use-pos-weight`：仅从 Train manifest 计算 `neg/pos`，用于 BCE；默认关闭。与 ASL 不同时启用。
- `--loss asl`：ASL（默认 `gamma_neg=4, gamma_pos=0, clip=0.05`）替代 BCE；与 `--use-pos-weight` 和 label smoothing 分开做消融。
- `--encoder-lr-multiplier 0.1`、`--warmup-epochs 1`、`--grad-clip 1.0`：迁移训练稳定化选项；高级入口默认值保持原基线（倍率 1.0、无 warmup、无裁剪），启用时作为独立训练侧消融。
- `--brightness`、`--contrast`、`--noise`：仅 Train 的轻度强度扰动，默认全关闭；验证阶段永不启用。
- `--deterministic`：关闭 cuDNN benchmark 并请求确定性算法，适合正式消融复现（吞吐可能下降）。

## 推荐消融

1. E0：`topk + mean + BCE`，确认与原基线一致。
2. E1：只改 `--spatial-pool gated_attention`。
3. E2：回到 E0，只加 `--use-pos-weight`；另起一组只用 `--loss asl`。
4. E3：在最佳单项上固定分层学习率、warmup、梯度裁剪。
5. E4：固定其它参数，比较 Study `max`/`logsumexp`。
6. E5：最后评估小幅亮度/对比度/噪声扰动。

每次只改一个因素，固定 seed、manifest、输入尺寸、训练预算和 Valid。只用 Valid Study Macro-AUC 选模型。Test 前冻结所有池化、预处理、阈值和 checkpoint。

注意力/evidence 图是弱监督分类证据，不是病灶分割真值。外部医学分割器、医学教师、伪标签、医学自监督、PGD/FGSM 训练和教师-学生一致性不在首版正式路径内。

- `--study-pool attention`：可选的 Study 内类别独立 attention；只使用每张图 logits，按真实图像数归一化，保持图像顺序不变性。`--study-attention-hidden` 控制隐藏维度，默认 32。
- 当前数据平均每个 Study 约 1.6 张图，且 Test 的 AP 比例有偏移，因此不使用手工视角权重或 ViewPosition 编码。

Study attention 单独作为 E6 消融，建议按 Study 图像数 1、2、≥3 分层记录 Macro-AUC；它没有稳定提升时保留 mean。

## 后处理工具

- `export_valid_logits.py`：从 advanced checkpoint 生成官方 Valid Study logits/targets/Study IDs，仅打开 `valid_studies.jsonl`。
- `calibrate_multilabel.py`：在导出的 Valid logits 上逐类拟合正温度，输出 NLL、Brier、ECE 和温度；温度固定后才可用于最终 Test 推理，不可用 Test 回调。

训练日志现在额外记录类别 1–3 的尾部 Macro-AUC 和类别 8–10 的头部 Macro-AUC，用于判断 ASL/`pos_weight` 是否只改善多数类；模型选择主指标仍是完整 Study Macro-AUC。
