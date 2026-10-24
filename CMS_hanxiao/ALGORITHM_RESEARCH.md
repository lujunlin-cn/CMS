# U-Net + MIL 改良研究与落地边界

本文针对当前胸部 X 光 10 类 Study 级多标签任务，研究能够直接接入现有
U-Net + 空间 MIL 的方法。所有建议都以 `unet_mil_compliance_guide.md` 的
合规边界为前提：只使用官方 Train/Valid 的 Study 标签监督；不加载外部医学
权重、分割器、伪掩码或教师模型；Test 不用于拟合统计量、选择参数或训练。

## 方法结论

### 1. 先做可学习 MIL 聚合

现有 masked Top-K mean 对每个类别选择最高响应的空间位置，优点是简单、可
解释、无新增权重，缺点是硬 Top-K 可能在早期训练造成梯度稀疏。建议将聚合
实现为固定 registry，并逐项比较：

- `masked_topk_mean`：保持当前基线；
- `masked_logsumexp`：以固定温度 `tau` 平滑近似 max；
- `GatedAttentionMIL`：令
  `a_i = softmax(w^T[tanh(Vh_i) * sigmoid(Uh_i)])`，对每个类别使用独立
  attention score，再对空间 token 加权；
- `masked_mean`：作为结构消融的下界。

Gated attention 来自 Ilse 等人的 Attention-based Deep MIL。它的新增参数从
随机初始化，只通过官方 Study 标签训练，符合当前路线；其 attention 图只
表示分类证据，不能解释为病灶分割。

多图 Study 目前按图像 logits 求均值。若预算允许，可在同一固定编码器下增加
Study 内的 class-specific gated image pooling，但必须在 Train/Valid 上固定，
且不能把 Test 预测分布用于选择池化形式。

### 2. 处理类别不平衡

BCEWithLogitsLoss 作为主基线保留。ASL（Asymmetric Loss，Ridnik 等）通过负类概率裁剪
和正负类不同的 focusing gamma，适合正例稀少的多标签任务。建议单因素运行
ASL（默认 `gamma_neg=4`, `gamma_pos=1`, `clip=0.05`），不要同时叠加
`pos_weight`，否则无法判断改进来源。

### 3. 训练策略

- 解码器和证据头先训练 1–2 个 epoch，再按较低学习率解冻已核验的通用编码器；
  编码器学习率可设为 head 的 0.1 倍；
- AdamW、cosine scheduler、梯度裁剪和 AMP 可作为固定工程设置；
- 每个实验保存随机种子、配置、manifest 哈希、Study 级 Macro-AUC、每类
  AUC 和 Macro-F1；
- 以单因素消融比较，推荐顺序为 `BCE+Top-K` → `BCE+LSE` → `BCE+Gated`
  → `ASL+Gated`。不重新开启编码器选型。

指南明确要求首版关闭教师—学生、医学自监督、重建损失、分割辅助损失和
伪掩码。Mean Teacher、外部 anatomy prior、ROI proposal、对抗训练等只能
作为后续经合规审查的研究分支，不能混入首版正式训练。

### 4. 校准和鲁棒性

训练完成后可仅用 Valid 拟合一个全局 temperature，或在每类样本量足够时拟合
受约束的 per-class temperature；保存 JSON 并在 Test 前冻结。校准不改变 AUC，
但可改善概率质量和固定阈值下的 F1。FGSM/PGD 先作为 Valid 上的审计，不把
攻击结果用于修改 Test 参数；若将对抗样本混入训练，必须单独记录为后续实验。

## 代码

`unet_mil_extensions.py` 提供：

- 空间 mask 安全的 Top-K 和 LogSumExp 池化；
- class-specific `GatedAttentionMIL`；
- ASL 多标签损失；
- 默认权重为 0 的 TV/L1 evidence 正则（无像素标签，不是分割损失）。

`smoke_test_unet_mil_extensions.py` 在 CPU 随机输入上检查空 mask、有限值和
梯度路径。把模块复制到远端工作目录后，在现有训练器中通过显式配置开关接入，
默认保持 BCE + masked Top-K 基线。

## 参考文献

1. Ilse, J. et al. *Attention-based Deep Multiple Instance Learning*. ICML 2018.
   <https://proceedings.mlr.press/v80/ilse18a.html>
2. Ridnik, T. et al. *Asymmetric Loss for Multi-Label Classification*. ICCV 2021.
   <https://openaccess.thecvf.com/content/ICCV2021/html/Ben-Baruch_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html>
3. Shao, Z. et al. *TransMIL: Transformer based Correlated Multiple Instance Learning
   for Whole Slide Image Classification*. NeurIPS 2021. <https://arxiv.org/abs/2106.00908>
4. Li, B. et al. *Dual-stream Multiple Instance Learning Network for Whole Slide Image
   Classification*. CVPR 2021. <https://arxiv.org/abs/2012.08974>
5. Lu, M. et al. *Data-efficient and weakly supervised computational pathology on whole-
   slide images*. Nature Biomedical Engineering 2021. <https://arxiv.org/abs/2004.09666>
6. Oktay, O. et al. *Attention U-Net: Learning Where to Look for the Pancreas*.
   <https://arxiv.org/abs/1804.03999>
7. Guo, C. et al. *On Calibration of Modern Neural Networks*. ICML 2017.
   <https://proceedings.mlr.press/v70/guo17a.html>
8. Madry, A. et al. *Towards Deep Learning Models Resistant to Adversarial Attacks*.
   ICLR 2018. <https://arxiv.org/abs/1706.06083>
9. Tarvainen, A. & Valpola, H. *Mean teachers are better role models*.
   NeurIPS 2017. <https://arxiv.org/abs/1703.01780>
10. *Weakly Supervised Object Detection in Chest X-Rays with Differentiable ROI Proposal
    Networks and Soft ROI Pooling*. <https://arxiv.org/abs/2402.11985>

