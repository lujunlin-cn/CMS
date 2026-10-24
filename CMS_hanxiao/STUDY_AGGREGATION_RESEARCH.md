# Study 多图聚合补充研究

检索与数据核查日期：2026-10-24。

## 数据事实

当前 manifest 的 Study 图像数分布如下：

| split | 图像 | Study | 平均图像/Study | 单图 | 双图 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 30,344 | 18,513 | 1.639 | 8,507 | 8,369 | 7 |
| Valid | 1,422 | 859 | 1.655 | 391 | 380 | 4 |
| Test | 2,218 | 1,414 | 1.569 | 775 | 496 | 5 |

Test 单图 Study 的比例高于 Train，因而带有图像数偏置的 `max` 或未归一化的 smooth-max 可能产生分布变化。Train/Valid/Test 的 ViewPosition 也不一致：Train AP 12,152/30,344，Test AP 1,090/2,218。不要把 AP、PA、LATERAL 或 LL 手工赋予阳性权重。

## 方法证据

- CheXFusion：Kim, *CheXFusion: Effective Fusion of Multi-View Features using Transformers for Long-Tailed Chest X-Ray Classification*, ICCV CVAMD 2023，DOI [10.1109/ICCVW60793.2023.00285](https://doi.org/10.1109/ICCVW60793.2023.00285)。其摘要描述用 self/cross-attention 融合 Study 多视图并利用标签共现。该方法说明可学习多视图融合有价值，但其外部 MIMIC-CXR-LT 数据、长尾自训练流程不进入本项目。
- Set Transformer：Lee et al., ICML 2019，[arXiv:1810.00825](https://arxiv.org/abs/1810.00825)。提供置换不变的 set attention。当前 Study 平均只有约 1.6 张图，深层 Transformer 容易过拟合，放在后续实验。
- Deep Sets：Zaheer et al., NeurIPS 2017，[arXiv:1703.06114](https://arxiv.org/abs/1703.06114)。`rho(sum(phi(x_i)))` 是低参数置换不变基线；当前固定 mean 聚合已经接近其无参数特例。

## 结论

1. `Study mean` 保持为主基线，按 logits 聚合，Study 内不跨患者、不跨 Study。
2. `max` 和归一化 `logsumexp` 只做固定参数对照；同时报告单图、双图和至少三图子集的 Macro-AUC。
3. 若验证集确实显示多图融合收益，再实现一层 class-wise image attention：从每张图的 decoder 全局特征生成每类 score，按真实图像数 mask 后 softmax；该层随机初始化，只接受 Study 标签监督。
4. 图像顺序来自 CSV，不能当作位置编码；聚合必须对 Study 内图像置换不变。
5. ViewPosition embedding 暂不作为默认方案，因为 Train 到 Test 的视角比例变化明显且存在空值。

所有聚合超参数只用 Train/Valid 确定，Test 只执行冻结后的推理。
