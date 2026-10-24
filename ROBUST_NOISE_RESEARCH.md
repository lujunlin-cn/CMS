# 标签噪声、校准和对抗鲁棒性补充研究

检索日期：2026-10-24。范围限定为官方 Train/Valid Study 标签，不使用 Test 统计，也不引入外部医学数据或医学教师。

## 直接相关证据

- Ai, Liao, Xia, **GLANCE: Combating Label Noise Using Global and Local Noise Correction for Multi-label Chest X-Ray Classification**, UNSURE 2024/LNCS 15167, DOI [10.1007/978-3-031-73158-7_18](https://doi.org/10.1007/978-3-031-73158-7_18)。全局模块从标签共现估计噪声转移矩阵；局部模块用历史预测 temporal ensembling 形成实例级伪标签。方法与胸片任务直接相关，但局部模块属于 teacher-like 伪标签链，当前比赛保守约束下不进入首版。
- Chen et al., **BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification**, ICCV 2023, DOI [10.1109/ICCV51070.2023.01946](https://doi.org/10.1109/ICCV51070.2023.01946)。通过语言模型语义描述进行噪声重标；外部语义模型链不符合当前首版约束。
- Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017, [PMLR](https://proceedings.mlr.press/v70/guo17a.html)。温度缩放适用于固定 logits 的后验校准。当前任务应在 Valid Study logits 上逐类拟合正温度，报告 classwise ECE、Brier、NLL；冻结后再做 Test 推理。
- Chen et al., **Dynamic Correlation Learning and Regularization for Multi-Label Confidence Calibration**, IEEE TIP 2024, DOI [10.1109/TIP.2024.3448248](https://doi.org/10.1109/TIP.2024.3448248)。标签相关性有助于校准，但训练期标签图会增加参数和验证负担，后置。
- Xu et al., **Noisy Label Calibration for Multi-View Classification**, AAAI 2025, DOI [10.1609/aaai.v39i20.35485](https://doi.org/10.1609/aaai.v39i20.35485)。跨视图共识和邻域置信度可用于多图 Study，但噪声检测与伪标签流程复杂，后置。

## 当前代码缺口

1. `train_unet_mil_advanced.py` 已有 ASL 和 Train-only `pos_weight`，但没有校准后处理。
2. 现有 `adversarial_audit.py` 面向旧的 image-level baseline；它按单图标签攻击，不能代表 U-Net + MIL 的 Study-level 鲁棒性。
3. 当前没有 Study-level clean/FGSM/PGD Macro-AUC、Macro-F1、Brier 或阈值翻转报告。

## 推荐落地顺序

1. 增加独立 `calibrate_multilabel.py`：Valid 前向保存每个 Study 的 logits/targets；逐类优化 `T_c>0` 的 BCE；保存温度、ECE、Brier 和 NLL。该脚本不读取 Test。
2. 改写鲁棒性审计：对同一 Study 的所有图像共同求聚合 BCE，再对拼接图像做 FGSM/PGD；报告 clean/attack 的 Study 级指标。先审计，不做对抗训练。
3. 低风险噪声实验可选 bootstrapped BCE：`y'=(1-β)y+β sigmoid(logit.detach())`，β 从小值开始；与 BCE/ASL 单独消融，默认关闭。
4. GLANCE 的噪声转移矩阵、历史伪标签、BoMD 语言模型描述和 teacher-student 一致性保留为论文参考，不进入正式首版。

对抗训练会增加计算时间并可能损失 clean AUC；应在鲁棒性审计确认问题后再考虑。标签噪声候选也不能把模型预测自动当作新的官方标签。
