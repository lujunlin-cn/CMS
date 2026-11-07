# 路线六：损失函数、排序学习与困难阴性

核心假设：普通 BCE 可能过度使用“异常 vs 正常”信号，而未学会在不同异常之间区分某一种疾病；直接优化困难正负排序是否更符合 Macro-AUC？

## 1. BCE、ASL 与 bootstrapping

早期试过 BCE、ASL gamma_neg=2/4、正例权重、有效样本权重、bootstrap target。Small 早期 ASL 峰值约 .7769/.7776，低于不同配方的 Small 主力，不能当严格单变量结论。

C7 bootstrap .80466 与 C1 .80467 几乎相同。X5 把 alpha=.02 bootstrapped 与 EMA、weight decay、增强一起组成强配方，.81128；因此 X5 成功不能全归于 bootstrap。

X12 HN gamma2 最高 .80896 ep4，后期显著退化；X21 在强配方上加 cosine 后 .81457。X64 将 heavy 样本损失额外加 .5（有效1.5倍），从 X21 热启动三轮，最好 ep1 .81034 /heavy .75968，低于基线。

## 2. 非负 PU / 部分标签假设

E6 nnPU 是早期尝试，但和当时弱配方混杂；X22 在 X5 完整配方下改 loss=pu，八轮 .80620 ep6，未超过 X21。PU 对缺标有合理动机，但风险估计和标签先验不自动适合此数据。

不能从“报告可能未提及疾病”直接推出所有零标签应软化；真实阴性、未观测阳性和视角不可见是不同问题。

## 3. Rank / AUC 分支已经多次尝试

| 分支 | 训练目标 | 结果 / 口径 |
|---|---|---|
| highres R0/R1 | BCE 对全局 rank | 内部 dev seed24 .77900→.78463；seed25 只两轮，未确认 |
| HR512 hybrid /AUC+EMA | 在512配置更换目标 | .80112/.79328，低于同族 BCE+EMA .81279 |
| C2 | pairwise AUC 主损失 | .79584，低于 C1 .80467 |
| C2b /C2c | BCE + .05/.10 rank | .80430/.80355，未超 C1 |
| H15 | BCE→AUC | .79029，接近 Small control .79008 |
| H16/H17 | 低 rank /AUC early stop | .79114/.79015，未突破 |
| EB | balanced + hybrid AUC +弱骨干配方 | .76227，多变量负结果 |
| X13 | patient lookup + within-patient pair rank | .80739，未超过强基线 |
| X54 | 全局 AUC queue + margin +偏采样 | .81064 ep2，heavy约 .7574；不是无限 batch 的保证 |
| X65 | X21 + heavy-only .05 pairwise | .81073 /heavy .75837 ep1，未获益 |

小 batch 中稀有类没有正负对、同患者相关、把未标注当负例、优化易配对而非目标错序，都可能降低统计效率。它们是待验证机制，不能只因 AUC 是排名指标就认为 rank loss 必然更好。

## 4. Abnormal-negative（AbnNeg）软目标

X35 设计对“本类阴性、其他异常阳性”的目标加大软化 alpha，从 .02 到 .15；代码计算后的目标被丢弃，最终仍使用原 loss。其输出与 X21 逐位相同，是 flag 无效的证据，不是方法等价或无效的证据。X38 的 AbnNeg 也未生效。

X45 修复实际 loss 路径，增加 flag 改变 loss 的检查。原版八轮 peak .81401，未超 X21；after_x55_clean 另一版 .81149。修复后仍没有显示突破，但两个版本不能混成“一个唯一 X45 分数”。

MC12 原计划在 Study batch 内最多25%条目构造相似异常匹配，目标类别3/5/6/7；先需要空间 residual 在 C 上成立。门控未通过，因此该具体“有界空间 residual +匹配异常”方案未训练，不能用旧 global queue 负结果宣称它也失败。

## 5. SAM 与 adversarial

X15 SAM 双步 +PGD/adversarial，七轮峰值 .79424；显存、小 batch 和多变量使它不是纯 SAM 的公平证据。X39 batch32 单独 SAM 八轮 .80733，仍低于 X21。继续增加 SAM 预算尚无支持。

T1 heavy focal、T3 heavy feature mixup 曾接入旧 v2 链，后来用户取消以跑 X1/A 系列。仅有旧队列和中断痕迹，不能当成已完整验收的六轮结果。

## 当前认识

损失路线没有未经探索的大空白；重复普通 BCE→rank/PU/重权重的合理性很低。真正尚缺的是把损失引起的变化追到具体错序：同视角/跨视角、同异常背景、不同患者的正负 margin，是否被改善，还是只提高了疾病共现概率和严重程度分数？

讨论问题：在能保留基线的有限 residual 中，对真实可区分的异常阴性构造小规模匹配监督，是否比对整个 backbone 改 loss 更安全？其收益门槛应是患者独立的净错序下降，而非 training loss 或 hn9 单项新高。

来源：[损失研究](../report/CMS_loss_research_v2_20261028.md)、[C/H 历史总结](../report/CMS_experiment_summary_and_opus_request_20261029.md)、[X11–15](../report/CMS_X11_X15_codex_impl_20261030.md)、[X31–42 纠错](../report/CMS_X31_X42_deep_analysis_20261101.md)、[X64–66](../report/CMS_X64_X66_heavy9_followup_20261101.md)、[MC 协议](../report/CMS_medium_class_4090_experiment_plan_20261106.md)。
