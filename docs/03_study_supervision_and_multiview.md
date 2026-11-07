# 路线三：Study 级监督、多图聚合与视角建模

核心问题：用一份 Study 标签监督多张图，是否重复计算监督、把不可见标签强加到某张图，或丢失了多视图证据？

## 1. 从逐图训练到 Study 原生训练

早期 structural_v1/v2 比较 image-train/eval-max 与 study_native。study_native treatment .79905，image control .79194，单图约 .73522 vs .71861，多图 .83352 vs .83617。说明当时 Study 训练的增益主要不来自多图更强。但该训练后期 NaN，两个 seed 确认也在 ep4 左右失败，不能把峰值当成稳定可重复的完成结果。

mean_prob、mean_logit、max、LSE、attention 曾分别试验。对单图 Study，普通聚合退化为单张图输出；换聚合公式不会凭空产生另一张侧位片的证据。相对优势必须看完整 Study 与子群两端。

## 2. C 系列：把监督和聚合拆开

| 实验 | 变化 | 历史官方 Valid Study Macro |
|---|---|---:|
| C1 | Study BCE + mean_prob | .80467 |
| C3 | mask-aware spatial global pooling | .80504 |
| C4 | view dropout | .80341 |
| C5 | class-token Study attention | .80790 |
| C7 | bootstrapped target | .80466 |

C5 是研究 Study 内类别条件证据选择的动机，但一个候选峰值不能证明所有后续 attention 都会有效。某些 C 目录是失败/早停副本，不能拿其几轮局部日志替换正式复算。

## 3. 视角 embedding、一致性和不同融合机制

- encoder D2：视角独立头 + 对称 KL，峰值 .80256；AP 单图未超 E1。
- E5：class-token + view-position，FP16/BF16 均有非有限问题，原结果不判方法输赢。
- X2：AP 专家路由全体约 .8061，通用包 .8102；专家本身弱，路由没有挽救。
- X4：同 Study 加心影中心裁剪作为视图，.73248；无裁剪 MIL 对照 .74117，负向。
- X25：用完整 X5 配方重做 class-token+view-position，7ep .80566，低于 X21。
- X52：完整配方的 Study mean_prob，.81138 /heavy 约 .7602；未超过 X21。
- X63：逐类 attention + noisy-OR，NoFinding 另用均值，.81056 /heavy .75727，峰值 ep1；未获益。
- A1：cross-view attention。Full 原始预训练六轮 .81021 /heavy .76415；正确 X21 Freeze 三轮 .81519 /heavy .76526。小幅候选改善，缺确认性增益。

## 4. 多视图“特权信息”前置测试

X53 比较全视图和单正位，AUC 约 .8491 vs .8449，差 .0042，低于预设 .01 门槛，未进入复杂的 OOF 多视图教师—学生链。这个结果只说明该模型、选定人群与指标下，没有达到继续投入的门槛，不能推导“侧位没有临床信息”。

## 5. PIV1 的信号与真实机制

PIV 用从未见 X49 患者的祖先权重：PIV0 对照、PIV1 Study class-token evidence。X49 AP-primary 为 .67545 vs .68444，差 +.00898，患者簇区间约 [-.00815,.02591]；官方 AP 却从 .74350 降到 .74030。信号局部、未跨域稳定。

源码复核发现 PIV1 的 token 是每张图，不是图内 patch。AP-single 只有一个 key，attention 权重恒为 1；共享 residual classifier 的实际补偿不能被解释为“每类病灶区域被定位”。它可能改变图像表示/训练轨迹，但不等于获得了新局部病灶证据。

随后 R0 与 RB 做六轮匹配比较：官方 macro .80500/.80470，heavy .75608/.76075。RB heavy 高于 R0，但没有稳定放大 PIV1 的原始主指标。R1 冻结 ImageNet失败不说明“冻结 PIV1 也失败”，因为不是同一初始化。

## 6. 仍待回答

Study 模式已被多次试过，不能把“改成 Study”本身列为全新方向。但下面的问题尚不能被旧结果关闭：保留全局 mean path 的有界、零初始化空间 residual，是否比图像 token attention 更能学习 AP 单图证据？监督是否真正按 Study 一次，而不是事后平均图像指标？冻结线是否同时保留了正确预处理、旧 head 与祖先隔离？

来源：[早期监督](../report/CMS_supervision_signal_research_20261027.md)、[C 系列](../report/CMS_C_series_single_analysis_high_risk_plan_20261028.md)、[X51–55](../report/CMS_X51_X55_codex_impl_20261101.md)、[X63](../report/CMS_X61_X63_codex_impl_20261101.md)、[A 修复](../report/CMS_heavy9_v2_repair_audit_20261104.md)、[PIV 结论](../report/CMS_PIV0_PIV3_analysis_20261106.md)、[PIV 机制复核](../report/CMS_heterogeneous_distillation_feasibility_20261106.md)。
