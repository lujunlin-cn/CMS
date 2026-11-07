# 路线十：软标签、OOF、噪声重权重与蒸馏

核心假设：教师能为不完整Study标签提供软监督、将区域/视角/异构集成的知识压缩进单模型。但教师的真实新信息、评价隔离与目标一致性必须先成立。

## 1. 从 OOF 噪声权重到软目标

早期D/R的D4受OOF hard-lock阻塞；encoder命名空间的D4是另一实验：D2+三折OOF噪声重权重，峰值.80116，未超过D2 .80256。两个D4不能合并。

X14将OOF聚合、条件化和可疑阴性软化接入训练，历史有路径映射/文件使用疑点。4090最终七轮macro .73204；X28显式核查并修正CSV路径、oof_alpha=.15，首轮.72616触发停止。路径修复没有恢复能力，所以不能把所有低分继续归因于“CSV不存在”。至少需要核查Study join、类序、正负软化和loss尺度。

部分旧报告按错类别名解释class0/3/7近随机，机制解释必须用官方映射纠正。

## 2. v3 的 view teacher 链

```text
X84 fold0 baseline
  → X85 AP teacher / X86 LAT teacher
  → 生成 view soft targets（流程阶段，相当于X87，无独立训练分数）
  → X88 soft-label student
  → X89 region model
  → X90 region-to-global student
```

X85/X86的原日志peak .75158/.81234是在不同视角子集，不是全Valid架构排名。X88正确内部fold Study .80093 /heavy .70807，低于X84 .80693/.72418。

关键实现偏离：教师输出后来合成Study一行软标签，再复制给每张训练图；没有真正让各视角承担不同可见性，也没有按Study bag计算软监督。教师边界没有形成患者隔离OOF闭环。

## 3. X89→X90 的“蒸馏新高”为什么不成立

X89实际绝对feature top-k +region max，与原attention设计不同；推理丢region head。X90的temperature flag没有进入损失，实际是sigmoid teacher概率的BCE，不是宣称的temperature-scaled KL。多标签本来也不应使用互斥十类softmax KL。

图像级高分不能和Study级基线比较。统一内部fold Study结果为X89 .81100/.73482、X90 .81194/.72666。X90最终Test .7662，低于既有.7766；这说明“图像级突破”不是可兑现的泛化突破，不需要凭空假设官方类别顺序变了。

## 4. X91→X92→X93 的版本史

| 版本 | 实际行为 | 证据判断 |
|---|---|---|
| X91五fold | Study StratifiedKFold，训练多个同架构教师 | 非患者group隔离；五个教师中每Study被四个见过 |
| 初版X92 | ensemble soft targets，学生用recovered_train_all | 包含全部3703验证Study/6114图，直接泄漏，0.85无效 |
| X92延长8/10ep | 延长蒸馏并衰减LR | 错误数据边界未修前，加轮不提供泛化证据 |
| 后来“strict”X92 | 清除Study/DICOM交集，预算5ep，实际停ep4 | 仍共享1594患者；仅Study-held-out |
| X93重定义 | X92+新增region head+hard BCE/heavy权重 | 没有实际集成X89/X90知识；原soft teacher anchor被丢掉，且原评价泄漏 |
| 旧X92/X93量化命名 | 早期仅导出INT8的方案 | 后被用户要求覆盖语义；不与正式蒸馏/region run混同 |

严格版Study Macro ep1–4 .81370→.81477→.81567→.81644；第四轮仍涨不等于值得继续，因为它不是患者独立验证。旧heavy值还受bucket与10/9类错误影响，不能再引用。

把五fold教师全平均的训练目标不等同于OOF；每Study只用没见过该患者的教师，才可称患者OOF。另一方面，训练内教师预测可用于普通蒸馏，本身不自动等于验证泄漏；问题在于将其包装成OOF及在见过的验证上评估。

## 5. 异构蒸馏：目前只做可行性审计

已对PIV教师组合和当前最佳X42+X41做预测审计，未据此新训异构学生。

- PIV0+PIV3对PIV0：X49 AP约+.0076点估计，官方AP约+.0053，区间跨零。
- 对当前最佳B，AP-single加入PIV3：官方heavy从.77277到.77411，仅+.00133，区间约[-.00739,.00937]。
- 当前最佳本身已是异构组合；“换成异构蒸馏”不能自动再次获得第一次异构收益。
- PIV1为图像token attention，不是空间病灶教师；共享失败占主导。

旧X92脚本不适合直接改教师名单重跑：它逐图训练、不同骨干吃同一学生tensor、Study教师接口不匹配，且验证时有EMA覆盖raw未恢复的问题。离线Study软目标、各教师独立尺寸/归一化、raw/EMA分离、匹配hard-label继续训练对照都是必要前提。

## 讨论问题

目标究竟是压缩现有77.66组合，还是超过它？若教师本身只有+.001的未确认新增收益，先训练一个更复杂学生是否值得？在患者隔离且Study对齐的协议下，学生能否保留现有异构净纠错，而不是蒸馏共同错误？

来源：[OOF与X28实现](../report/CMS_X21_X30_codex_impl_20261031.md)、[X93失败](../report/CMS_X93_failure_analysis_20261105.md)、[泄漏审计](../report/CMS_X90_X92_leakage_generalization_audit_20261106.md)、[v3重审](../report/CMS_heavy9_v3_reassessment_20261106.md)、[异构可行性](../report/CMS_heterogeneous_distillation_feasibility_20261106.md)。
