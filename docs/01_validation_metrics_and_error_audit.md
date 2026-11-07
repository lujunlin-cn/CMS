# 路线一：验证协议、指标重构与错误证据审计

核心问题：一个更高的分数，是否意味着对未见患者、正确提交单位和目标人群更好？本项目多次发现，评估错误足以超过模型改进量。

## 1. 从图像指标转向 Study 指标

早期 image-train/eval-max、study_native、Study mean_prob、max/LSE 并存。最终提交是 Study；因此每个评价入口必须检查 Study 覆盖、类别顺序和图像聚合，而不是看训练器输出字段名是否叫 macro_auc。

v3 复核是最明显的反例：X89 图像 Macro 0.81987，正确 Study mean_prob 后 0.81100；X90 图像 0.82127，Study 0.81194。内部 fold 正确 heavy9 分别 0.73482 / 0.72666。不能用图像新高解释 Test 必应提高。

## 2. 从官方 Valid 一个数，转向患者和子群

早期 study_native 的 Valid 0.79905、Test 约 0.7489，引出频率审计：Valid 最大患者有 89 个 Study，占 10.36%；删除该患者分数变化约 0.0086。官方 Valid 有 228 患者，heavy 仅 23；Study 不是独立同分布样本。

项目逐步加入：患者等权诊断、频率桶、AP-single、多视图、hn9、Test-matched、heavy9、AP-single×heavy、逐类 AUC，以及患者簇 bootstrap。它们揭示构成差异，不应不断换主指标以保留最好结论。

两次提交 B 0.7744、X42/X41 0.7766 与 heavy9 约 0.7714、0.77277 接近，支持把 heavy9 当工作代理；样本量和观测次数不足以证明其是无偏、普适的 Test 估计器。尤其不能只最大化 23 名患者上的 best。

## 3. 内部验证的演变

| 阶段 | 做法 | 结果 / 限制 |
|---|---|---|
| subject_v3 / highres | Train 内患者划分及配对臂 | 提醒后续不要 Study 随机分折；具体 split 与官方 Valid 不同 |
| encoder D1 | Train 患者 90/10，1,850 Study holdout | ep3 0.80890，支持早峰并非只来自官方 Valid |
| X49 | 从 Train 留 40 位 heavy 患者、417 Study | AP-single 271 Study /39 患者；不同人群，约 0.69 不能与官方 0.81 直接相减 |
| v3 X84–X93 | 按 Study 分层五折 | 不是患者独立；后续发现严重泄漏和 pn 语义错误 |
| PIV0–PIV3 | 所有训练/初始化祖先排除 X49 患者 | 实现真正患者隔离的 AP 研究对照，结果仍探索性 |
| MC | X49 分 C：20患者208 Study；H：20患者209 Study | C 选模型/拟合，H 比较；X49 历史已被看过，H 不能称全新盲验证 |

从 X21 热启动再到 X49 验证，不能仅因微调阶段排除了 X49 就称患者独立：X21 已见过这些 Train 患者。验证边界必须向所有祖先权重、教师和软目标追溯。

## 4. v3 的三重失真

### 分桶字段错误

原 `pn_buckets.csv` 应为 0/1/2：1 Study、2–8 Study、≥9 Study。fallback 将原始次数 1…21 写入 `pn_bucket`，评估器却取 `pn_bucket==2` 当 heavy。fold0 有 3,703 Study：错误 heavy 为 745，真正原始 Train heavy 为 180。此外旧评估还平均了全部十类，违反 heavy9 的九异常类口径。

### 原版 X92/X93 的直接泄漏

`recovered_train_all.csv` 包含 fold0_val 全部 3,703 Study /6,114 DICOM；0.85 级 AUC 是训练样本上的评价。不能用来定量判断泛化，更不应通过延长蒸馏追高该值。

### “严格修复版”的剩余患者泄漏

修复后的 X92 去掉 Study/DICOM 重叠，但 Train/Valid 仍共享 1,594 位患者，占 Valid 患者 50.22%。ep1–4 正确 Study Macro 为 0.81370/0.81477/0.81567/0.81644，确实在涨，但只支持 Study-held-out，不支持 patient-held-out。该阶段用户已中止；不能据此说应该继续训练。

## 5. 共同错误与病例审阅

做过共同失败/标签质量分析和离线病例审阅材料：代表性 Train/Valid 患者、隐藏预测与标签、约 32 患者/47 图的材料。但没有独立影像阅片人，未得到可靠医学判读。相同患者不同时间标签变化不是错标证据；多个模型一致错误也不是错标证据。

可做的非医学审计包括 ID/标签 join、图像完整性、视角/设备/尺寸分布、同患者跨时间关联、模型错序交集。它们最多定位待阅片样本，不可擅自改成临床真值。

## 已得到什么，仍缺什么

- 已得到：Study 统一复算、原始 pn 语义和患者祖先审计的重要性；PIV/MC 的边界比旧 v3 更清楚。
- 尚缺：具有足够阳性患者的独立确认集；在反复查看官方 Valid 后，对新方法增益的可靠估计。
- 尚缺：能独立确认“图中证据/标签明确性/遗漏病灶”的阅片结果。当前不能以“数据天花板”解释所有失败。

讨论问题：是否先统一一个强制失败的评价器和祖先清单，再定义有限次数的患者独立比较？若新外层样本实在不足，应该如何报告不确定性，而不是继续挑 best？

来源：[早期患者审计](../report/CMS_subject_balance_v3_guide.md)、[代理与选择](../report/CMS_heavy9_proxy_and_selection_20261101.md)、[泄漏审计](../report/CMS_X90_X92_leakage_generalization_audit_20261106.md)、[v3 重审](../report/CMS_heavy9_v3_reassessment_20261106.md)、[病例审计](../report/CMS_evidence_audit_20261102.md)、[PIV 分析](../report/CMS_PIV0_PIV3_analysis_20261106.md)、[MC 协议](../report/CMS_medium_class_4090_experiment_plan_20261106.md)。
