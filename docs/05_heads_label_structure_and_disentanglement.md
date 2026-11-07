# 路线五：分类头、疾病结构与严重程度解耦

核心假设：类别共现和全局严重程度可能主导特征，稀有或局部疾病被挤压；改变头部几何、标签层次或去相关目标能否产生新增判别信息？

## 1. Cosine / prior head：最有用的一次“重新隔离变量”

EA 曾在 ConvNeXt-Small@224、encoder LR×.1 等组合下只有 .79270。旧报告因此宣称余弦先验头无效，因果依据不足。

X21 用完整 X5 强正则配方，只加入 prior-cosine-head，八轮峰值 ep5 .8145687，官方 Valid heavy9 .76430。它成为后续热启动与对照的主力，说明早期混杂负结果不能关闭整个方向。

X29 在 XL 上加 cosine .80723，未复制 Large 的收益；不能假定同一个头与任意容量都兼容。

## 2. 逐类温度 / scale

X31 从 X21 热启动四轮，峰值 ep1 .81533。十个 scale 约停在 9.95–9.96，几乎没学动；收益可能来自多训一个低 LR epoch，缺干净热启动控制时不能归于逐类温度。

X40 原 XL+per-class 方向取消/拆成 X41、X42：

- X41：SwinV2-L per-class cosine，.81072 /heavy .76529，作为异构成员有用。
- X42：X21 heavy 过采样热启动，.81546 ep1 /heavy .76607；hn9 曾提高 .0074，但 AP-heavy 未提高。

X43 后来补无过采样热启动对照，原版 .81516；存在 after_x55_clean 另一版 .81053。其结果必须和相同版本/配方的 X42 对齐，不能跨重跑归因。

## 3. HierarchicalHead / 标签层次

X11 层级头及 hierarchical_loss，修复 OOM、迁移 5090、batch/缓存后完成七轮，峰值 ep5 .81050。4090 两轮 .79485 是中断快照，不能当最终结论。没有超过 X21，但早于 X21 的协议背景不同，不能以分差孤立评判层级先验。

X55 additive severity head 尝试将全局严重程度与疾病特异部分分开，八轮 .80644，低于 X21；有过 GRL 参数冲突的失败副本。未显示净收益。

## 4. Logit 去相关 / 特征去相关

| 分支 | 改法 | 官方 Valid 峰值 | 结果 |
|---|---|---:|---|
| X20 | 从预训练完整训练 + logit decorr | .80316 ep4 | 后期退化 |
| X26 | X5 ep4 热启动，轻 decorr .02，低 LR | .80991 ep1 | 未超过起点 |
| X30 stage1 | 强 decorr .10，四轮 | .79940 ep3 | 强约束不保留原排序 |
| X30 stage2 | stage1 ep4 起，去掉 decorr，四轮 | .80195 ep1 | 未恢复主力能力 |
| X34 | GRL severity disentanglement | .73339 | 明显退化 |

疾病本来共现；相关性包含真实病理信号，也可能含捷径。把 corr 降低并不等于 AUC 提高，不能以去相关诊断值达标替代独立患者排序增益。GRL 若去掉与疾病真值相关的严重程度，可能伤害任务，但当前证据不足以定量分解具体机制。

## 5. 组合实验

X38 原设计叠逐类温度 + rare feature mixup + AbnNeg，实际 AbnNeg 未生效、rare 索引选错，峰值 .81148。它不是三种正确部件组合的有效检验。X42 的小幅收益也不证明 severity collapse 已被修复。

## 讨论问题

如果温度几乎不动、attention 没有空间自由度、decorr 只改变诊断相关性，我们究竟是否改变过困难病例的证据表示？下一次头部实验是否应保留强全局路径、从恒等初始化开始，先比较同训练步数的无改动热启动，再解释新模块作用？

来源：[X11 实现](../report/CMS_X11_X15_codex_impl_20261030.md)、[X21–30](../report/CMS_X21_X30_codex_impl_20261031.md)、[X31–42 分析及纠错](../report/CMS_X31_X42_deep_analysis_20261101.md)、[X43–49](../report/CMS_X43_X49_codex_impl_20261101.md)、[统一瓶颈复算](../report/CMS_X21_X42_bottleneck_20261106.md)、[原始指标](14_run_evidence_inventory.md)。
