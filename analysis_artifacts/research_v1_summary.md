# CMS 第一阶段研究实验结果

排序规则：Valid Study Macro-AUC 降序；仅完全相等时比较 Macro-F1。Test 未读取。

|排名|实验|模型/结构|best epoch|Macro-AUC|Macro-F1|相对固定参考|状态|
|---:|---|---|---:|---:|---:|---:|---|
|1|cs_lr01|convnext_small  |2|0.786671|0.266916|+0.000161|complete|
|2|cs_p0|convnext_small  |3|0.783646|0.331068|-0.002864|complete|
|3|eff_norm_half|efficientnetv2_s  |1|0.778685|0.257918|+0.009722|complete|
|4|cs_asl_g4|convnext_small  |2|0.777593|0.388946|-0.008917|complete|
|5|cs_asl_g2|convnext_small  |2|0.776856|0.342915|-0.009654|complete|
|6|unet_residual|densenet121 residual topk|5|0.775032|0.261956|+0.050961|complete|
|7|eff_norm_imagenet|efficientnetv2_s  |2|0.770700|0.254336|+0.001737|complete|
|8|unet_global_v2|densenet121 global mean|4|0.770043|0.256329|+0.045972|complete|
|9|cs_study_mean|convnext_small  |3|0.767752|0.247914|-0.018758|complete|
|10|unet_lse|densenet121 decoder logsumexp|4|0.727270|0.170195|+0.003199|complete|
|11|unet_global|densenet121 global mean|1|0.716281|0.122746|-0.007790|invalid_head_only_freeze_bug|

## 解释

只把 Macro-AUC 的增量视为主结果；Macro-F1 仅在 Macro-AUC 完全相等时用于比赛排序。单个 Valid 选择结果属于探索性证据，优胜者需用匹配对照和多 seed 复核。U-Net 的全局、残差和空间池化实验分别检验预训练全局语义、局部证据补充与平滑空间 MIL。
