# 单卡迁移学习比较计划

## 目标

在完全相同的数据划分、Study 级标签聚合、预处理、随机种子和验证指标下，比较 6 个已下载的非医学预训练视觉骨干。每个进程只使用一张 GPU，不使用 DDP 或多卡同步。

## 阶段 A：初筛

- GPU 0 与 GPU 1 各运行一个模型；每轮结束后再运行下一对。
- 模型顺序：
  1. `densenet121`
  2. `efficientnetv2_s`
  3. `convnext_tiny`
  4. `convnext_small`
  5. `swin_small`
  6. `dinov2_base_reg4`
- 默认 5 epochs、batch size 16、224×224 letterbox、ImageNet 均值/方差、AMP、AdamW、初始学习率 3e-4、weight decay 1e-4。
- 默认不做水平翻转，避免胸片左右解剖侧别被改变。
- 指标：Study 级 Macro-AUC 为主，Macro-F1 为辅；多图 Study 使用逐类别最大图像概率聚合。
- Train 中已确认的低信息图像通过 manifest 排除；Valid 保持完整；Test 不参与任何训练、统计、选择或比较。

启动全部三轮比较：

```bash
/home/zhaona/CMS_hanxiao/run_compare_all.sh
```

手动只比较一对：

```bash
/home/zhaona/CMS_hanxiao/run_compare_pair.sh densenet121 efficientnetv2_s
```

结果目录：

```text
/home/zhaona/CMS_hanxiao/runs/compare/<model>/
```

每个模型目录包含 `config.json`、`train.log`、`metrics.jsonl`、`best.pt`、`last.pt` 和 `done.ok`/`FAILED`。汇总结果为 `runs/compare/comparison_summary.csv`。

## 阶段 B：复训

从阶段 A 的 Macro-AUC 前两名中选择 2 个模型，固定数据处理和验证集，增加 epochs 或进行有限学习率/解冻策略比较。任何选择都只基于 Train/Valid，不读取 Test。

## 阶段 C：鲁棒性与提交准备

对最终候选执行 Valid 集上的强度扰动、遮挡和 FGSM/PGD 检查；固定模型、阈值和后处理后才生成 Test 提交。Test 只做形状/尺寸核验和最终固定参数推理。
