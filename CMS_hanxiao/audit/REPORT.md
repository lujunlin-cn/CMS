# CMS 数据审计最终报告与单卡迁移学习计划

**审计日期**：2026-10-24  
**远端目录**：`/home/zhaona/CMS_hanxiao`  
**规则依据**：`/home/hajimi2025/CMS/比赛规则.md` 与官网人工智能赛题页 https://www.ncccu.org.cn/index/Paper/case1.html

## 1. 审计边界

比赛任务是按 `Study_id` 评测的胸部 X 光 10 类多标签分类。审计严格区分数据用途：

- Train/Valid：执行 CSV、路径、Study/Subject、标签、逐图解码、尺寸、像素统计、哈希重复和质量异常审计。
- Test：只核验字段、ID、路径、文件存在性、图像头部形状、Rows/Columns 和文件字节大小；不做像素统计、归一化统计、标签推断、模型选择或对抗样本分析。
- Test 没有进入训练、伪标签、阈值选择、checkpoint 选择或任何模型参数计算。

## 2. 数据规模和文件完整性

| Split | CSV 图像行 | Study | Subject | 允许的检查 | 结果 |
|---|---:|---:|---:|---|---|
| Train | 30,344 | 18,513 | 11,055 | 完整逐图审计 | 30,344 张全部解码成功 |
| Valid | 1,422 | 859 | 228 | 完整逐图审计 | 1,422 张全部解码成功 |
| Test | 2,218 | 1,414 | 136 | 仅形状/文件大小 | 2,218 张头部读取成功 |

Train/Valid 检查结果：

- 文件缺失：0
- CSV 重复键：0
- 图像解码失败：0
- CSV Rows/Columns 与实际图像尺寸不匹配：0
- 图像文件数与 CSV 引用数一致
- 图像均为灰度 `L` 模式

Test 形状/大小检查结果：

- 文件缺失：0
- CSV 重复键：0
- 图像头部读取失败：0
- CSV Rows/Columns 与图像头部尺寸不匹配：0
- 所有图像均为灰度 `L`
- 文件总大小：3,676,899,103 字节
- 单文件大小范围：588,468–4,499,624 字节

## 3. 标签审计

`Predict_class` 按官方多标签格式解析，分号表示同一图像/Study 的多个类别，例如 `7;9` 或 `3;5;9`。

- Train 多标签图像行：9,766
- Valid 多标签图像行：503
- Train 非法标签数：0
- Valid 非法标签数：0
- Train Study 标签数：18,513
- Valid Study 标签数：859
- Train `No Finding` 与其他类别共现：0
- Valid `No Finding` 与其他类别共现：0

此前的 `INVALID` 是旧审计脚本把分号字段误当成单个整数导致的误报，原始 CSV 未被修改。

Study 级标签分布：

| 类别 | Train Study 数 | Valid Study 数 | Train 占比 | Valid 占比 |
|---:|---:|---:|---:|---:|
| 1 | 622 | 24 | 3.36% | 2.79% |
| 2 | 900 | 37 | 4.86% | 4.31% |
| 3 | 942 | 46 | 5.09% | 5.36% |
| 4 | 1,436 | 68 | 7.76% | 7.92% |
| 5 | 2,404 | 115 | 12.99% | 13.39% |
| 6 | 3,885 | 177 | 20.99% | 20.61% |
| 7 | 3,968 | 193 | 21.43% | 22.47% |
| 8 | 4,508 | 214 | 24.35% | 24.91% |
| 9 | 4,654 | 260 | 25.14% | 30.27% |
| 10 | 6,683 | 306 | 36.10% | 35.62% |

## 4. 数据泄漏和重复

Train/Valid/Test 之间的 Subject 和 Study 均无交集：

- Train vs Valid：Subject 0，Study 0
- Train vs Test：Subject 0，Study 0
- Valid vs Test：Subject 0，Study 0

SHA-256 完全重复图像：

- 14 组、28 张图像
- 全部在 Train 内部
- 重复成员位于同一 Study
- 没有跨 Train/Valid/Test 的精确重复

感知哈希候选：

- 共 503 组碰撞
- 31 组跨 Train/Valid
- 形成 53 个跨集配对
- 256×256 像素相关性最高：0.929
- 最低 MAE：0.073
- 近似完全重复候选：0

因此没有确认的跨集图像泄漏。感知哈希结果保留为审计记录，不自动删除数据。

## 5. Train/Valid 图像质量

Train 图像尺寸范围：宽 1067–4280，高 1409–4280。  
Valid 图像尺寸范围：宽 1266–3520，高 1716–4280。

逐图统计包括灰度均值、标准差、最小/最大值、1%/99% 分位数、黑白像素比例、SHA-256、aHash 和 dHash。

发现 1 张低信息图像：

```text
train/subject_05239/study_13172/pic_021690.jpg
```

其 97.23% 像素为 0，人工查看确认只有右下角局部影像；同一 Study 的 `pic_021691.jpg` 是完整胸片。处理方式：

- 原始图像保留
- 不删除官方数据
- 通过 `audit/exclude_train_images.csv` 仅从基线训练排除
- Valid 和 Test 不受影响
- 该处理可以通过删除 manifest 条目完全恢复

元数据空值也被记录但未强行填充：

- Train：ViewPosition 空 1,202；PatientOrientation 空 3,307
- Valid：ViewPosition 空 39；PatientOrientation 空 135
- Test：ViewPosition 空 139；PatientOrientation 空 369

规则没有要求这些字段必须非空，因此保留原始值。

## 6. 单卡迁移学习比较计划

已写好以下脚本：

```text
/home/zhaona/CMS_hanxiao/train_single_gpu.py
/home/zhaona/CMS_hanxiao/run_compare_pair.sh
/home/zhaona/CMS_hanxiao/run_compare_all.sh
/home/zhaona/CMS_hanxiao/start_compare_when_idle.sh
/home/zhaona/CMS_hanxiao/summarize_compare.py
/home/zhaona/CMS_hanxiao/TRAINING_PLAN.md
```

训练协议：

- 不使用 DDP，不做单模型多卡同步
- GPU 0 和 GPU 1 各运行一个独立模型
- 每轮同时比较两个模型，完成后进入下一轮
- 模型顺序：DenseNet-121、EfficientNetV2-S、ConvNeXt-Tiny、ConvNeXt-Small、Swin-Small、DINOv2-Base-Reg4
- 默认 5 epochs、batch size 16、224×224 letterbox
- ImageNet normalization、AMP、AdamW、学习率 3e-4、weight decay 1e-4
- 默认关闭水平翻转，避免胸片左右侧别改变
- 以 Study 级 Macro-AUC 为主指标，Macro-F1 为辅助指标
- 多图 Study 使用逐类别最大图像概率聚合
- Test 不参与训练和模型比较

启动全部 6 个模型的三轮比较：

```bash
/home/zhaona/CMS_hanxiao/run_compare_all.sh
```

手动启动一对：

```bash
/home/zhaona/CMS_hanxiao/run_compare_pair.sh densenet121 efficientnetv2_s
```

结果目录：

```text
/home/zhaona/CMS_hanxiao/runs/compare/<model>/
```

每个模型保存 `config.json`、`train.log`、`metrics.jsonl`、`best.pt`、`last.pt` 和完成/失败标记；汇总文件为：

```text
/home/zhaona/CMS_hanxiao/runs/compare/comparison_summary.csv
```

## 7. 定时和当前状态

当前 GPU 0/1 仍被其他任务占用，因此没有抢占或终止其他进程。已设置：

```text
每小时整点：hourly_health_check.sh
每 5 分钟：start_compare_when_idle.sh
```

只有当 GPU 0 和 GPU 1 都低于空闲显存阈值时，比较任务才会自动启动。健康检查会记录比较 PID、每个模型目录状态、磁盘和 GPU 使用情况。

基于模型的 FGSM/PGD 审计脚本已准备，但会在比较模型确定后针对 Valid 执行；不会对 Test 做对抗分析。
