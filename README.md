# CMS 医学影像比赛实验代码

本仓库保存 CMS 比赛的数据审计、合规记录、U-Net + MIL 研究与迁移学习训练代码。

数据集、预训练权重、检查点和运行日志不纳入 Git。训练时将数据和权重放在本地工作目录，并按脚本中的路径配置使用。

## 主要内容

- `比赛规则.md`：比赛规则与提交约束
- `model_selection.md`：预训练模型基座选择记录
- `CMS_hanxiao/audit/`：训练集、验证集审计报告及统计文件
- `CMS_hanxiao/compliance/`：训练谱系、权重来源和许可证记录
- `CMS_hanxiao/train_single_gpu.py`：单 GPU 分类迁移学习
- `CMS_hanxiao/train_unet_mil_advanced.py`：U-Net + MIL 训练入口
- `CMS_hanxiao/run_experiment_queue.py`：四 GPU 七模型自动实验队列
- `CMS_hanxiao/run_post_training_inference.py`：等待七个最佳检查点后，自动批量推理 train+valid
- `CMS_hanxiao/full_inference_analysis.py`：按 Study 输出全量预测、类别错误和模型分歧
- `UNET_MIL_RESEARCH.md`、`CMS_hanxiao/UNET_MIL_IMPROVEMENTS.md`：算法研究记录

训练队列的默认实验为 6 个视觉骨干基线，加 1 个 DenseNet121 编码器的 U-Net + MIL 变体；每个实验只占用一张 GPU。
训练完成后只读取 train 和 valid 做全量推理与错误分析，test 保持隔离。
