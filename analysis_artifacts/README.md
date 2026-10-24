# CMS 训练与推理复核产物

本目录由 `sync_remote_results.py` 从训练机自动同步。包含七个模型的训练指标、日志、train/valid 全量 Study 推理结果、错误分析和 FGSM/PGD 对抗敏感性审计结果。权重、原始影像和测试集未同步。

`remote_runs/runs/adversarial_audit/` 下的 `metrics_summary.csv`、各模型 `summary.json`、`study_sensitivity.csv` 和 `top_sensitive_*.csv` 已发布到 GitHub。合并后的 `study_sensitivity_all.csv` 约 105 MB，保留在本机分析目录以遵守 GitHub 单文件限制。
