# CMS research_v1 结果

本目录来自远端 `/home/zhaona/CMS_hanxiao/runs/research_v1`，包含第一阶段筛选的 11 个 run。每个完成 run 保留 `config.json`、`metrics.jsonl`、`train.log`、`done.ok` 和 `best.pt`；`last.pt` 没有同步到本机以节省空间。Test 未读取。

主汇总：

- `research_v1_summary.csv` / `.md`：所有 run 的 best Valid 指标和 Macro-AUC 排名。
- `research_v1_analysis.md`：按报告边界撰写的证据解释、类别级差异和下一步计划。
- `remote_runs/runs/research_v1/queue.log`：顺序队列的启动/结束记录。

`unet_global` 被标记为 `invalid_head_only_freeze_bug`：首轮冻结时 encoder 参数被错误排除出 optimizer，后续解冻没有生效。修复后重跑的 `unet_global_v2` 才用于结构结论。其余 10 个 run 均完整完成 5 轮。

排序严格使用未舍入 Valid Study Macro-AUC；只有完全相等时才比较 Macro-F1。单次 seed 结果属于筛选证据，候选仍需多 seed 确认。
