# 路线十二：量化、提交链路、缓存加速与实验执行

这条工程路线影响结果可信度与可用预算。它不能替代方法研究，但多次错误曾让好分数、真实权重和提交概率脱节。

## 1. 历史可用的 weight-only INT8

主路径：Conv/Linear按输出通道INT8存盘，带scale；加载后反量化FP16，在GPU用FP16权重/激活前向；LayerNorm/bias等保留浮点。它是INT8存储、FP16计算，不是W8A8整数运算，也不是必须使用TensorRT。

S1、四模型包、X42/X41等沿此思路满足模型文件体积要求。必须声明真实计算精度；不能把文件INT8说成执行INT8。

X5导出需确认EMA所选键，Swin/DINO各自输入尺寸/归一化不能套ConvNeXt；cosine/per-class scale/Study pooling与checkpoint结构必须一致。后续找到原X42 ep1 INT8用于DZ，不能以其他seed的X42代替原提交成员。

## 2. X90 W8A8 与导出事故

初版X90 TensorRT INT8激活推理官方Test .6377；后来改回历史weight-only INT8→FP16激活，Test .7662。前者与后者不是同一计算配置，不能拿.6377证明浮点模型灾难。

过程中还有FP16 engine、NaN列、错误/混乱的输出说明和依赖安装耗时。v3 dynamic Linear INT8只量化少数Linear，主体卷积仍FP32，并有强制CPU推理路径，不能作为已满足500MB和T4限制的成品。

不应仅因看到Predict_class循环1…10就认定错误：长表本来每Study十行。真正应检查的是每行 `Study_id + Subject_id + Predict_class` 与概率是否对齐、每Study类别是否覆盖唯一、概率有限且在[0,1]，以及Test ID集合完整。

## 3. CSV、规则与推理成本

至少四个必要字段包含Subject_id、Study_id、Predict_class和预测概率；第五列Gate可选，不得误当必填。遵循项目已有正式示例的列名和ID格式，不新造宽表协议。1414 Study应有14140条预测记录。

模型大小按推理实际使用文件解压后合计，≤500MB；不能只称压缩包小于500MB。cost.py与predict.py应共用加载/前向/聚合逻辑，模型FP16/INT8精度差应先在独立Valid验证。T4计时按官方真实口径，5090/4090速度不能替代T4实测。

## 4. 缓存与 CUDA EMA

多次GPU低利用而CPU高，来自原始JPEG解码、缓存未覆盖/路径回退、CPU增强或CPU EMA。做过持久RGB+mask缓存、RAM/mmap、384/448/512独立分辨率、workers24、prefetch、CUDA EMA、gradient checkpointing等。

图像batch42/累积1是5090常用吞吐点；X18用14×3曾因显存回退，其他较大模型21×2。Study batch16不是42个Study。增大batch会改变步数/优化与EMA时间尺度，不能无条件称“只加速不改协议”。

MC224/256从384缓存INTER_AREA重采样、mask用NEAREST，是明确注册的近似，不是和原JPEG一次resize逐位相同。缓存存在不代表生效，应核对meta、覆盖率、实际命中和读取代码，禁止静默回退后仍称缓存实验。

## 5. 队列、生产资源与断点

项目曾用AutoDL、四卡4090和DGX夜间H3感知队列。4090有共享卡锁、真正空闲连续60秒才领取、依赖门控、自动日志/NPZ回传；多台克隆造成相同run多份副本，不是独立重复实验。

DGX约定经历变更：早期窗口内生产任务出现即中止，后改为“单组实验完成后再检查H3，空闲连续60秒才启动下一组”。ComfyUI常驻，仅关注H3。旧报告仍可能写早版本，不能据此声称当前正在执行其中某种策略。

完整 `last.pt` 必须含raw/EMA、optimizer、scheduler、AMP scaler、epoch/step、RNG和DataLoader generator。只有best权重不构成严格续训。中断在epoch中间时还要说明是否重放未提交epoch，避免丢失或重复更新。

自动关机曾多次加入、失效、随后用户取消；本次只记录历史，没有启用关机或修改调度器。

## 6. 结果保全与目前缺口

用户要求回传DGXSPARK的主要是config/log/metrics/NPZ/报告，清理远端失败ckpt副本。多数旧权重已删除，预测和记录仍可复算；不能许诺从现存NPZ生成训练教师目标。

原始指标清单有343份文件、259种内容，不等于343次独立完成实验：包含克隆、重传、失败、同输出no-op和局部日志。目录名字/`done.ok`也不等于方法验收，必须结合状态与产物检查。

本次MC最新快照为：MC10/11已完成、MC12 skipped_gate、MC20恒等性断言失败。应保留“工程待核实”标记，不把归档阶段融合算成最终通过。

## 讨论问题

是否应把数据manifest、预处理/模型/量化配置、祖先hash和唯一Study评价输出固化为一个最小结果包？这样每次研究先比较真实可交付预测，减少“漂亮指标—错误权重—错误提交”的断裂。

来源：[规则与格式](../比赛规则.md)、[X90掉分审计](../report/CMS_X90_test_score_drop_audit_20261105.md)、[v3重审](../report/CMS_heavy9_v3_reassessment_20261106.md)、[DGX调度](../report/CMS_DGXSPARK_H3_aware_scheduler_20261101.md)、[MC缓存/恢复](../report/CMS_medium_class_4090_experiment_plan_20261106.md)、[原始清单](14_run_evidence_inventory.md)。
