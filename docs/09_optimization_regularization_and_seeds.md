# 路线九：优化配方、正则、EMA、训练时长与种子

核心问题：模型究竟在缺少学习，还是早已过拟合？改变优化动力学能否保留排序，而非只降低训练损失？

## 1. encoder 学习率和全量微调

早期Small、encoder LR×.1与不同normalization混用。D/R里的R2 encoder倍率.01只有.78042，明显低于D0 .79791；R1 weight_decay .01约.79815，R3增强.79990。

E1 corrected package改为Large384、encoder LR×1、layer decay.8、warmup2，峰值ep3 .80382，预注册固定ep5只有.78945。不能事后把正式ep5替换成峰值还宣称未改协议。Train患者独立D1 holdout也在ep3达.80890后下降，说明早峰不只是官方Valid偶然。

## 2. X5 强正则配方和 EMA

X5设置WD .05、EMA .999、bootstrapped .02、增强、minLR ratio .02，10ep最好ep4 EMA .81128。相比E1高，但包含多个同时变化，只支持整体配方。

EMA是参数时间平滑，不是额外数据。需要同时保留raw训练模型和EMA副本，验证/导出使用明确所选权重。X5若读错 `model` /`ema_model`，会把未平滑模型和EMA分数错配；不同年代保存语义不同，应读取config和selected_weights。

早期C6因CPU/DataLoader/EMA执行问题取消；后来CPU EMA导致大量CPU拷贝、GPU利用低，统一迁到CUDA改善吞吐。它解决执行效率，不应被统计为一种新模型机制。

## 3. SWA、SAM、stochastic depth

X6对ep2/3/4参数平均，多run检验；ConvNeXt不如各自best，X5约-.0064；只有DINO某run通过局部门槛。不能假设SWA必优EMA。

X15 SAM+adversarial弱，X39纯SAM也只有.80733。X46 drop_path.2完整八轮原版.80887，clean重跑.80760；旧分析引用单轮约.76并说掉.048，不是完整训练峰值。低drop_path.1热启动X58三轮.81143，未赢。高drop_path.4是提案，未确认独立完成结果。

## 4. 热启动与“多训一个低 LR epoch”

| 分支 | 预算 | 观察 |
|---|---|---|
| X26 X5+轻decorr | 6ep | ep1最好.80991 |
| X31 X21+逐类scale | 4ep | ep1最好.81533，scale几乎不变 |
| X42 X21+heavy采样 | 3ep | ep1最好.81546 |
| X43 无heavy热启动控制 | 2ep | 原版.81516，显示热启动本身可能解释部分小收益 |
| X56/X58 | 4ep门控/3ep | 首轮或早期最佳，未超X21 |
| X64/X65 | 3ep | 均ep1最好，后续未救回 |
| A1 Freeze_X21 | 3ep | ep1最好.81519；不能据低loss或末轮说必需加轮 |

这些结果支持短程微调容易改变校准或丢失原排序，但不证明“任何后续训练都无用”。新模块学习速度与bone更新速度需要匹配控制。

## 5. 种子与超参扫描

X44只换seed20261025，原版macro .80693；heavy曾较X21下降约.012。它是一次配对观察，不是所有配置的方差估计。

4090后续X21 seed26–29六轮峰值分别.80500/.80383/.81131/.80985；相应X42热启动三轮.80335/.80555/.81011/.80979，没有复制原seed的.81457/.81546。预算不完全一致，不能把差异全解释为seed。

LR/EMA扫描存有五组日志：lr1e-4/ema.9985约.80952；1e-4/.999约.80933；8e-5/.9985约.81077；8e-5/.999约.81019；8e-5/.9995约.80796。不同轮数与选择过程记录在清单里，没有发现比当前最佳组合强的正式Test证据。

一个危险旧报告将queue里的overall值命名heavy9，并假设seed25的X42从X20 checkpoint开始；该假设不是“同种子X21”的证明。比较前必须核对实际config和祖先路径，不能照报告字段名算种子均值。

## 6. 预算、batch和严格续训

历史预算多次由用户改变：X5十轮、X21八轮、X25七轮、X30四+四、后来的本地方案最多六轮、A Full六/Freeze三、v2 X1十二轮。不存在适用于全史的统一“最多六轮”。

图像batch42/accum1是5090吞吐默认；Study batch16/worker24不同于42个Study。batch变化会改变优化步数和EMA时间尺度，不能说任何batch调整都只是无损加速。完整续训还需optimizer/scheduler/scaler/raw+EMA/RNG/DataLoader generator，只有best.pt不足以恢复同一训练轨迹。

## 讨论问题

若目标是证实+.005，应否先用少量匹配seed估计“同协议差值”而不是挑最优seed？热启动新模块是否应先冻结保留主路径、明确选择epoch0为合法候选，防止把必然退化的后续epoch当成果？

来源：[E1正式对照](../report/CMS_E1_corrected_control_analysis_20261029.md)、[encoder复盘](../report/CMS_encoder_budget_v2_retrospective_and_next_stage_20261029.md)、[X31–42](../report/CMS_X31_X42_deep_analysis_20261101.md)、[种子执行与已知限制](../report/CMS_seed_scan_execution_20261102.md)、[A协议](../report/CMS_heavy9_v2_revised_queue_20261104.md)、[原始日志清单](14_run_evidence_inventory.md)。
