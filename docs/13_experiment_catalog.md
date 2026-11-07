# 实验编号总表：名称、版本、路线与证据状态

整理日期：2026-11-07。这张表索引“做过什么”，不构成统一排行榜。数值与解释见各路线；原始日志逐项见 [14](14_run_evidence_inventory.md)。

## 读表规则与命名冲突

- **完成**：有训练/预测结果；不代表显著有效或可提交。
- **失败/中止**：运行过但未完成预算或技术验收。
- **无效**：没有实施所称变量或数据/指标边界错误。
- **门控/计划**：方案或代码存在，无确认完成证据。
- 同名多个版本均保留。旧 retrospective 的“候选 X1–X10”并不是最后实际 X 队列的编号定义。
- `D2_s20261024` 是早期患者权重；`D2_VIEW_CONSISTENCY` 是encoder视角一致性。
- `R1_s20261024` 是正则；highres `R1` 是rank；PIV放大 `R1_FROZEN_IMAGENET` 是冻结。不能混并。
- `X1_AP_SPECIALIST` 与 v2 `X1_CONTRASTIVE_HEAVY_FULL` 完全不同。
- encoder `E1` 与 v2后处理 `E1_slice`、structural A/B与v2架构A1–A4也不同。

路线编号：01验证；02骨干/分辨率；03Study/多视图；04空间；05头/结构；06损失；07采样；08预处理；09优化；10蒸馏；11融合；12部署。每个编号对应同前缀文件。

## 早期实验族系（2026-10-23 至 10-29）

| 族系 / 实验 | 尝试方向 | 状态与限制 | 主路线 |
|---|---|---|---|
| backbone screen | DenseNet121、EfficientNetV2-S、ConvNeXt-T/S、Swin-S、DINOv2-B | 多个五轮筛选；不同配方，非公平排行 | 02 |
| Small640 e6/e7/e8 | 提高分辨率及配方变体 | 有八轮日志，非首次高分辨率空白 | 02/09 |
| cs_p0 /cs_lr01 /eff_norm /cs_asl_g2,g4 /cs_study_mean | normalization、encoderLR、ASL和Study聚合 | 有日志；只按实际配置解释 | 02/03/06/08 |
| structural_v1 A/B/gamma | 监督/结构、局部辅助、gamma | 配对结果存在；协议不同 | 03/04/08 |
| structural_v2 gamma seed24/25/26 | 增强确认 | 三seed正均值，image/eval-max | 08/09 |
| study_native + confirm | 图像→Study原生监督 | 有早期峰值；后期NaN，确认失败 | 03/12 |
| view_attention | Study视角attention | 小幅负向 | 03 |
| unet_local_aux | 局部辅助 | 极小正点差，未确认 | 04 |
| subject_v3 A1/A2 | rotate、mask-aware photo | 配对增强试验 | 08 |
| subject_v3 B0/B1/B2/B3 | 修复控制、频率组、患者sqrt、类别正例sqrt | 特定dev配对，不与官方全量混排 | 07 |
| U0 /U0W | decoder、DenseNet全局/残差 | 多seed/小子集，未成强路线 | 04 |
| U3 multiscale residual | ConvNeXt多尺度残差 | 五轮有结果 | 04 |
| highres R0/R1 | BCE vs全局rank | seed24有小信号，seed25未完成确认 | 06 |
| highres U-Net anchor/staged/DDP smoke | 接线、分阶段、数值试验 | 小预算/中止，不能当完整效果 | 04/12 |
| D0 | 224基础控制 | 完成12轮 | 09 |
| D1 | Study 1/m_s | 有小幅点差，区间跨零 | 07 |
| D2 | 患者^-1/2 | 完成，负向 | 07 |
| D3 | 稀有正例1.5 | 完成，未突破 | 06/07 |
| D4（早期） | OOF噪声方案 | hard lock阻塞，无完成结果 | 10 |
| D5 | 患者batch去冲突 | 近似实现，未满足严格协议 | 07 |
| D6 | D1×D2 | 完成，未优于D1 | 07 |
| D7/D7U | 患者Study封顶/匹配Study预算 | 完成，存在曝光预算混杂 | 07 |
| D0_SHORT /D1_R3 /D1_D3 /ASL_D0 | 控制缩短与组合试探 | 有短日志/一轮中止，分别记录 | 06/07/09 |
| R1/R2/R3（D/R族） | WD、encoderLR.01、photo/noise | 完成，未稳定突破 | 08/09 |
| C0多seed /C1_lr01 /N1_eff_half | C系控制和输入比较 | 早期命名空间；不与C1_STUDY_BCE混同 | 02/08/09 |
| HR512 BCE_MEAN/HYBRID/AUC_EMA/BCE_LSE/EMA_MEAN | 512损失×聚合×EMA | 多组五轮完成 | 02/03/06/09 |
| HR512 CLASS_TOKEN /VIT_OVERLAP_CROP | attention/ViT裁剪 | 短日志或无有效epoch，技术未完成 | 02/03 |
| C0–C7的384 FP32协议 | 数值修复后的监督消融 | 多版本；见下面正式C结果 | 03/12 |
| C1 | Study BCE mean_prob | 完成，.80467 | 03 |
| C2/C2b/C2c | rank主损失、BCE+.05/.10rank | 完成，未优于C1 | 06 |
| C3 | masked global | 完成，小点差 | 03/08 |
| C4 | view dropout | 完成，负向 | 03 |
| C5 | class-token | .80790候选信号 | 03 |
| C6 | EMA | 执行问题取消，无有效完成结果 | 09/12 |
| C7 | bootstrap | 完成，近中性 | 06 |
| YOLO26X GLOBAL/WSL | 初版分类/弱监督 | 输入契约曾错；不可单独判架构 | 02/04 |
| H1/H2/H3 | raw global、officialgeometry、true multiscaleWSL | 修正后20轮；仍弱，峰值和末轮区分 | 02/04 |
| H4 | rank-pool Small | 只有短运行，未完成正式验证 | 03/06 |
| H10/H15/H16/H17 | Small控制、BCE→AUC、低rank、earlystop | 有结果，未突破 | 06/09 |
| E1 corrected | Large384全量encoder+warmup+layerdecay | 完成，峰值ep3与正式ep5分列 | 02/09 |
| E2/E4/E7 | encoder预算中的其他提案 | 已跳过/未找到完成证据 | 02/09 |
| E3 | E1增强版 | 完成，成为S1成员 | 08/11 |
| E5 | class-token+view position | 多精度非有限，技术未完成 | 03/12 |
| E6 | nnPU早期配方 | 有历史比较，配方混杂；勿当X22同义 | 06 |
| EA | Small224 cosine prior、encoderLR×.1 | 完成；不是cosine单变量 | 05 |
| EB | Small+balanced+hybrid+cosine组合 | 完成，多变量失败 | 06/07 |
| encoder D1 | Train患者90/10 holdout | 完成，独立早峰诊断 | 01/09 |
| encoder D2 | view-specific head+KL | 完成，未修AP | 03 |
| encoder D3 | 旧队列后续推理/评估项 | 曾取消自动执行；不补造独立训练结果 | 11/12 |
| encoder D4 | D2+OOF噪声权重 | 完成，未超D2 | 10 |
| S1/S2 | 同构平均hflip /AP2×2 crop | 已有正式Test.7653/.7586 | 11/12 |

## 实际 X1–X20

| 编号 | 实际内容 | 状态 / 关键版本 | 路线 |
|---|---|---|---|
| X1 | AP-family专家384 | 完成；与v2 X1分开 | 07 |
| X2 | AP-family /AP-single确定性路由 | Valid已评估，负向；不另计训练 | 07/11 |
| X3 | AP专家512 | 完成，低于X1 | 02/07 |
| X4 | 心影中心crop同Study MIL +无crop控制 | 两臂完成，crop负向 | 03/04 |
| X5 | 强正则十轮、EMA | 完成，ep4 .81128 | 08/09 |
| X6 | SWA ep2/3/4 | 多run重评，ConvNeXt无收益 | 09 |
| X7 | Swin-S384 /DINOv2-B384 | 两骨干完成 | 02 |
| X8 | 固定模型包多尺度TTA | Valid完成，微小点差 | 11 |
| X9 | JPEG40–95、noise增广及q50审计 | 完成，clean中性 | 08 |
| X10 | 患者配对bootstrap、末轮/best/SWA分析 | 评估工具已接入；不是新训练 | 01 |
| X11 | HierarchicalHead | 4090失败/中断；5090七轮完成 | 05 |
| X12 | hard-negative gamma2 | 十轮老版完成，ep4最好 | 06 |
| X13 | patient lookup+pair ranking | 七轮完成 | 06 |
| X14 | OOF软标签 | 旧路径疑点+七轮低分；不是纯路径结论 | 10 |
| X15 | SAM+adversarial | 七轮完成，低于强基线 | 06 |
| X16 | ConvNeXt-XL384 | 七轮完成 | 02 |
| X17 | ConvNeXt-L512 | 10ep/7ep两版本，峰值不同 | 02 |
| X18 | DINOv2-L-reg4@518 | 七轮完成；加五轮提案后取消 | 02 |
| X19 | GroupDRO pn桶 | 现存早版9轮日志，不能写10/10 | 07 |
| X20 | logit decorr | 原参数未实施→修复七轮完成 | 05 |

## X21–X42

| 编号 | 实际内容 | 状态 / 关键版本 | 路线 |
|---|---|---|---|
| X21 | X5配方+prior cosine | 八轮完成；基础对照，contain | 05/09 |
| X22 | X5配方nnPU | 八轮完成，负向 | 06 |
| X23 | gamma jitter+noise | 八轮完成，负向 | 08 |
| X24 | SwinV2-L256 | 八轮完成，异构成员前身 | 02 |
| X25 | class-token+view-position | 七轮完成，低于X21 | 03 |
| X26 | X5热启动轻decorr | 六轮完成，ep1最好 | 05 |
| X27 | Study class-balanced单变量隔离 | 七轮完成，下降 | 07 |
| X28 | OOF CSV路径核查+alpha.15 | 首轮低分触发中止 | 10 |
| X29 | XL+cosine | 八轮完成，未超Large | 02/05 |
| X30 | 强decorr四轮→finetune四轮 | 两阶段完成，未突破 | 05 |
| X31 | X21热启动per-class scale | 四轮完成；scale近不动，热启动混杂 | 05 |
| X32 | rare feature mixup | 八轮完成；rare类0/7选择错误 | 08 |
| X33 | heavy5×+within-heavy contrastive | 八轮完成，负向、多变量 | 07/08 |
| X34 | GRL severity去信息 | 八轮完成，明显退化 | 05 |
| X35 | AbnNeg alpha | 原实现no-op，不能判方法无效 | 06 |
| X36 | heavy checkpoint重选 | 零训练评估，未见突破 | 01/09 |
| X37 | X21opt+X24固定融合 | 零训练评估，有小点差 | 11 |
| X38 | scale+mixup+AbnNeg组合 | 八轮完成，但两个部件失配/无效 | 05/06/08 |
| X39 | 纯SAM batch32 | 八轮完成，未突破 | 06 |
| X40 | 原XL+per-class候选 | 取消，实际更新为X41/X42 | 02/05 |
| X41 | SwinV2-L+per-class cosine | 八轮完成，ep6；77.66成员 | 02/05/11 |
| X42 | X21热启动heavy5× | 三轮完成，ep1；77.66成员 | 07/11 |

## X43–X66

| 编号 | 实际内容 | 状态 / 关键版本 | 路线 |
|---|---|---|---|
| X43 | 无oversample热启动控制 | 原版/after_x55_clean均有；必须分版本 | 09 |
| X44 | 换seed测波动 | 原版/clean、后续seed扫描分开 | 09 |
| X45 | 修复AbnNeg loss | 两版八轮，未超X21 | 06 |
| X46 | drop_path .2 | 原版/clean八轮；不是旧报告单轮低分 | 09 |
| X47 | aspect/mask感知增强 | invalid_contain失败后完成；X21本已contain | 08 |
| X48 | Swin heavy hotstart | 两轮完成，ep1最好 | 07 |
| X49 | Train heavy患者第二验证集与其控制 | 417Study/40患者；不是官方Valid分数 | 01 |
| X50 | 正确rare mixup等候选 | 提案，未确认独立完成run | 08 |
| X51 | Study权重1/m_s | 八轮完成 | 07 |
| X52 | Study原生mean_prob | 八轮完成 | 03 |
| X53 | 多视图特权前置推理 | 完成；+.0042未过.01门槛 | 03 |
| X54 | 全局AUC queue | 两轮已有结果；无更长完成证据 | 06 |
| X55 | additive severity head | 冲突修复后八轮完成 | 05 |
| X56 | AP-heavy5×/frontlatlight.5×、X21热启动 | revised首轮触发停损；非原从头版 | 07 |
| X57 | contain+心影四边crop | 八轮完成，负向 | 08 |
| X58 | X21温和drop/低LR/采样热启动 | 三轮完成，负向 | 07/09 |
| X59 | 448+stretch | 八轮完成，分辨率/几何混杂 | 02/08 |
| X60 | 多crop TTA | 纯推理完成，负向 | 11 |
| X61 | Train+Valid合并固定ep5 | 五轮训练记录；禁用Valid选择，无独立Valid AUC | 01/09 |
| X62 | Test可观测密度比Study权重 | 八轮完成，低分且规则疑点 | 07 |
| X63 | attention/noisy-OR Study MIL | 八轮完成，未突破 | 03 |
| X64 | X21 heavy loss额外.5 | 三轮完成，负向 | 06 |
| X65 | X21 heavy-only pairwise | 三轮主运行；另有干净控制中止版本 | 06 |
| X66 | heavy checkpoint selector | 代码/依赖控制存在，用户中止后未确认最终选择执行 | 01 |

## X67–X93 与命名缺口

| 编号 | 实际内容 | 状态 / 关键版本 | 路线 |
|---|---|---|---|
| X67 | 448 contain masked_gap | DGX六轮有结果；与最早八轮计划分开 | 02/08 |
| X68 | 512 contain masked_gap | 门控方案，无确认完成证据 | 02 |
| X69 | 384 CLAHE | DGX六轮有结果 | 08 |
| X70 | 448+CLAHE+masked_gap | DGX六轮有结果；后迁5090计划不等于另已完成 | 02/08 |
| X71 | 底部crop+偏置erasing | 已实现，未确认有效完成 | 08 |
| X72 | 条件多尺度TTA | 门控方案，未确认完成 | 11 |
| X73 | 最佳新配置温和热启动 | 门控方案，未确认完成 | 09 |
| X74 | AP-heavy residual热启动 | 有DGX首轮和停损记录，未确认完整预算 | 07 |
| X75 | Train-only对比SSL→监督适配 | 有启动计划，未确认完成；合规待核对 | 08 |
| X76–X83 | 编号缺口 | 本次报告/源码/日志未找到可核实的此编号run，不能补造；其间实际活动以T/A/v2 X1命名 | — |
| X84 | fold0 baseline | 六轮完成；旧heavy口径错，患者非隔离 | 01 |
| X85 | AP-only teacher | 六轮子集完成，非完整Valid成绩 | 07/10 |
| X86 | LAT-only teacher | 六轮子集完成，非完整Valid成绩 | 03/10 |
| X87 | view soft-label生成阶段 | 队列有生成步骤，未见独立X87训练run | 10 |
| X88 | view soft-label student | 六轮完成，正确Study/trueheavy下降 | 10 |
| X89 | region-aware top-k/max | 六轮完成；image/Study指标差异、非患者CV | 04 |
| X90 | region-to-global蒸馏 | 三轮完成；错误W8A8与修正weight-only提交分开 | 10/12 |
| X91 | 五折同架构教师 | 五个六轮run；Study分折非患者分折，OOF语义错误 | 01/10 |
| X92初版 | ensemble蒸馏，后加到8/10轮 | 验证Study100%进入训练，0.85无效 | 10 |
| X92严格修复版 | 去除Study/DICOM重叠，5轮预算 | 实际停ep4；仍1594患者重叠，非患者独立 | 01/10 |
| X93新设计 | X92+region+hard BCE/heavy | 运行过但不是X89/X90集成，且泄漏指标无效 | 04/10 |
| X92/X93旧量化命名 | 早期纯量化收尾 | 已被重定义覆盖，不并入新训练 | 12 |

## v2 T/A/E 双线与相关零训练

| 实验 | 内容 | 状态 / 解释 | 路线 |
|---|---|---|---|
| T2 | heavy oversampling3× | 旧v2链曾接入，取消/中断，缺正式验收结果 | 07 |
| T1 | heavy focal | 旧链受CPU EMA/缓存/低batch影响，后取消 | 06/12 |
| T3 | heavy feature mixup .4 | 旧链取消，未确认完整结果 | 08 |
| v2 X1 FULL | contrastive-heavy，原始预训练 | 十二轮完成，peak ep6；另有X21启动一轮中止副本 | 08 |
| A1 FULL /FREEZE_X21 | cross-view attention | 六/三轮完成，Freeze有小幅候选信号 | 03 |
| A2 FULL /FREEZE_X21 | patch attention | 六/三轮完成，未超基线 | 04 |
| A3 FULL /FREEZE_X21 | multires FPN | 修复互斥/接口后六/三轮完成 | 04 |
| A4 FULL /FREEZE_X21 | hierarchical MIL | 修复后六/三轮完成 | 04 |
| A1/A2/A3旧FREEZE | resume=null冻结ImageNet | 协议无效，不代表X21冻结结果 | 04/09 |
| v2 E1 /E2 | slice weights /LightGBM stacking | 后处理代码存在，未确认完整独立验收 | 11 |
| X21 seed26–29 /X42 seed26–29 | 六轮原训+三轮热启动 | 有日志，未超原始seed；不可当同预算重复 | 09 |
| LR/EMA grid | lr8e-5/1e-4×ema.9985/.999/.9995 | 五组有日志；不是正式Test突破 | 09 |

## 患者独立 PIV 及信号放大

| 实验 | 内容 | 状态 / 结果范围 | 路线 |
|---|---|---|---|
| PIV0 | X49隔离、X21配方控制 | 八轮完成，X49 AP-primary选ep5 | 01/07 |
| PIV1 | Study evidence class-token | 八轮完成，AP-primary选ep4，小幅局部信号 | 03 |
| PIV2 | AP-heavy温和加权 | 八轮完成，未稳定获益 | 07 |
| PIV3 | SwinV2 AP专家 | 六轮完成，AP-primary选ep5；官方仅AP不当全Valid | 02/07 |
| R0_STUDY_MEAN_6EP | 放大链匹配mean control | 六轮完成 | 03 |
| RB_PIV1_MATCHED_6EP | 匹配PIV1机制、预算 | 六轮完成，未放大主信号 | 03 |
| R1_FROZEN_IMAGENET_6EP | 冻结原始公开预训练 | 六轮完成，不是冻结PIV1 | 09 |
| R2_FREEZE2_ENCODER01_6EP | 先冻结2轮、再低encoderLR | 六轮完成，低于R0/RB | 09 |
| R3 bounded residual | .25tanh+.01penalty | 门控跳过，无训练结果 | 03/04 |
| R4 | 固定PIV0/PIV1平均 | 零训练事后比较，不能称确认结果 | 11 |

## 中等类 MC 与固定 DINO DZ

| 实验 | 内容 | 本机最新同步状态 | 路线 |
|---|---|---|---|
| MC00 | R0固定raw ep6父模型 | 导出完成 | 01/11 |
| MC01 | 干净SwinV2-L父模型 | 六轮完成，C选ep4 | 02 |
| MC02 | Small弱配方互补成员 | 六轮完成，C选ep4 | 02/11 |
| MC10 | 四类冻结GAP residual | AMP修复后三轮完成；C选ep0 | 04 |
| MC11 | 匹配空间residual | AMP修复后三轮完成；C选ep2 | 04 |
| MC12 | 空间residual+相似异常匹配 | C门控未过，skipped_gate | 04/06 |
| MC20 | 逐类融合、Small融合、AP截距 | 阶段父模型结果有；最终含residual重算断言失败 | 11/12 |
| DZ0 | 原X42/X41基线重现 | 完成；精度版本差异有记录 | 11 |
| DZ1 | 原概率加DINO .2 | 完成，heavy+.000515 | 11 |
| DZ2 | 只DINO z+sigmoid | 完成，负向 | 11 |
| DZ3 | 线性双路z后sigmoid | 完成，heavy+.000479 | 11 |
| DZ4 | 双路各sigmoid(z/2)后融合 | 完成，heavy+.000444，区间跨零 | 11 |
| DZ Phase2 | 达门槛后再训练 | 未过门控，未启动 | 11 |

## 哪些材料只能算历史假设

报告中出现过的“Test可到.80–.83”“加分辨率会涨.01–.02”“所有head/loss全失败”“AP已到判别上限”等不是统一复核后的事实。未来讨论应先找到对应真实run和评价人群；找不到就列为提案，不补造已测分数。

主要来源：[报告索引与原始运行清单](14_run_evidence_inventory.md)、[实际X1–10审计](../report/CMS_X1_X10_implementation_audit_20261030.md)、[X21–30](../report/CMS_X21_X30_codex_impl_20261031.md)、[X43–49](../report/CMS_X43_X49_codex_impl_20261101.md)、[v3重审](../report/CMS_heavy9_v3_reassessment_20261106.md)、[A实际协议](../report/CMS_heavy9_v2_revised_queue_20261104.md)、[PIV](../report/CMS_PIV0_PIV3_analysis_20261106.md)、[MC状态](../runs/remote_4090_medium_class_20261106/queue_state.json)、[DZ诊断](../report/CMS_DINO_zscore_failure_diagnosis_20261107.md)。
