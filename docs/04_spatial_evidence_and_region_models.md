# 路线四：局部证据、区域建模、空间 attention 与层级 MIL

核心假设：全局分类器可能依赖严重程度、视角和设备等捷径；显式保留局部证据可能改善肺炎、肺不张、阴影等中等难度类别。做过很多空间近似，但它们不是同一种方法。

## 1. 早期 U-Net / decoder 分支

探索了 DenseNet121 U-Net+MIL、global、LSE、residual、local auxiliary、staged encoder/decoder，以及 ConvNeXt 多尺度残差 U3。

- 早期 U-Net global .71628、LSE .72727；global_v2 .77004、residual .77503，是不同修正版本的日志峰值。
- structural `local_aux` 对照 .770693、处理 .770904，差 +.000211，不足以认定有效。
- subject_v3 U0 decoder 可见 seed24/25 峰值约 .74308/.74052；频率平衡和 decoder 不能混在同一因果结论里。
- U0W 全局/残差比较约 .79091/.79026，只有 2,797 Study、DenseNet121 的局部协议，不与完整 ConvNeXt 的 .81 直接比。
- highres U-Net anchor/staged smoke 在 .52 左右并伴随中止；是接线/数值与小预算现场，不是完成训练的模型上限。
- U3 ConvNeXt-L multiscale residual 五轮峰值 .79864，未超当时主力。

没有真实病灶/器官区域监督，decoder 的空间输出不自动对应临床结构。

## 2. YOLO、crop 与图像级弱监督定位

YOLO global→WSL→真实多尺度 WSL 见路线02。X4 训练时中心心影 crop、X57 四边 crop、X71 底部 crop，以及 S2/X60 推理 crop 是不同假设，不能互为严格复现。已有有效结果总体不支持丢弃周围胸廓证据。

## 3. A1–A4：架构扩展的 Full / Freeze 双线

Full 从原始公开预训练启动、所有参数训练六轮；Freeze 从 X21 启动、仅新模块和分类头训练三轮。旧版 A1/A2/A3 Freeze 的 `resume=null`，分数约 .70–.72，属于初始化错误，不能分析收敛不足。

| 架构 | 研究目标 | Full 官方 Macro | 正确 X21 Freeze Macro |
|---|---|---:|---:|
| A1 Cross-View | Study 内图像之间 attention | .81021 | .81519 |
| A2 Patch Attention | 图内 patch 加权 | .80589 | .81220 |
| A3 Multi-Resolution FPN | 多层空间特征融合，fpn_dim128 | .79660 | .80105 |
| A4 Hierarchical MIL | Patch→4×4 Region→Study | .80442 | .79582 |

A1 属于跨图机制，不能拿它的小幅正点估计证明图内空间 attention 有效。A3/A4 的新 head 与 X21 cosine 兼容性修复记录清楚：保留骨干迁移，但不能暗示所有旧分类头参数完全等价复用。

## 4. X89 region-aware 与 X90 压缩

原设计描述 attention 选择区域，实际 X89 是 feature map 绝对值 top-k + class-wise region max。局部 region 接受 Study 标签，未证明每个 region 含该疾病。推理丢弃 region head，只留 global head，训练和部署路径不同。

X89 原图像 AUC .81987，正确 Study .81100，内部 fold 真 heavy9 .73482；X90 distill 后 Study .81194 /heavy .72666，正式 Test .7662。空间辅助没有形成官方 Test 突破；但该结果受内部患者重叠、错误旧 heavy 指标和实现偏离影响，不能一票否决所有区域学习。

X93 未加载 X89/X90 的教师输出或区域参数，实际是 X92 + 新区域头 + hard BCE；因此不是“融合 X89/X90 优势”的有效检验。

## 5. MC：把空间证据与普通 residual 区分

目标类 3/5/6/7：肺炎、心影扩大、肺不张、肺部阴影。固定干净 MC00 ConvNeXt 和 MC01 Swin 父模型；C 选、H 比较、官方 Valid 仅 secondary。

- MC10：冻结父模型，98,884 参数的 GAP residual 非空间对照。
- MC11：同初始化和预算，改成逐类 query 空间 residual；内容 mask 随几何变化，排除黑边。
- 两者保留原 global path，末层零初始化，改变量限制 .25*tanh；其他六类理论上应恒等。
- 每 Study 先 mean_prob 再计算四类 BCE，EMA 只更新 CUDA 上新增模块，三轮预算。
- MC12：仅当 C 空间 residual 超父模型 +.004、超 GAP +.002 才加匹配异常采样。

原 MC10/11 在首个 loss 的 AMP 概率 BCE 失败，修复后均完成三轮。本机最新同步结果：

| 选择结果 | 完整 Valid Macro | heavy9 |
|---|---:|---:|
| MC10 C 选 ep0，单成员 | .80212 | .74087 |
| MC10 + 0.5 Swin | .81250 | .75814 |
| MC11 C 选 ep2，单成员 | .80177 | .74015 |
| MC11 + 0.5 Swin | .81235 | .75801 |

C 上空间 heavy 提升仅约 .000022，MC12 门控跳过。最终 MC20 重算在未改六类恒等性断言失败：1247/1248 元素不逐位相同，最大绝对差约 .000346。尚未核明是数值前向路径还是参数变化；不能把它写成完整最终融合验收通过，也不在本次文档整理中修改实验。

## 仍待回答

多次“空间路线”改变了初始化、分类头、监督单位或推理 head，未必测到同一因果变量。MC 已比旧路线更接近受控测试，但结果没有显示大幅新证据。继续投入前，应回答：空间 attention 是否真的关注可辨认病灶、是否改变正负 pair margin、是否在独立患者同时胜过 GAP 对照，而不只产生更漂亮的 attention 图？

来源：[早期 U-Net/患者指南](../report/CMS_subject_balance_v3_guide.md)、[A 修复](../report/CMS_heavy9_v2_repair_audit_20261104.md)、[v3 重审](../report/CMS_heavy9_v3_reassessment_20261106.md)、[MC 计划](../report/CMS_medium_class_4090_experiment_plan_20261106.md)、[MC 最新完整 Valid 表](../runs/remote_4090_medium_class_20261106/official_valid_summary.md)、[MC20 失败日志](../runs/remote_4090_medium_class_20261106/MC20_FUSION_VIEW_DIAGNOSTICS/train.log)。
