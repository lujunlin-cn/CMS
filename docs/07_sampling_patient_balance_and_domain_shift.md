# 路线七：Study/患者平衡、困难格子与分布偏移

核心问题：训练主要优化谁，评测主要考谁？增加困难病例的权重，是否带来新的证据，还是只重复有限的患者与标签？

## 1. 最早的频率和监督计权

同 Study 多图会在 image training 重复监督；同患者多 Study 又会产生相关曝光。项目分别试过：

- structural/subject_v3：频率组总权重、患者1/sqrt(n)、类别正例 sqrt 权重。
- subject_v3 B0/B1/B2/B3 配对峰值分别约 .75791→.76468、.75883→.76537、.75985→.76441、.76132→.76853。它们属于早期弱基线与特定 split，不能和 X21 的 .81457 直接比较。
- D0–D7：224 配方中 D1 Study 1/m_s 达 .80241，对 D0 .79791 增 .00450，但患者簇区间约 [-.00127,.01090]；没有稳健证明。
- D2 患者^-1/2 .79467；D3 稀有正例1.5 .79843；D5 患者 batch .79716；D6 D1×D2 .79952；D7 每患者封顶2 Study .79245；D7U 匹配 Study 预算 .79510。D5 的近似去冲突并不满足严格图像级患者无冲突协议，D7 还改变图像预算。

## 2. 在强配方上重新隔离采样

EB class-balanced+hybrid+Small 等捆绑失败不能单独否定采样。后来的隔离试验更有意义：

| 实验 | 改法 | 结果 |
|---|---|---|
| X27 | Study + class-balanced | .78627，下降 |
| X51 | image loss Study 1/m_s | .81213 /heavy约 .7613，未超 X21 |
| X52 | Study 原生等权 mean_prob | .81138 /heavy约 .7602，未超 X21 |
| X19 | pn 桶 GroupDRO | .80612 ep4；现存早版日志只有9轮，勿写10轮全完成 |
| X33 | 从预训练 heavy5× +within-heavy contrastive | .79776，下降且变量混杂 |
| X42 | X21 热启动 heavy5×，三轮 | .81546 ep1 /heavy .76607；AP-heavy .72694，未超过 X21 .72764 |
| X48 | Swin 热启动 heavy 过采样 | .80691 ep1，未超过 X41 |

X43 无 oversample 热启动补对照；没有它就不能把 X42 的千分位收益全归为采样。

## 3. AP-single×heavy 的定向尝试

统一元数据审计得到：Train 604/18,513（3.26%）、Valid255/859（29.69%）、Test612/1,414（43.28%）。Train front+lat×light 为8,231（44.46%），Test仅约9.83%。这些是历史诊断事实，不应继续用 Test 构成拟合新权重。

这个格子独立正例很少，Train 纵隔心影增大25、气胸46、实变62。气胸在 AP-heavy 已接近 .95，说明“AP 都不可识别”错误；纵隔、实变、肺炎、心影和阴影的瓶颈不同。

- **X56 修订版**：取消患者封顶，AP-heavy study5×、front_lat_light .5×；从X21热启动，lr3e-5、最多4ep。ep1 heavy .75997 <X21 .76430，按注册条件停止，macro .80996。不是八轮从头重训。
- **X58**：温和采样3×、X21热启动、drop_path .1、低LR、EMA .9995，三轮 .81143 /heavy .75885，负向。
- **X62**：sqrt(Test/Train cell density ratio)，裁剪[.5,3]，Study模式；.80692 /heavy .75723。用Test统计拟合有规则问题，该历史执行不能转为可提交推荐。
- **X64/X65**：不重复采样，分别 heavy loss /heavy rank；同样未超基线。
- **X74**：AP-heavy residual hotstart，DGX ep1 .81338，存在首轮停损记录；本地同步没有支持后续完整预算完成。

## 4. AP 专家与低相关模型

X1 AP-family 专家 .74794，与通用包同域约 .7528；X3 512进一步下降；X2路由也下降。结论是这条专家训练没有修复AP，不是证明AP存在固定判别上限。

PIV2 在祖先患者隔离下给 AP-heavy 加权，X49 AP .67613，对PIV0 .67545仅+.00068；PIV3 Swin AP专家 .67481。PIV1 .68444有局部信号，但官方AP下降。

放大研究 R0/RB/R1/R2 使用最多六轮：R1冻结ImageNet很低，R2先冻结2ep再encoderLR×.1也低，R3 bounded residual门控跳过。不能把R1当作冻结已训练PIV1后发生遗忘的证据。

## 5. 从“只攻最难类”转向中等类

MC 将目标定为3/5/6/7，依次做干净ConvNeXt/Swin/Small父模型、GAP vs空间 residual、条件匹配采样、低容量融合和视角偏置。它避免因最难类正例过少而只围绕同一批AP病例重复梯度。

但目前结果：MC10/11未显示空间相对GAP的实质收益；MC12跳过；阶段性逐类融合对heavy仅约+.000085，Small混入约+.000815。方向可解释，不代表收益已经出现。

## 讨论问题

重复采样提供的是权重，不是独立阳性数量。真正可调的对象是否应该是不同视角正负对的可比性、同异常背景中的局部证据，以及对中等类的净纠错？若新的验证阳性患者仍不足，应如何限制方法搜索，避免把采样随机波动认成机制？

来源：[D/R](../report/CMS_D0_D7_R1_R3_analysis_next_directions_20261027.md)、[X56 实测](../runs/remote_5090_x56_x60_20261101/results_summary.md)、[X61–63](../report/CMS_X61_X63_codex_impl_20261101.md)、[AP失败复核](../report/CMS_AP_failure_mechanism_vs_PIV_20261106.md)、[PIV](../report/CMS_PIV0_PIV3_analysis_20261106.md)、[中等类空间](../report/CMS_easy_medium_class_headroom_20261106.md)、[MC](../report/CMS_medium_class_4090_experiment_plan_20261106.md)。
