# 胸部 X 光分类比赛：预训练模型候选与基准选型

> 核对日期：2026-10-23。
>
> 适用赛事：2026 年第八届全国高校计算机能力挑战赛，人工智能挑战赛。
>
> 当前任务：下载 6 个明确的非医学预训练权重，使用官方训练数据做基础有监督微调，在官方验证集比较后选定基准模型。
>
> 本文件是团队的选型建议，不是官方比赛规则。不修改已有的 `比赛规则.md`。
>
> 尚未得到这些模型在本赛事上的训练结果、验证指标或 T4 推理耗时。不得将候选顺序写成已验证的性能排名。

## 1. 结论

首轮候选固定为：DenseNet-121、EfficientNetV2-S、ConvNeXt-Tiny V1、ConvNeXt-Small V1、Swin-Small、DINOv2 ViT-B/14（4 个 register tokens）。

候选覆盖传统 CNN、现代 CNN、层级 Transformer 和通用自监督视觉表征。保留 ConvNeXt 的两个规模，是为了检查同一模型家族增加容量是否值得。

现在比较的是“公开预训练权重 + 基础微调”的实际效果。各候选的预训练数据量不同，因此不能把比较结果解释成单纯由网络架构造成的差异。

当前阶段不预定最终基准模型。不叠加复杂损失、蒸馏、多模型集成或测试时增强。先建立可以复现的单模型比较结果。

## 2. 比赛约束与选型边界

以下要求来自比赛规则 [R1][R2]：

- 只能选择非医学领域的公开预训练视觉权重。
- 权重须允许赛事使用与再分发；不能选择 NC 许可证或需授权的 gated 权重。
- 只能用赛方允许的数据进行训练；不能用外部医学数据。
- 禁止对医学数据做二次预训练或自监督续训。允许使用已经在通用数据上自监督预训练的公开权重，再用赛方数据进行有监督微调。
- 在选型阶段，保留官方验证集用于模型选择，不能先并入训练再用来比较模型。
- 测试集不能用于训练、伪标签、统计量拟合、阈值选择或 checkpoint 选择。
- 最终实际推理权重及必要配套文件解压后合计不超过 500 MB。
- 官方复测 GPU 为 NVIDIA T4 16 GB。官网的 100 ms 与 PDF 的 100 秒时限仍冲突，不能认定已经统一。
- 主要指标为 Study 级 Macro-AUC；Macro-F1 只在主指标相等时参与二次排序。

**500 MB 不是本地全部候选权重的下载总量限制。** 可以保留全部候选用于研究。若最终做集成，则必须对实际加载的全部推理文件去重后求和。

## 3. 固定候选与精确权重 ID

下表参数量来自发布者模型卡。文件大小来自公开文件列表，采用页面约数，单位 MB；不是本地实测体积。原始分类头和缓冲区可能使文件大小与“参数量 × 4 字节”不同。[M1]–[M6]

| 别名 | 模型 | 精确 `timm` ID | 主要预训练数据 | 参数量约数 | `model.safetensors` 页面大小约数 |
|---|---|---|---|---:|---:|
| `densenet121` | DenseNet-121 | `densenet121.tv_in1k` | ImageNet-1k | 8.0 M | 32.3 MB |
| `efficientnetv2_s` | EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | ImageNet-21k → ImageNet-1k | 21.5 M | 86.5 MB |
| `convnext_tiny` | ConvNeXt-Tiny V1 | `convnext_tiny.fb_in22k_ft_in1k` | ImageNet-22k → ImageNet-1k | 28.6 M | 114 MB |
| `convnext_small` | ConvNeXt-Small V1 | `convnext_small.fb_in22k_ft_in1k` | ImageNet-22k → ImageNet-1k | 50.2 M | 201 MB |
| `swin_small` | Swin-Small | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | ImageNet-22k → ImageNet-1k | 49.6 M | 200 MB |
| `dinov2_base_reg4` | DINOv2 ViT-B/14，4 个 register tokens | `vit_base_patch14_reg4_dinov2.lvd142m` | LVD-142M，通用图像自监督预训练 | 86.6 M | 346 MB |

对应 Hugging Face 仓库名均为 `timm/` 加上表中的完整 ID。

只下载一份 `model.safetensors` 时，6 个权重文件的页面约数合计为 **979.8 MB，约 0.98 GB**。模型卡、配置、许可证归档和下载缓存还会占用磁盘。此数值不是训练存储需求估计。

### 3.1 每个候选的作用

**DenseNet-121：小规模对照。** 用于检验更大模型的收益是否足以抵消额外成本。这里选择原始 ImageNet 权重，不使用任何胸片训练版本。参数少不能直接推导出实际延迟最低。

**EfficientNetV2-S：较小的现代 CNN 候选。** 选择 ImageNet-21k 预训练并经 ImageNet-1k 微调的具体权重。`tf_` 表示此权重对应的 TensorFlow 来源实现细节；发布者已转换为 PyTorch，不要求使用 TensorFlow 训练。[M2]

**ConvNeXt-Tiny V1：现代 CNN 候选。** 用于与 EfficientNetV2-S 和 Transformer 比较。这里不是 ConvNeXt V2。[M3]

**ConvNeXt-Small V1：容量对照。** 与 Tiny 同时训练，检查更大容量在官方数据上是否带来可复现的收益。不能预设 Small 一定优于 Tiny。[M4]

**Swin-Small：层级 Transformer 候选。** 用于比较窗口注意力模型与 CNN 的迁移效果。输入尺寸和窗口设置必须与实现兼容。[M5]

**DINOv2 ViT-B/14：通用自监督表征候选。** 选择标准 DINOv2 加 registers 的权重，而不是医学变体。官方模型卡明确了 LVD-142M、Apache-2.0 和 14 像素 patch 的配置。[M6][U4]

以上“候选作用”是实验设计建议，不是已验证的医学识别能力结论。

## 4. 许可证记录与排除项

### 4.1 公开许可证据

| 候选 | 所选 Hub 模型卡的许可证标记 | 上游仓库许可证 | 记录要求 |
|---|---|---|---|
| DenseNet-121 | Apache-2.0 | torchvision：BSD-3-Clause | 同时保留模型卡与上游 LICENSE |
| EfficientNetV2-S | Apache-2.0 | TensorFlow TPU：Apache-2.0 | 保留转换来源和上游 LICENSE |
| ConvNeXt-Tiny/Small V1 | Apache-2.0 | Meta ConvNeXt：MIT | 保留两层声明，不把上游许可简单改写成 Apache |
| Swin-Small | MIT | Microsoft Swin-Transformer：MIT | 保留模型卡与 MIT 全文 |
| 通用 DINOv2 | Apache-2.0 | Meta DINOv2：Apache-2.0 | 保留标准 DINOv2 权重许可声明和 MODEL_CARD |

依据：[M1]–[M6]、[U1]–[U6]。

上游代码许可证本身不等于对所有衍生权重的授权证明。模型卡的权重许可标记、权重来源、上游声明和训练数据说明必须一起保留。

这些材料支持将 6 个明确权重列入候选，但不等于主办方已审批。若某个权重的材料链仍有疑义，应在投入完整训练前向主办方核验。

### 4.2 本轮不纳入

- **ConvNeXt V2 官方预训练/微调权重**：官方仓库明确将这部分权重列为 CC-BY-NC。不能因代码为 MIT，就把权重当作无 NC 限制。[X1]
- **DINOv3 官方 ViT-B/16 LVD 权重**：官方模型页要求登录并接受访问条件。不符合本次“无需登录或授权”的公开获取要求。第三方重新上传不能自动解决来源合规问题。[X2]
- **医学预训练模型或医学续训变体**：不符合本赛题的预训练数据限制。[R2]

尤其不要因模型名称包含 `dino`，就将 XRay-DINO、其他医学 DINO 与表中的通用 DINOv2 混用。当前 DINOv2 仓库也包含医学模型条目，许可范围不能按整个仓库统一推断。[U4]

## 5. 下载与归档

### 5.1 文件说明

本材料包包括：

```text
pretrained_model_selection/
├── model_selection.md
├── models.json
├── download_models.py
├── test_download_models.py
└── requirements-download.txt
```

`models.json` 固定了模型名称、训练数据说明、许可证标记及材料来源。首次实际下载时，脚本解析完整 commit SHA，并写入对应候选的 `checkpoint.lock.json`。后续执行复用该版本，不自动跟随 `main` 更新。

脚本只下载 `model.safetensors`、`config.json`、`README.md`，避免同时下载内容重复的 `.bin` 权重。

下载器还会匿名检查 `gated`、`private` 和许可证字段；计算本地 SHA-256；与 Hub 提供的 LFS SHA-256 及文件长度核对；保存上游许可证、说明文件及来源记录。

这些检查不能自动证明训练数据绝无违规，也不能代替赛事合规审核。

### 5.2 执行步骤

前置条件：在训练服务器的既有 Python 环境中运行；网络能够访问 Hugging Face 与 GitHub raw。下载全部权重预计传输约 0.98 GB，不含重试和额外缓存。以下步骤不会安装或替换 PyTorch。

1. 进入解压后的材料目录。

   ```bash
   cd pretrained_model_selection
   ```

2. 安装下载器依赖。

   ```bash
   python -m pip install -r requirements-download.txt
   ```

3. 查看下载计划。该命令不访问网络。

   ```bash
   python download_models.py
   ```

4. 下载全部候选。

   ```bash
   python download_models.py --download --output /YOUR/PROJECT/pretrain_model
   ```

将 `/YOUR/PROJECT/pretrain_model` 替换为真实保存路径。

单独重试指定候选的示例：

```bash
python download_models.py --download \
  --output /YOUR/PROJECT/pretrain_model \
  --only convnext_tiny dinov2_base_reg4
```

不要并行运行两个下载器实例写入同一个输出目录。如果许可证变更、权重哈希不一致或变为 gated，脚本会报错。不得直接删去校验逻辑绕过错误。

### 5.3 预期归档结构

以下是脚本设计输出，不表示这些文件已经在你们的服务器生成：

```text
pretrain_model/
└── convnext_tiny/
    ├── model.safetensors
    ├── config.json
    ├── README.md
    ├── checkpoint.lock.json
    ├── download_record.json
    └── evidence/
        ├── upstream_LICENSE.txt
        ├── timm_LICENSE.txt
        ├── upstream_README.md
        └── sources.json
```

不同候选的证据文件略有区别。模型配置、模型卡和权重按同一个 Hub commit 下载。外部上游文档按获取时间归档并计算哈希，不能将这些外部文档误称为与 Hub commit 同版本。

### 5.4 当前验证状态

- 已核对发布者模型卡、公开文件列表和上游许可证页面。
- 已执行下载器语法检查。
- 已执行 8 项离线单元测试，均通过。
- 已执行无网络的下载计划输出。
- **未完成真实权重下载**：文档生成环境的外网 DNS 访问失败。
- **未验证模型加载、前向推理、迁移学习效果或 T4 耗时**。
- 以上限制不应被改写成“6 个模型已成功下载并通过训练测试”。

## 6. 首轮比较协议：仅用于选基准模型

以下是团队实验建议，不是新增比赛规则。

### 6.1 固定比较口径

- 使用同一份官方训练集与官方验证集。核对 `Subject_id` 与 `Study_id`，检查重复记录和患者重叠；发现问题时记录并处理，不能偷偷重划官方验证集。
- 每个候选都采用基础有监督微调。不要把“只训练分类头”与“全参数微调”的结果放在同一列直接判优。
- 输出固定为 10 个独立类别的 logits/probability，类别顺序沿用 `比赛规则.md`。
- 采用同一份数据读取、几何预处理、基础损失、训练预算和验证代码。模型原有的归一化配置分别记录，不能静默改成同一组不适配的参数。
- 若同一学习率导致某类模型未收敛，不能直接据此淘汰。可以在实验开始前定义同规模的小型学习率候选集合；不要对某一个模型无限追加调参预算。
- 评估粒度必须是 Study。多图检查的聚合方式在比较前固定；所有候选使用同一聚合规则。
- 第一阶段不使用模型集成、TTA、伪标签、蒸馏或新设计的复杂损失。
- 选择依据来自验证集。不能按测试集排行榜反复选模型或调参。

建议使用“同样的数据暴露量/优化更新预算”作为首轮控制项，并额外记录 GPU 时长。固定更新预算与固定 GPU 时长不是同一实验问题，不要混写。

### 6.2 分辨率处理

为了控制首轮资源，可以将 `224 × 224` 设为筛选分辨率。这个尺寸是建议，不是官方要求，也不是对胸片最佳尺寸的判断。

不能只改图片 resize，而忽略模型本身的输入约束：

- Swin 必须检查输入尺寸与窗口实现。
- 本轮 DINOv2 的 `timm` 配置默认是 `518 × 518`。如用 `224 × 224` 筛选，需要显式配置模型尺寸并确认位置编码加载/插值成功。[M6][I3]
- DINOv2 的 patch size 为 14。更高分辨率复核时，选择 14 的整数倍，并检查具体实现；不要盲目复用 CNN 的尺寸。[U4]
- EfficientNetV2-S 的公开模型卡列出训练尺寸 300、测试尺寸 384。统一到 224 的筛选结果不能作为该权重在更高分辨率下的最终结论。[M2]

入围模型必须在计划采用的最终分辨率上复核。低分辨率初筛只能用于减少候选，不能宣称穷尽了每个模型的能力。

### 6.3 选型顺序

1. 完成 6 个候选的合规材料归档与加载检查。
2. 在同一基础监督微调流程下各运行 1 个固定随机种子。
3. 按验证集 Macro-AUC、逐类 AUC 和工程约束筛出 2–3 个候选。
4. 对入围候选复核计划使用的分辨率。
5. 在固定配置下使用 3 个随机种子复跑入围候选。
6. 依据 Macro-AUC 的均值、波动和部署可行性确定主基准。
7. 固定主基准后，再开展训练方法优化。

随机种子数和候选数量是资源控制建议，不是统计充分性的保证。

### 6.4 最低结果记录项

```text
model_alias
repo_id
pretrain_revision
pretrain_sha256
train_config_hash
seed
input_size
patient_overlap_check
study_aggregation
val_macro_auc
val_auc_per_class
val_macro_f1
epoch_or_update_selected
gpu_training_hours
peak_training_memory_gb
inference_weight_bytes
t4_end_to_end_ms
latency_rule_status
```

未测量的字段写 `NOT_MEASURED`，不要填写估计成绩。

先排除不合规、文件超限或无法运行的候选，再比较验证集表现。若多个候选的差异不稳定，保留 1 个主基准和 1 个候补，不要把一次训练的微小优势写成确定结论。

正式推理时限未澄清前，同时保留 100 ms 和 100 秒两种口径的风险记录。初筛只能在训练/验证数据上测工程性能；不能借用官方计时测试样本做模型优化。

## 7. 给 Coding Agent 的任务边界

当前目标是完成候选下载和基础微调对比，不是提前设计最终冲榜系统。

交付物应包括：

- 6 个候选的下载及合规材料记录。
- 可复现的基础训练配置。
- Study 级验证预测和逐类指标。
- 训练成本、推理体积和实际测速记录。
- 基于实测数据的主基准/候补模型建议。

禁止用公开 ImageNet 成绩代替本赛事验证结果。禁止将尚未运行的训练、下载或测速写成完成。禁止为追求更新的模型，自动替换成 NC/gated 或医学权重。

## 8. 来源

规则来源：

- [R1] 官方赛题页：`https://www.ncccu.org.cn/index/Paper/case1.html#cs4`
- [R2] 《比赛参赛规则》PDF，第 1–8 条及附录 A：`https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/9c2194e6968d044c785e166df10c8255.pdf`

候选模型卡与文件列表：

- [M1] `https://huggingface.co/timm/densenet121.tv_in1k`；文件列表为同路径 `/tree/main`。
- [M2] `https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k`；文件列表为同路径 `/tree/main`。
- [M3] `https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k`；文件列表为同路径 `/tree/main`。
- [M4] `https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k`；文件列表为同路径 `/tree/main`。
- [M5] `https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k`；文件列表为同路径 `/tree/main`。
- [M6] `https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m`；文件列表为同路径 `/tree/main`。

上游许可与说明：

- [U1] torchvision BSD-3-Clause：`https://raw.githubusercontent.com/pytorch/vision/main/LICENSE`
- [U2] TensorFlow TPU Apache-2.0：`https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE`
- [U3] ConvNeXt V1 MIT：`https://github.com/facebookresearch/ConvNeXt`；全文 `https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/LICENSE`
- [U4] DINOv2：`https://github.com/facebookresearch/dinov2`；MODEL_CARD `https://raw.githubusercontent.com/facebookresearch/dinov2/main/MODEL_CARD.md`
- [U5] Swin MIT：`https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/LICENSE`
- [U6] timm Apache-2.0：`https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/LICENSE`

排除项：

- [X1] ConvNeXt V2 官方 License 段落：`https://github.com/facebookresearch/ConvNeXt-V2`
- [X2] DINOv3 官方 gated 模型页：`https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`

实现参考：

- [I1] Hugging Face 下载与固定 revision：`https://huggingface.co/docs/huggingface_hub/guides/download`
- [I2] HfApi：`https://huggingface.co/docs/huggingface_hub/package_reference/hf_api`
- [I3] 所选 DINOv2 输入配置：`https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m/raw/main/config.json`

---

**当前决策：下载明确的 6 个权重，先做可复现的基础监督微调比较；最终基准由官方验证集实测表现决定。**
