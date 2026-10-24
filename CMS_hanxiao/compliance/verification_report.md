# U-Net + MIL 验证报告（准备阶段）

**时间**：2026-10-24

## 已通过

- `prepare_unet_mil_data.py` 运行成功。
- Train manifest：30,343 张图像、18,513 Study、11,055 Subject；排除 1 张已人工确认的低信息图像。
- Valid manifest：1,422 张图像、859 Study、228 Subject。
- Train/Valid manifest Subject 重叠：0；Study 重叠：0。
- Test 未写入 manifest。
- `prepare_unet_mil_data.py`、`unet_mil_model.py`、`train_unet_mil_single_gpu.py` 和 smoke test 均通过 `py_compile`。
- 六个编码器均完成结构与梯度 smoke test：DenseNet-121、EfficientNetV2-S、ConvNeXt-Tiny、ConvNeXt-Small、Swin-Small、DINOv2-Base-Reg4。
- 所有 smoke test 均产生 10 类 image logits、空间 evidence map，并验证 encoder 和 decoder 的非零梯度。

## 载入说明

- DenseNet、EfficientNet、ConvNeXt 的 checkpoint 完整加载，无 missing/unexpected learned keys。
- Swin 的 `relative_position_index` 和 `attn_mask` 是输入窗口派生 buffer，明确列为允许忽略项。
- DINOv2 features-only 中未使用最终分类 normalization 的 `norm.weight`/`norm.bias`，明确列为允许忽略项；编码器主体参数加载成功。
- 任何其他 unexpected trainable key 会直接抛错，不会由 `strict=False` 静默吞掉。

## 尚未完成

- 尚未启动正式训练。
- 尚未产生正式验证 Macro-AUC/Macro-F1。
- 尚未选择 checkpoint、阈值或 Test 推理参数。
- 尚未进行 T4 100 ms 端到端计时。

## 运行入口

```bash
/home/zhaona/CMS_hanxiao/prepare_unet_mil_data.sh
GPU=0 MODEL=densenet121 /home/zhaona/CMS_hanxiao/run_unet_mil_single_gpu.sh
```
