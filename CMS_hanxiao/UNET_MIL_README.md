# U-Net + MIL 运行说明

先准备官方 Train/Valid Study manifest：

```bash
/home/zhaona/CMS_hanxiao/prepare_unet_mil_data.sh
```

数据只来自官方 `cxr_first/train.csv` 和 `valid.csv`。Train 中已确认的 1 张低信息图像通过审计 manifest 排除；Test 不生成训练 manifest。

先运行结构与梯度冒烟：

```bash
cd /home/zhaona/CMS_hanxiao
/home/zhaona/anaconda3/envs/ljhtorch/bin/python smoke_test_unet_mil.py --model densenet121
```

可切换 6 个已核验通用编码器：

```text
densenet121
efficientnetv2_s
convnext_tiny
convnext_small
swin_small
dinov2_base_reg4
```

单卡训练示例：

```bash
GPU=0 MODEL=densenet121 EPOCHS=5 \
  /home/zhaona/CMS_hanxiao/run_unet_mil_single_gpu.sh
```

模型结构：

```text
approved local generic encoder
  -> multiscale feature adapter
  -> randomly initialized U-Net decoder
  -> randomly initialized 10-channel spatial evidence head
  -> masked spatial Top-K mean MIL
  -> mean image logits within Study
  -> BCEWithLogitsLoss using official Study labels
```

输入中的 letterbox padding 通过几何有效区域 mask 排除；该 mask 不是医学肺部掩码，也不是伪标签。训练和验证保持 Study 级分组，验证集不进入梯度更新。

输出目录包含：`config.json`、`metrics.jsonl`、`best.pt`、`last.pt`、`done.ok` 或 `FAILED`。

正式训练前必须检查：

1. `compliance/asset_inventory.csv` 和 `weight_provenance.json` 的公开链接、许可证和 SHA-256；
2. `unet_mil_data/manifest_summary.json` 的 manifest 哈希；
3. smoke test 中 encoder 和 decoder 的非零梯度；
4. 不使用 Test 选择 checkpoint、MIL 比例或预处理参数。
