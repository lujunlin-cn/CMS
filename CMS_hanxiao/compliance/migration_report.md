# U-Net + MIL 合规改造记录

- 依据：`unet_mil_compliance_guide.md`。
- 已准备官方 Train/Valid Study manifest；Test 未进入 manifest。
- 已实现经核验的通用编码器、随机初始化 U-Net 解码器、10 通道空间证据头、几何 padding mask 和 masked Top-K MIL。
- Study 内聚合为图像 logits 的固定均值。
- 未引入外部医学图像、医学权重、教师、分割器、伪掩码或医学自监督阶段。
- 尚未启动正式训练；已完成 densenet、efficientnet、convnext、swin、DINOv2 的结构/梯度 smoke test。
- Swin 的 relative_position_index/attn_mask 是输入窗口派生 buffer；DINOv2 features-only 路径明确记录未使用最终分类 norm 参数，需在正式实验记录中保留。

- 按指南“不重新开展编码器选型”，旧六模型比较的定时启动已从 crontab 移除；比较脚本保留为手动复现实验入口。
- 当前未启动正式 U-Net + MIL 训练，等待明确的训练启动安排；磁盘剩余空间约 20 GB，启动前需再次检查。
