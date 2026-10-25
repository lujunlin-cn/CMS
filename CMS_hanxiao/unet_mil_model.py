#!/usr/bin/env python3
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from timm.models._features import FeatureListNet

MODEL_ROOT = Path(os.environ.get('CMS_MODEL_ROOT', '/home/zhaona/CMS_hanxiao/pretrain_model'))
MODEL_CONFIG = {
    'densenet121': ('densenet121.tv_in1k', 'densenet121'),
    'efficientnetv2_s': ('tf_efficientnetv2_s.in21k_ft_in1k', 'efficientnetv2_s'),
    'convnext_tiny': ('convnext_tiny.fb_in22k_ft_in1k', 'convnext_tiny'),
    'convnext_small': ('convnext_small.fb_in22k_ft_in1k', 'convnext_small'),
    'swin_small': ('swin_small_patch4_window7_224.ms_in22k_ft_in1k', 'swin_small'),
    'dinov2_base_reg4': ('vit_base_patch14_reg4_dinov2.lvd142m', 'dinov2_base_reg4'),
}


class EncoderFeatures(nn.Module):
    def __init__(self, alias):
        super().__init__()
        model_id, folder = MODEL_CONFIG[alias]
        checkpoint = MODEL_ROOT / folder / 'model.safetensors'
        if alias == 'dinov2_base_reg4':
            # DINOv2 was published at 518px; dynamic_img_size enables fixed 224px inputs
            # while retaining the original positional embedding and local checkpoint.
            self.encoder = timm.create_model(
                model_id, pretrained=False, features_only=True,
                dynamic_img_size=True, out_indices=(2, 5, 8, 11))
            state = load_file(str(checkpoint), device='cpu')
            core = getattr(self.encoder, 'model', None)
            if core is None:
                raise RuntimeError('DINOv2 feature wrapper has no underlying model')
            missing, unexpected = core.load_state_dict(state, strict=False)
            self.channels = list(self.encoder.feature_info.channels())[-4:]
        else:
            base = timm.create_model(model_id, pretrained=False)
            state = load_file(str(checkpoint), device='cpu')
            missing, unexpected = base.load_state_dict(state, strict=False)
            out_indices = tuple(range(len(base.feature_info)))
            self.encoder = FeatureListNet(base, out_indices=out_indices, flatten_sequential=True)
            self.channels = [int(x['num_chs']) for x in base.feature_info]
        unexpected = list(unexpected)
        if alias == 'swin_small':
            allowed_unexpected = [x for x in unexpected if x.endswith('attn_mask') or x.endswith('relative_position_index')]
            ignored_reason = 'Swin attention masks and relative-position indices are input-shape-derived buffers.'
        elif alias == 'dinov2_base_reg4':
            allowed_unexpected = [x for x in unexpected if x in {'norm.weight', 'norm.bias'}]
            ignored_reason = 'DINOv2 features-only intermediate maps do not use the final classifier normalization parameters; this is recorded explicitly.'
        else:
            allowed_unexpected = []
            ignored_reason = ''
        unexpected_trainable = [x for x in unexpected if x not in allowed_unexpected]
        if unexpected_trainable:
            raise RuntimeError(f'unexpected non-derived encoder keys for {alias}: {unexpected_trainable[:20]}')
        self.load_info = {
            'model_id': model_id,
            'checkpoint': str(checkpoint),
            'missing_keys': list(missing),
            'unexpected_keys': unexpected,
            'allowed_derived_or_unused_keys': allowed_unexpected,
            'unexpected_trainable_keys': unexpected_trainable,
            'ignored_keys_reason': ignored_reason,
            'feature_channels': self.channels,
        }

    def forward(self, x):
        features = list(self.encoder(x))
        converted = []
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4:
                raise RuntimeError(f'encoder feature must be 4D, got {tuple(feature.shape)}')
            if feature.shape[1] == channels:
                converted.append(feature)
            elif feature.shape[-1] == channels:
                converted.append(feature.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(f'feature/channel mismatch: {tuple(feature.shape)} vs {channels}')
        if len(converted) != len(self.channels):
            raise RuntimeError('encoder returned an unexpected number of feature maps')
        return converted


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        bottleneck = min(256, max(32, encoder_channels[-1] // 4))
        decoder_channels = [max(32, bottleneck // (2 ** (i + 1))) for i in range(len(encoder_channels) - 1)]
        self.first = ConvBlock(encoder_channels[-1], bottleneck)
        blocks = []
        in_channels = bottleneck
        for skip_channels, out_channels in zip(reversed(encoder_channels[:-1]), decoder_channels):
            blocks.append(ConvBlock(in_channels + skip_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_channels

    def forward(self, features):
        x = self.first(features[-1])
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


class UNetMIL(nn.Module):
    def __init__(self, encoder_alias, num_classes=10, topk_ratio=0.2):
        super().__init__()
        self.encoder_alias = encoder_alias
        self.topk_ratio = float(topk_ratio)
        self.encoder = EncoderFeatures(encoder_alias)
        self.decoder = UNetDecoder(self.encoder.channels)
        self.evidence_head = nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1)
        self.num_classes = num_classes

    @property
    def load_info(self):
        return self.encoder.load_info

    def masked_topk_mean(self, evidence, valid_masks):
        batch, classes, height, width = evidence.shape
        flat = evidence.flatten(2)
        mask = valid_masks.flatten(1).bool()
        outputs = []
        for b in range(batch):
            valid = mask[b]
            if int(valid.sum()) == 0:
                valid = torch.ones_like(valid)
            values = flat[b, :, valid]
            k = max(1, min(values.shape[-1], math.ceil(self.topk_ratio * values.shape[-1])))
            outputs.append(values.topk(k, dim=-1).values.mean(dim=-1))
        return torch.stack(outputs, dim=0)

    def forward(self, images, valid_masks=None):
        features = self.encoder(images)
        decoded = self.decoder(features)
        evidence = self.evidence_head(decoded)
        if valid_masks is None:
            valid_masks = torch.ones(images.shape[0], 1, images.shape[-2], images.shape[-1], device=images.device)
        valid_masks = F.interpolate(valid_masks.float(), size=evidence.shape[-2:], mode='nearest')
        image_logits = self.masked_topk_mean(evidence, valid_masks)
        return image_logits, evidence, valid_masks


def freeze_encoder(model):
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
