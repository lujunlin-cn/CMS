#!/usr/bin/env python3
"""Optional, auditable improvements on top of the frozen U-Net + MIL baseline.

The module deliberately reuses the already-audited encoder/decoder and adds only
randomly initialized, label-supervised pooling/regularization components.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_mil_model import UNetMIL


class ImprovedUNetMIL(UNetMIL):
    """U-Net evidence maps with selectable spatial MIL pooling.

    ``topk`` is the baseline. ``gated_attention`` implements class-wise gated
    attention over valid spatial positions, following Ilse et al. (2018). The
    attention parameters are randomly initialized and trained only from the
    official Study-level labels.
    """

    def __init__(
        self,
        encoder_alias: str,
        num_classes: int = 10,
        topk_ratio: float = 0.2,
        spatial_pool: str = "topk",
        attention_dim: int = 128,
        attention_temperature: float = 1.0,
        hybrid_alpha: float = 0.5,
    ):
        super().__init__(encoder_alias, num_classes=num_classes, topk_ratio=topk_ratio)
        allowed = {"topk", "gated_attention", "hybrid"}
        if spatial_pool not in allowed:
            raise ValueError(f"spatial_pool must be one of {sorted(allowed)}")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1]")
        self.spatial_pool = spatial_pool
        self.attention_temperature = float(attention_temperature)
        self.hybrid_alpha = float(hybrid_alpha)
        hidden = max(32, min(int(attention_dim), 256))
        self.attn_v = nn.Conv2d(self.decoder.out_channels, hidden, kernel_size=1)
        self.attn_u = nn.Conv2d(self.decoder.out_channels, hidden, kernel_size=1)
        self.attn_w = nn.Conv2d(hidden, num_classes, kernel_size=1)

    @staticmethod
    def _safe_mask(valid_masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        mask = F.interpolate(valid_masks.float(), size=(height, width), mode="nearest")
        mask = mask[:, 0].flatten(1).bool()
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty] = True
        return mask

    def _gated_attention_pool(
        self, decoded: torch.Tensor, evidence: torch.Tensor, valid_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, classes, height, width = evidence.shape
        valid = self._safe_mask(valid_masks, height, width)
        v = torch.tanh(self.attn_v(decoded))
        u = torch.sigmoid(self.attn_u(decoded))
        scores = self.attn_w(v * u).flatten(2) / self.attention_temperature
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(evidence.dtype)
        attention = attention * valid[:, None, :].to(attention.dtype)
        # Renormalize after masking to avoid tiny numerical drift under AMP.
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = (evidence.flatten(2) * attention).sum(dim=-1)
        valid_count = valid.sum(dim=-1, keepdim=True).to(attention.dtype).clamp_min(1)
        entropy = -(attention.float().clamp_min(1e-8) * attention.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / valid_count.float().log().clamp_min(1.0)
        return logits, attention.view(batch, classes, height, width), entropy

    def forward(self, images: torch.Tensor, valid_masks: Optional[torch.Tensor] = None, return_aux: bool = False):
        features = self.encoder(images)
        decoded = self.decoder(features)
        evidence = self.evidence_head(decoded)
        if valid_masks is None:
            valid_masks = torch.ones(
                images.shape[0], 1, images.shape[-2], images.shape[-1], device=images.device
            )
        valid_masks = F.interpolate(valid_masks.float(), size=evidence.shape[-2:], mode="nearest")

        aux: Dict[str, torch.Tensor] = {}
        if self.spatial_pool == "topk":
            image_logits = self.masked_topk_mean(evidence, valid_masks)
        else:
            attention_logits, attention, entropy = self._gated_attention_pool(decoded, evidence, valid_masks)
            aux["attention"] = attention
            aux["attention_entropy"] = entropy
            if self.spatial_pool == "gated_attention":
                image_logits = attention_logits
            else:
                topk_logits = self.masked_topk_mean(evidence, valid_masks)
                image_logits = self.hybrid_alpha * attention_logits + (1.0 - self.hybrid_alpha) * topk_logits
        if return_aux:
            return image_logits, evidence, valid_masks, aux
        return image_logits, evidence, valid_masks


class AsymmetricLoss(nn.Module):
    """ASL for multi-label logits (Ridnik et al., ICCV 2021).

    The loss handles positive and negative labels with separate focusing and
    clips easy negative probabilities. It is an optional alternative to BCE;
    do not combine it with ``pos_weight`` in the trainer.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0 or clip < 0 or clip >= 1:
            raise ValueError("invalid ASL parameters")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)
        prob_for_weight = pos_probs * targets + neg_probs * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        asymmetric_weight = (1.0 - prob_for_weight).clamp_min(1e-8).pow(gamma)
        log_pos = torch.log(pos_probs.clamp_min(1e-8))
        log_neg = torch.log(neg_probs.clamp_min(1e-8))
        loss = targets * log_pos + (1.0 - targets) * log_neg
        return -(asymmetric_weight * loss).mean()


def evidence_total_variation(evidence: torch.Tensor, valid_masks: torch.Tensor) -> torch.Tensor:
    """Small optional TV penalty; disabled by default because maps are not masks."""
    mask = valid_masks[:, :1].float()
    mask = F.interpolate(mask, size=evidence.shape[-2:], mode="nearest")
    dx = (evidence[:, :, :, 1:] - evidence[:, :, :, :-1]).abs()
    dy = (evidence[:, :, 1:, :] - evidence[:, :, :-1, :]).abs()
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    denom = mx.sum() + my.sum()
    if float(denom.detach().cpu()) == 0:
        return evidence.new_zeros(())
    return (dx * mx).sum() / denom + (dy * my).sum() / denom


class ModelEMA:
    """In-memory exponential moving average of trainable model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, shadow in self.shadow.items():
            value = current[name].detach()
            if not torch.is_floating_point(value):
                shadow.copy_(value)
            else:
                shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return {name: value.detach().cpu() for name, value in self.shadow.items()}
