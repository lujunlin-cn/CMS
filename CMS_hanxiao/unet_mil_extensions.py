#!/usr/bin/env python3
"""Optional, label-only U-Net+MIL variants.

The module keeps the approved encoder -> random U-Net decoder -> spatial evidence
path and adds configurable spatial MIL pooling.  All added heads are randomly
initialized and trained only with the official Study labels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_mil_model import EncoderFeatures, UNetDecoder


class AdvancedUNetMIL(nn.Module):
    """U-Net + spatial MIL with optional class-specific attention pooling.

    ``spatial_pool`` options:
      * ``topk``: existing masked top-k mean of evidence logits;
      * ``mean``: masked spatial mean;
      * ``max``: masked spatial maximum;
      * ``logsumexp``: temperature-normalized soft max;
      * ``attention``: randomly initialized class-specific attention;
      * ``gated_attention``: gated attention (tanh * sigmoid).

    The default remains ``topk`` for comparability with the baseline.
    """
    POOLS = ("topk", "mean", "max", "logsumexp", "attention", "gated_attention")

    def __init__(self, encoder_alias, num_classes=10, topk_ratio=0.2,
                 spatial_pool="topk", pool_temperature=1.0,
                 attention_hidden=128, fusion="decoder"):
        super().__init__()
        if spatial_pool not in self.POOLS:
            raise ValueError(f"unknown spatial_pool={spatial_pool}; choose from {self.POOLS}")
        if not (0 < float(pool_temperature)):
            raise ValueError("pool_temperature must be > 0")
        self.encoder_alias = encoder_alias
        self.fusion = str(fusion)
        if self.fusion not in {"decoder", "global", "residual", "encoder_mil"}:
            raise ValueError("fusion must be decoder, global, residual, or encoder_mil")
        self.topk_ratio = float(topk_ratio)
        self.spatial_pool = spatial_pool
        self.pool_temperature = float(pool_temperature)
        self.encoder = EncoderFeatures(encoder_alias)
        self.decoder = UNetDecoder(self.encoder.channels)
        decoder_channels = self.decoder.out_channels
        # ``global`` and ``encoder_mil`` are deliberately lightweight controls:
        # they expose the pretrained final feature map without a random U-Net
        # decoder.  ``residual`` keeps the decoder as a gated local supplement.
        self.global_head = nn.Linear(self.encoder.channels[-1], num_classes)
        self.encoder_evidence_head = nn.Conv2d(self.encoder.channels[-1], num_classes, kernel_size=1)
        self.evidence_head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.decoder_gate = nn.Parameter(torch.full((num_classes,), -2.1972246))
        self.num_classes = int(num_classes)
        self.attention_hidden = int(attention_hidden)
        if spatial_pool == "attention":
            self.attention_head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        elif spatial_pool == "gated_attention":
            hidden = max(8, int(attention_hidden))
            self.attention_v = nn.Conv2d(decoder_channels, hidden, kernel_size=1)
            self.attention_u = nn.Conv2d(decoder_channels, hidden, kernel_size=1)
            self.attention_w = nn.Conv2d(hidden, num_classes, kernel_size=1)
        else:
            self.attention_head = None

    @property
    def load_info(self):
        return self.encoder.load_info

    @staticmethod
    def _safe_mask(valid_masks, n, spatial_size):
        mask = F.interpolate(valid_masks.float(), size=spatial_size, mode="nearest")
        mask = mask.flatten(2).squeeze(1).bool()
        # A malformed all-zero mask must not create NaNs; this fallback only
        # restores the geometric domain and does not introduce labels.
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty] = True
        return mask, F.interpolate(valid_masks.float(), size=spatial_size, mode="nearest")

    def _masked_topk(self, evidence, mask):
        flat = evidence.flatten(2)
        outputs = []
        for b in range(flat.shape[0]):
            values = flat[b, :, mask[b]]
            k = max(1, min(values.shape[-1], math.ceil(self.topk_ratio * values.shape[-1])))
            outputs.append(values.topk(k, dim=-1).values.mean(dim=-1))
        return torch.stack(outputs, dim=0)

    def _pool(self, evidence, decoded, mask):
        flat_evidence = evidence.flatten(2)
        if self.spatial_pool == "topk":
            return self._masked_topk(evidence, mask)
        if self.spatial_pool == "mean":
            weights = mask.unsqueeze(1).to(flat_evidence.dtype)
            return (flat_evidence * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)
        masked = flat_evidence.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if self.spatial_pool == "max":
            return masked.max(dim=-1).values
        if self.spatial_pool == "logsumexp":
            tau = self.pool_temperature
            # Subtract log(number of valid positions) to keep the scale close
            # to an average while retaining a smooth max.
            valid_count = mask.sum(-1, keepdim=True).clamp_min(1).to(flat_evidence.dtype)
            # Use the masked values; padding must never contribute to a
            # smooth-max pool.  ``masked`` has -inf at invalid locations.
            return tau * (torch.logsumexp(masked / tau, dim=-1) - valid_count.log())

        if self.spatial_pool == "attention":
            scores = self.attention_head(decoded).flatten(2)
        else:
            v = torch.tanh(self.attention_v(decoded))
            u = torch.sigmoid(self.attention_u(decoded))
            scores = self.attention_w(v * u).flatten(2)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights * flat_evidence).sum(dim=-1)

    def forward(self, images, valid_masks=None):
        features = self.encoder(images)
        if valid_masks is None:
            valid_masks = torch.ones(images.shape[0], 1, images.shape[-2], images.shape[-1], device=images.device)
        final = features[-1]
        final_mask = F.interpolate(valid_masks.float(), size=final.shape[-2:], mode="nearest")
        global_vec = (final * final_mask).sum(dim=(-2, -1)) / final_mask.sum(dim=(-2, -1)).clamp_min(1.0)
        global_logits = self.global_head(global_vec)
        if self.fusion == "global":
            return global_logits, None, final_mask
        if self.fusion == "encoder_mil":
            evidence = self.encoder_evidence_head(final)
            mask, resized_masks = self._safe_mask(valid_masks, images.shape[0], evidence.shape[-2:])
            image_logits = self._pool(evidence, final, mask)
            return image_logits, evidence, resized_masks
        decoded = self.decoder(features)
        evidence = self.evidence_head(decoded)
        mask, resized_masks = self._safe_mask(valid_masks, images.shape[0], evidence.shape[-2:])
        local_logits = self._pool(evidence, decoded, mask)
        if self.fusion == "residual":
            image_logits = global_logits + torch.sigmoid(self.decoder_gate).to(local_logits.dtype) * local_logits
        else:
            image_logits = local_logits
        return image_logits, evidence, resized_masks

class AsymmetricLossMultiLabel(nn.Module):
    """Optional ASL loss (Ridnik et al., ICCV 2021).

    Use as a replacement for BCE; do not combine with ``pos_weight`` in a
    single-factor ablation.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0 or not 0 <= clip < 1:
            raise ValueError("invalid ASL parameters")
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = float(gamma_neg), float(gamma_pos), float(clip), float(eps)

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")
        targets = targets.to(dtype=logits.dtype)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp_min(self.eps)) + (1.0 - targets) * torch.log(xs_neg.clamp_min(self.eps))
        # The focusing factor is detached, matching the reference ASL recipe.
        with torch.no_grad():
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = (1.0 - pt).clamp_min(self.eps).pow(gamma)
        return -(weight * loss).mean()


def evidence_regularizer(evidence, valid_mask, tv_weight=0.0, l1_weight=0.0):
    """Optional weak evidence-map regularization, disabled by default.

    It is a diagnostic/regularization term, never a segmentation objective;
    weights should remain very small because small abnormalities are valid.
    """
    if tv_weight < 0 or l1_weight < 0:
        raise ValueError("regularization weights must be non-negative")
    mask = valid_mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask, size=evidence.shape[-2:], mode="nearest")
    prob = torch.sigmoid(evidence)
    denom = mask.sum().clamp_min(1.0)
    l1 = (prob.abs() * mask).sum() / denom / max(evidence.shape[1], 1)
    pair_x = mask[..., :, 1:] * mask[..., :, :-1]
    pair_y = mask[..., 1:, :] * mask[..., :-1, :]
    tv_x = ((prob[..., :, 1:] - prob[..., :, :-1]).abs() * pair_x).sum() / pair_x.sum().clamp_min(1.0)
    tv_y = ((prob[..., 1:, :] - prob[..., :-1, :]).abs() * pair_y).sum() / pair_y.sum().clamp_min(1.0)
    tv = 0.5 * (tv_x + tv_y) / max(evidence.shape[1], 1)
    return float(tv_weight) * tv + float(l1_weight) * l1

# Generic token-level utilities used by the offline smoke test.  They are
# independent of the CNN wrapper above and do not load data or weights.
def _as_bool_mask(mask, length):
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim != 2:
        raise ValueError(f"valid mask must be [B,L] or [B,1,L], got {tuple(mask.shape)}")
    if mask.shape[-1] != length:
        raise ValueError(f"mask length {mask.shape[-1]} does not match token length {length}")
    return mask.bool()


def masked_logsumexp(values, valid_mask, tau=1.0):
    if values.ndim != 3 or tau <= 0:
        raise ValueError("values must be [B,C,L] and tau must be positive")
    mask = _as_bool_mask(valid_mask, values.shape[-1]).unsqueeze(1)
    safe = (values / tau).masked_fill(~mask, torch.finfo(values.dtype).min)
    out = tau * torch.logsumexp(safe, dim=-1)
    return torch.where(mask.any(dim=-1), out, torch.zeros_like(out))


def masked_topk_mean(values, valid_mask, ratio=0.2):
    if values.ndim != 3 or not 0 < ratio <= 1:
        raise ValueError("values must be [B,C,L] and ratio must be in (0,1]")
    mask = _as_bool_mask(valid_mask, values.shape[-1])
    result = []
    for b in range(values.shape[0]):
        if not bool(mask[b].any()):
            result.append(torch.zeros(values.shape[1], device=values.device, dtype=values.dtype))
            continue
        selected = values[b, :, mask[b]]
        k = max(1, min(selected.shape[-1], int(torch.ceil(torch.tensor(ratio * selected.shape[-1])).item())))
        result.append(selected.topk(k, dim=-1).values.mean(dim=-1))
    return torch.stack(result, dim=0)


class GatedAttentionMIL(nn.Module):
    """Class-specific gated attention over generic [B,L,D] tokens."""
    def __init__(self, in_dim, num_classes, hidden_dim=128, dropout=0.0):
        super().__init__()
        if min(in_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("dimensions must be positive")
        self.v = nn.Linear(in_dim, hidden_dim)
        self.u = nn.Linear(in_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier_weight = nn.Parameter(torch.empty(num_classes, in_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(self, tokens, valid_mask):
        mask = _as_bool_mask(valid_mask, tokens.shape[1])
        hidden = self.dropout(torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens)))
        scores = self.attention(hidden).transpose(1, 2)
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        nonempty = mask.any(dim=-1)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(nonempty[:, None, None], weights, torch.zeros_like(weights))
        pooled = torch.einsum("bcl,bld->bcd", weights, tokens)
        logits = torch.einsum("bcd,cd->bc", pooled, self.classifier_weight) + self.classifier_bias
        return logits, weights

class StudyAttentionPool(nn.Module):
    """Permutation-invariant, class-specific attention over one Study's images.

    ``image_logits`` is ``[V,C]`` for one Study. The scorer is randomly
    initialized and trained only through the official Study-level objective.
    """
    def __init__(self, num_classes: int = 10, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        if num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("num_classes and hidden_dim must be positive")
        layers = [nn.Linear(num_classes, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.scorer = nn.Sequential(*layers)

    def forward(self, image_logits: torch.Tensor, return_weights: bool = False):
        if image_logits.ndim != 2:
            raise ValueError(f"image_logits must be [V,C], got {tuple(image_logits.shape)}")
        if image_logits.shape[0] == 0:
            raise ValueError("Study must contain at least one image")
        # scores[V,C] -> class-wise softmax over the true image count V.
        scores = self.scorer(image_logits)
        weights = torch.softmax(scores.transpose(0, 1), dim=-1).transpose(0, 1)
        pooled = (weights * image_logits).sum(dim=0)
        return (pooled, weights) if return_weights else pooled
