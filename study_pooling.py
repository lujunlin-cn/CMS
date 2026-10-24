#!/usr/bin/env python3
"""Optional permutation-invariant Study-level image pooling.

This module is intentionally separate from the baseline trainer. It consumes
per-image decoder summaries and image logits, then pools only images belonging
to the same Study. It has no view-position prior and no external data path.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class ClasswiseImageAttentionPool(nn.Module):
    """Class-specific gated attention over the images of each Study.

    Parameters are randomly initialized and receive only Study-level loss.
    Input order inside each Study does not affect the output. ``counts`` is the
    collate-time list of real image counts; no padded image enters a Study mean.
    """

    def __init__(self, token_dim: int, num_classes: int = 10, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        if min(token_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("token_dim, num_classes and hidden_dim must be positive")
        self.num_classes = int(num_classes)
        self.v = nn.Linear(token_dim, hidden_dim)
        self.u = nn.Linear(token_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, image_tokens: torch.Tensor, image_logits: torch.Tensor, counts: Sequence[int]):
        if image_tokens.ndim != 2 or image_logits.ndim != 2:
            raise ValueError("image_tokens and image_logits must be [N,D] and [N,C]")
        if image_tokens.shape[0] != image_logits.shape[0]:
            raise ValueError("token/logit image count mismatch")
        if image_logits.shape[1] != self.num_classes:
            raise ValueError("image_logits class dimension mismatch")
        if any(int(c) <= 0 for c in counts) or sum(int(c) for c in counts) != image_logits.shape[0]:
            raise ValueError("counts must be positive and sum to the image count")
        hidden = self.dropout(torch.tanh(self.v(image_tokens)) * torch.sigmoid(self.u(image_tokens)))
        scores = self.score(hidden)  # [N,C]
        outputs: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        start = 0
        for count in counts:
            end = start + int(count)
            alpha = torch.softmax(scores[start:end], dim=0)  # permutation invariant set attention
            outputs.append((alpha * image_logits[start:end]).sum(dim=0))
            weights.append(alpha)
            start = end
        return torch.stack(outputs, dim=0), weights


def masked_mean_pool(image_logits: torch.Tensor, counts: Iterable[int]) -> torch.Tensor:
    """Reference Study mean used by the official baseline."""
    counts = [int(c) for c in counts]
    if image_logits.ndim != 2 or any(c <= 0 for c in counts) or sum(counts) != image_logits.shape[0]:
        raise ValueError("invalid image logits or counts")
    out = []
    start = 0
    for c in counts:
        out.append(image_logits[start:start + c].mean(dim=0))
        start += c
    return torch.stack(out)
