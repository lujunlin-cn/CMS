#!/usr/bin/env python3
"""CPU smoke test for optional U-Net + MIL research components."""

import torch

from unet_mil_extensions import (
    AsymmetricLossMultiLabel,
    GatedAttentionMIL,
    evidence_regularizer,
    masked_logsumexp,
    masked_topk_mean,
)


def main() -> None:
    torch.manual_seed(20261024)
    b, classes, tokens, dim = 3, 10, 17, 32
    values = torch.randn(b, classes, tokens, requires_grad=True)
    mask = torch.zeros(b, tokens, dtype=torch.bool)
    mask[0, :17] = True
    mask[1, :5] = True
    # Intentionally keep the third bag empty to check the safe fallback.
    for fn in (masked_topk_mean, masked_logsumexp):
        pooled = fn(values, mask, **({"ratio": 0.2} if fn is masked_topk_mean else {"tau": 0.7}))
        assert pooled.shape == (b, classes)
        assert torch.isfinite(pooled).all()
        pooled.sum().backward(retain_graph=True)
    tokens_x = torch.randn(b, tokens, dim, requires_grad=True)
    mil = GatedAttentionMIL(dim, classes, hidden_dim=16)
    logits, attention = mil(tokens_x, mask)
    assert logits.shape == (b, classes)
    assert attention.shape == (b, classes, tokens)
    assert torch.isfinite(logits).all() and torch.isfinite(attention).all()
    logits.sum().backward()
    labels = torch.randint(0, 2, (b, classes), dtype=torch.float32)
    loss = AsymmetricLossMultiLabel()(logits, labels)
    assert torch.isfinite(loss)
    evidence = torch.randn(b, classes, 8, 8)
    valid = torch.ones(b, 1, 8, 8)
    reg = evidence_regularizer(evidence, valid, tv_weight=0.01, l1_weight=0.001)
    assert torch.isfinite(reg)
    print("unet_mil_extensions smoke test passed")


if __name__ == "__main__":
    main()

