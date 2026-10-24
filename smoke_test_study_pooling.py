#!/usr/bin/env python3
import torch
from study_pooling import ClasswiseImageAttentionPool, masked_mean_pool

torch.manual_seed(20261024)
tokens = torch.randn(5, 12, requires_grad=True)
logits = torch.randn(5, 10, requires_grad=True)
counts = [1, 2, 2]
pool = ClasswiseImageAttentionPool(12, 10, hidden_dim=8)
out, weights = pool(tokens, logits, counts)
assert out.shape == (3, 10)
assert all(w.shape[1] == 10 and torch.allclose(w.sum(0), torch.ones(10), atol=1e-6) for w in weights)
out.sum().backward()
assert torch.isfinite(tokens.grad).all() and any(p.grad is not None for p in pool.parameters())
# Permutation invariance inside each Study.
perm = torch.tensor([0, 2, 1, 4, 3])
out2, _ = pool(tokens.detach()[perm], logits.detach()[perm], counts)
assert torch.allclose(out, out2, atol=1e-5)
print('study pooling smoke test passed')
