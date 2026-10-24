#!/usr/bin/env python3
import argparse
import torch
from unet_mil_model import MODEL_CONFIG, UNetMIL

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='densenet121', choices=sorted(MODEL_CONFIG))
parser.add_argument('--size', type=int, default=224)
args = parser.parse_args()
model = UNetMIL(args.model, num_classes=10, topk_ratio=0.2)
model.train()
images = torch.randn(2, 3, args.size, args.size)
valid_masks = torch.ones(2, 1, args.size, args.size)
image_logits, evidence, mask = model(images, valid_masks)
study_logits = image_logits.view(1, 2, 10).mean(dim=1)
target = torch.zeros_like(study_logits); target[0, 9] = 1
loss = torch.nn.functional.binary_cross_entropy_with_logits(study_logits, target)
loss.backward()
enc_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad and p.grad is not None]
dec_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
assert evidence.shape[1] == 10
assert image_logits.shape == (2, 10)
assert enc_grads and max(float(g.abs().max()) for g in enc_grads) > 0
assert dec_grads and max(float(g.abs().max()) for g in dec_grads) > 0
print({'model': args.model, 'image_logits': list(image_logits.shape), 'evidence': list(evidence.shape), 'loss': float(loss), 'encoder_gradient_max': max(float(g.abs().max()) for g in enc_grads), 'decoder_gradient_max': max(float(g.abs().max()) for g in dec_grads), 'load_info': model.load_info})
