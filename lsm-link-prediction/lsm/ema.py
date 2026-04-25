"""Exponential Moving Average wrapper for model parameters."""

import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[n])
        self.backup = {}
