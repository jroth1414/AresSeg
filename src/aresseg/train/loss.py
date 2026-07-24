"""Segmentation loss: class-weighted cross-entropy + Dice, both ignoring the 255 label (MS2).

Dice complements CE under heavy class imbalance (soil dominates AI4Mars). Both terms compute over
valid pixels only — the ignore label (unlabeled / rover-self / >30 m range) never contributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, eps: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); target: (B, H, W)
        c = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
        tgt = target.clamp(0, c - 1)
        onehot = F.one_hot(tgt, c).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        probs = probs * valid
        onehot = onehot * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """``CE(weight) + dice_weight * Dice``, both with ``ignore_index``."""

    def __init__(self, class_weights=None, ignore_index: int = 255, dice_weight: float = 1.0):
        super().__init__()
        w = (
            torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
