"""
Loss functions for semantic segmentation.

Implements:
    CrossEntropyLoss  — standard pixel-wise cross-entropy
    DiceLoss          — manually implemented multi-class Dice loss
    CombinedLoss      — weighted sum of CE + Dice
"""

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


# ── Cross-Entropy ─────────────────────────────────────────────────────────────

class CrossEntropyLoss(nn.Module):
    """Pixel-wise cross-entropy loss (wraps nn.CrossEntropyLoss for a consistent API)."""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W)    integer class labels
        """
        return self.ce(logits, targets)


# ── Dice Loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss implemented from scratch.

    For each class c:
        Dice_c = (2 * sum(p_c * y_c) + smooth) / (sum(p_c) + sum(y_c) + smooth)
    Final loss = 1 - mean(Dice_c over all classes)

    Softmax is applied internally; logits are expected as input.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) integer labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        targets_one_hot = F.one_hot(safe_targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Zero out ignored pixels
        mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask

        # Flatten spatial dims: (B, C, H*W)
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)          # (B, C)
        cardinality  = probs_flat.sum(dim=2) + targets_flat.sum(dim=2) # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted combination: loss = (1 - w) * CE + w * Dice

    Args:
        dice_weight: weight of the Dice term (0 → pure CE, 1 → pure Dice)
    """

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_loss  = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loss_fn(cfg: Config) -> nn.Module:
    """Instantiate the loss function specified in cfg.loss_type."""
    if cfg.loss_type == "ce":
        return CrossEntropyLoss()
    elif cfg.loss_type == "dice":
        return DiceLoss(smooth=cfg.dice_smooth)
    elif cfg.loss_type == "combined":
        return CombinedLoss(dice_weight=cfg.dice_weight, smooth=cfg.dice_smooth)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type!r}. Choose 'ce', 'dice', or 'combined'.")
