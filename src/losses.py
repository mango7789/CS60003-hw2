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
    Final loss = 1 - weighted_mean(Dice_c over present classes)

    Softmax is applied internally; logits are expected as input.
    Pass class_weights (C,) to up-weight rare classes (inverse-frequency weighting).
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

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

        # Flatten to (C, B*H*W) so Dice is computed per class across the whole batch.
        # Per-image-per-class averaging gives absent classes a free Dice=1 (smooth/smooth),
        # which kills gradients for rare classes.
        probs_flat   = probs.permute(1, 0, 2, 3).reshape(num_classes, -1)           # (C, N)
        targets_flat = targets_one_hot.permute(1, 0, 2, 3).reshape(num_classes, -1) # (C, N)

        intersection   = (probs_flat * targets_flat).sum(dim=1)                      # (C,)
        cardinality    = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)             # (C,)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Only average over classes that actually appear in this batch.
        present = targets_flat.sum(dim=1) > 0
        if present.any():
            d = dice_per_class[present]
            if self.class_weights is not None:
                w = self.class_weights[present]
                dice_loss = 1.0 - (d * w).sum() / w.sum()
            else:
                dice_loss = 1.0 - d.mean()
        else:
            dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted combination: loss = (1 - w) * CE + w * Dice

    Args:
        dice_weight: weight of the Dice term (0 → pure CE, 1 → pure Dice)
        class_weights: per-class weights for the Dice term (inverse-frequency)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_loss   = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loss_fn(cfg: Config, class_counts=None) -> nn.Module:
    """
    Instantiate the loss function specified in cfg.loss_type.

    Args:
        class_counts: optional array/tensor of per-class pixel counts from the
                      training split (returned by get_dataloaders). When provided,
                      median-frequency weights are computed and passed to DiceLoss.
    """
    weights = None
    if class_counts is not None:
        import numpy as np
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.where(counts == 0, 1.0, counts)   # avoid div-by-zero for absent classes
        median = np.median(counts)
        weights = torch.tensor(median / counts, dtype=torch.float32)

    if cfg.loss_type == "ce":
        return CrossEntropyLoss()
    elif cfg.loss_type == "dice":
        return DiceLoss(smooth=cfg.dice_smooth, class_weights=weights)
    elif cfg.loss_type == "combined":
        return CombinedLoss(dice_weight=cfg.dice_weight, smooth=cfg.dice_smooth, class_weights=weights)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type!r}. Choose 'ce', 'dice', or 'combined'.")
