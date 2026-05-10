"""
Segmentation evaluation metrics.

Computes per-batch and cumulative:
    - Pixel Accuracy
    - Mean IoU (mIoU)  — used as the "mAP" proxy for segmentation
    - Per-class IoU
"""

from typing import List, Optional
import torch
import numpy as np


class SegmentationMetrics:
    """
    Accumulates confusion matrix over batches, then computes metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=8)
        for batch in loader:
            preds = model(images).argmax(1)
            metrics.update(preds, labels)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds:   (B, H, W) predicted class indices
            targets: (B, H, W) ground-truth class indices
        """
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clip to valid range to avoid index errors from bad labels
        preds   = np.clip(preds,   0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        np.add.at(self.confusion, (targets, preds), 1)

    def compute(self) -> dict:
        """Return dict with pixel_acc, miou, per_class_iou."""
        conf = self.confusion.astype(np.float64)

        # Pixel accuracy
        pixel_acc = np.diag(conf).sum() / (conf.sum() + 1e-10)

        # Per-class IoU
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp
        iou_per_class = tp / (tp + fp + fn + 1e-10)

        # Only average over classes that appear in ground truth
        present = conf.sum(axis=1) > 0
        miou = iou_per_class[present].mean() if present.any() else 0.0

        return {
            "pixel_acc":     float(pixel_acc),
            "miou":          float(miou),
            "iou_per_class": iou_per_class.tolist(),
        }
