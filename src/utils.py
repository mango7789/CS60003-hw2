"""
Utility functions: checkpoint I/O and prediction visualization.
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    miou: float,
    path,
):
    torch.save(
        {
            "epoch":      epoch,
            "miou":       miou,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, mIoU {ckpt['miou']:.4f})")
    return ckpt


# ── Colour palette for 8 Stanford Background classes ─────────────────────────

# sky, tree, road, grass, water, building, mountain, foreground
_PALETTE = np.array([
    [128, 128, 255],  # sky       — light blue
    [ 34, 139,  34],  # tree      — forest green
    [128, 128, 128],  # road      — grey
    [124, 252,   0],  # grass     — lawn green
    [  0, 191, 255],  # water     — deep sky blue
    [210, 105,  30],  # building  — chocolate
    [139,  90,  43],  # mountain  — saddle brown
    [255, 165,   0],  # foreground— orange
], dtype=np.uint8)


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Convert (H, W) integer label map to (H, W, 3) RGB image."""
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls_id, colour in enumerate(_PALETTE):
        rgb[label == cls_id] = colour
    return rgb


# ── Prediction visualisation ──────────────────────────────────────────────────

def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None,
    n_samples: int = 4,
) -> plt.Figure:
    """
    Plot a grid of (image | ground truth | prediction) for up to n_samples items.

    Returns the matplotlib Figure so it can be logged to TensorBoard via
    writer.add_figure().
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    n = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_to_rgb(labels_np[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(label_to_rgb(preds[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=_PALETTE[j] / 255.0, label=class_names[j])
        for j in range(len(class_names))
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=100)

    return fig
