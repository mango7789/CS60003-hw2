"""
Central configuration for Task 3: U-Net semantic segmentation on Stanford Background Dataset.
Edit this file to change hyperparameters between experiments.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: str = "data/stanford_background"
    num_classes: int = 8
    image_size: Tuple[int, int] = (320, 240)   # (W, H) — original resolution
    input_size: Tuple[int, int] = (256, 256)   # resize to square for U-Net
    val_split: float = 0.2
    random_seed: int = 42

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4

    # ── Loss function ─────────────────────────────────────────────────────────
    # Options: "ce"  |  "dice"  |  "combined"
    loss_type: Literal["ce", "dice", "combined"] = "combined"
    dice_weight: float = 0.3   # weight of Dice term in combined loss (CE weight = 1 - dice_weight)
    dice_smooth: float = 1e-6  # smoothing constant in Dice Loss

    # ── Model ─────────────────────────────────────────────────────────────────
    base_channels: int = 64   # channels in first encoder block; doubles each level

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    # Options: "adam" | "sgd" | "adamw"
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    step_size: int = 15        # for StepLR
    gamma: float = 0.5         # for StepLR

    # ── Logging / Checkpointing ───────────────────────────────────────────────
    log_dir: str = "runs"      # TensorBoard log root; sub-dir named by experiment
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10       # save checkpoint every N epochs
    experiment_name: str = ""  # auto-generated from loss_type if empty

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"unet_{self.loss_type}_lr{self.learning_rate}_bs{self.batch_size}"

    # Convenience: class names for Stanford Background Dataset
    CLASS_NAMES = [
        "sky", "tree", "road", "grass",
        "water", "building", "mountain", "foreground",
    ]
