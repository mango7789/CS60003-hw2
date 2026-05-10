"""
Training loop with TensorBoard logging.

Logs per-epoch:
    - train/loss, val/loss
    - val/pixel_acc
    - val/miou  (reported as mAP in the report)
    - val/iou_class_{name}
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .metrics import SegmentationMetrics
from .utils import save_checkpoint


class Trainer:
    """
    Encapsulates the full train / validate loop.

    Args:
        model:        U-Net (or any nn.Module with matching output shape)
        loss_fn:      loss function (CrossEntropyLoss / DiceLoss / CombinedLoss)
        train_loader: DataLoader for training set
        val_loader:   DataLoader for validation set
        cfg:          Config object
        device:       torch device
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.loss_fn      = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        log_path = Path(cfg.log_dir) / cfg.experiment_name
        self.writer = SummaryWriter(log_dir=str(log_path))
        print(f"TensorBoard logs → {log_path}")
        print(f"  Run:  tensorboard --logdir {cfg.log_dir}")

        self.metrics = SegmentationMetrics(cfg.num_classes)
        self.best_miou = 0.0

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=cfg.learning_rate,
                momentum=0.9, weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.num_epochs
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        else:
            return None

    # ── One epoch ─────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"  train", leave=False, unit="batch")
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"  val  ", leave=False, unit="batch")
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            self.metrics.update(preds, labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        results = self.metrics.compute()
        results["loss"] = total_loss / len(self.val_loader)
        return results

    # ── TensorBoard logging ───────────────────────────────────────────────────

    def _log(self, train_loss: float, val_results: dict, epoch: int):
        w = self.writer
        w.add_scalar("Loss/train",      train_loss,              epoch)
        w.add_scalar("Loss/val",        val_results["loss"],     epoch)
        w.add_scalar("Accuracy/val",    val_results["pixel_acc"], epoch)
        w.add_scalar("mIoU/val",        val_results["miou"],     epoch)

        for i, (iou, name) in enumerate(
            zip(val_results["iou_per_class"], self.cfg.CLASS_NAMES)
        ):
            w.add_scalar(f"IoU_per_class/{name}", iou, epoch)

        w.add_scalar(
            "LR", self.optimizer.param_groups[0]["lr"], epoch
        )

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        ckpt_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Experiment : {cfg.experiment_name}")
        print(f"Loss       : {cfg.loss_type}")
        print(f"Epochs     : {cfg.num_epochs}  |  LR: {cfg.learning_rate}  |  BS: {cfg.batch_size}")
        print(f"{'='*60}\n")

        epoch_bar = tqdm(range(1, cfg.num_epochs + 1), desc=cfg.experiment_name, unit="epoch")
        for epoch in epoch_bar:
            t0 = time.time()

            train_loss  = self._train_epoch(epoch)
            val_results = self._val_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self._log(train_loss, val_results, epoch)

            elapsed = time.time() - t0
            epoch_bar.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                val_loss=f"{val_results['loss']:.4f}",
                acc=f"{val_results['pixel_acc']:.4f}",
                mIoU=f"{val_results['miou']:.4f}",
                t=f"{elapsed:.1f}s",
            )

            # Save best model
            if val_results["miou"] > self.best_miou:
                self.best_miou = val_results["miou"]
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_results["miou"], ckpt_dir / "best.pth"
                )

            # Periodic checkpoint
            if epoch % cfg.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_results["miou"], ckpt_dir / f"epoch_{epoch:03d}.pth"
                )

        self.writer.close()
        print(f"\nTraining complete. Best mIoU: {self.best_miou:.4f}")
        return self.best_miou
