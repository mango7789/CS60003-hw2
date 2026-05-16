"""
Stanford Background Dataset loader.

Dataset structure expected:
    data_root/
        images/   *.jpg
        labels/   *.regions.txt   (space-separated integer labels, one row per image row)

8 classes: sky(0), tree(1), road(2), grass(3), water(4), building(5), mountain(6), foreground(7)
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .config import Config


class StanfordBackgroundDataset(Dataset):
    """Loads images and pixel-level segmentation labels from the Stanford Background Dataset."""

    def __init__(
        self,
        data_root: str,
        file_list: List[str],
        input_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.file_list = file_list
        self.input_size = input_size  # (H, W)
        self.augment = augment

        self.img_dir = self.data_root / "images"
        self.lbl_dir = self.data_root / "labels"

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_label(self, stem: str) -> np.ndarray:
        """Read a .regions.txt label file and return an (H, W) int array."""
        label_path = self.lbl_dir / f"{stem}.regions.txt"
        rows = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([int(v) for v in line.split()])
        return np.array(rows, dtype=np.int64)

    def _augment(self, image: Image.Image, label: np.ndarray):
        """Apply identical spatial augmentations to image and label."""
        label_img = Image.fromarray(label.astype(np.uint8))

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label_img = TF.hflip(label_img)

        # Random crop (pad then crop to input_size)
        pad = 20
        image = TF.pad(image, pad)
        label_img = TF.pad(label_img, pad, fill=255)
        i, j, h, w = T.RandomCrop.get_params(
            image, output_size=self.input_size
        )
        image = TF.crop(image, i, j, h, w)
        label_img = TF.crop(label_img, i, j, h, w)

        label = np.array(label_img, dtype=np.int64)
        return image, label

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        stem = self.file_list[idx]

        image = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        label = self._load_label(stem)

        # Resize both image and label to input_size first
        image = image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        label_img = Image.fromarray(label.astype(np.uint8))
        label_img = label_img.resize(
            (self.input_size[1], self.input_size[0]), Image.NEAREST
        )
        label = np.array(label_img, dtype=np.int64)

        if self.augment:
            image, label = self._augment(image, label)

        image_tensor = self.img_transform(image)
        label_tensor = torch.from_numpy(label).long()

        return image_tensor, label_tensor


# ── Factory ───────────────────────────────────────────────────────────────────

def _collect_stems(data_root: str) -> List[str]:
    """Return sorted list of file stems that have both image and label."""
    img_dir = Path(data_root) / "images"
    lbl_dir = Path(data_root) / "labels"
    stems = sorted(
        p.stem for p in img_dir.glob("*.jpg")
        if (lbl_dir / f"{p.stem}.regions.txt").exists()
    )
    return stems


def _stratified_split(
    stems: List[str], data_root: str, val_split: float, seed: int, num_classes: int = 8
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Split stems so that images containing rare classes are distributed
    proportionally between train and val.

    Returns (train_stems, val_stems, train_class_counts) where
    train_class_counts[c] is the total pixel count for class c in the
    training split — used to compute inverse-frequency weights for Dice loss.
    """
    from collections import defaultdict

    lbl_dir = Path(data_root) / "labels"
    rare = {4, 6, 7}

    groups: dict = defaultdict(list)
    pixel_counts: dict = {}  # stem -> (H*W array of class counts)

    for stem in stems:
        label_path = lbl_dir / f"{stem}.regions.txt"
        rows = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([int(v) for v in line.split()])
        arr = np.array(rows, dtype=np.int64)
        valid = arr.flatten()
        valid = valid[(valid >= 0) & (valid < num_classes)]
        counts = np.bincount(valid, minlength=num_classes)[:num_classes]
        pixel_counts[stem] = counts

        classes = set(np.unique(arr).tolist())
        rare_present = sorted(classes & rare)
        stratum = rare_present[0] if rare_present else -1
        groups[stratum].append(stem)

    rng = random.Random(seed)
    train_stems: List[str] = []
    val_stems: List[str] = []
    for group in groups.values():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_split))
        val_stems.extend(group[:n_val])
        train_stems.extend(group[n_val:])

    train_class_counts = sum(pixel_counts[s] for s in train_stems)
    return train_stems, val_stems, train_class_counts


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a Config."""
    stems = _collect_stems(cfg.data_root)
    if not stems:
        raise FileNotFoundError(
            f"No matching image/label pairs found in {cfg.data_root}. "
            "Make sure images/ and labels/ sub-directories exist."
        )

    train_stems, val_stems, train_class_counts = _stratified_split(
        stems, cfg.data_root, cfg.val_split, cfg.random_seed, cfg.num_classes
    )

    train_ds = StanfordBackgroundDataset(
        cfg.data_root, train_stems,
        input_size=cfg.input_size, augment=True,
    )
    val_ds = StanfordBackgroundDataset(
        cfg.data_root, val_stems,
        input_size=cfg.input_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader, train_class_counts
