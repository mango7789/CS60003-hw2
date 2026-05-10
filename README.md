# Task 3 — U-Net Semantic Segmentation on Stanford Background Dataset

Pixel-level semantic segmentation using a U-Net built from scratch (no pretrained weights).
Compares three loss functions — Cross-Entropy, Dice, and Combined — on the Stanford Background Dataset.

---

## Project Structure

```
hw2/
├── src/
│   ├── __init__.py
│   ├── config.py       # All hyperparameters in one place
│   ├── dataset.py      # Stanford Background Dataset loader + augmentation
│   ├── model.py        # U-Net (DoubleConv, EncoderBlock, DecoderBlock)
│   ├── losses.py       # CrossEntropyLoss, DiceLoss (manual), CombinedLoss
│   ├── metrics.py      # SegmentationMetrics (mIoU, pixel accuracy)
│   ├── trainer.py      # Training loop + TensorBoard logging
│   └── utils.py        # Checkpoint I/O, prediction visualisation
├── notebooks/
│   └── task3_experiment.ipynb   # Main notebook — runs all 3 experiments
├── report/
│   └── report.tex
├── data/
│   └── stanford_background/     # ← place dataset here (see below)
│       ├── images/  *.jpg
│       └── labels/  *.regions.txt
├── runs/                        # TensorBoard logs (auto-created)
├── checkpoints/                 # Model checkpoints (auto-created)
└── README.md
```

---

## Environment Setup

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision
- tensorboard
- numpy, Pillow, matplotlib, pandas

### Install

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard numpy Pillow matplotlib pandas jupyter
```

---

## Dataset

Download the **Stanford Background Dataset** from the official source:

> http://dags.stanford.edu/projects/scenedataset.html

Extract and place the files so the directory looks like:

```
data/stanford_background/
├── images/
│   ├── 0000000.jpg
│   ├── 0000001.jpg
│   └── ...
└── labels/
    ├── 0000000.regions.txt
    ├── 0000001.regions.txt
    └── ...
```

Each `.regions.txt` file contains space-separated integer class labels (0–7), one row per image row.

---

## Training

### Option A — Jupyter Notebook (recommended)

```bash
cd hw2
jupyter notebook notebooks/task3_experiment.ipynb
```

Run cells sequentially. The notebook runs all three experiments and launches TensorBoard inline.

### Option B — Python script

```python
import sys
sys.path.insert(0, ".")          # run from hw2/

from src.config  import Config
from src.dataset import get_dataloaders
from src.model   import UNet
from src.losses  import get_loss_fn
from src.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config(
    loss_type     = "combined",   # "ce" | "dice" | "combined"
    num_epochs    = 50,
    batch_size    = 8,
    learning_rate = 1e-3,
)

train_loader, val_loader = get_dataloaders(cfg)
model   = UNet(num_classes=cfg.num_classes, base_channels=cfg.base_channels)
loss_fn = get_loss_fn(cfg)

trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg, device)
trainer.train()
```

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

Logged metrics:
| Tag | Description |
|-----|-------------|
| `Loss/train` | Training loss per epoch |
| `Loss/val` | Validation loss per epoch |
| `Accuracy/val` | Validation pixel accuracy |
| `mIoU/val` | Validation mean IoU |
| `IoU_per_class/{name}` | Per-class IoU on validation set |
| `LR` | Current learning rate |

---

## Changing Hyperparameters

All hyperparameters live in `src/config.py` (or can be passed as keyword arguments to `Config`).
Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | `"combined"` | `"ce"` / `"dice"` / `"combined"` |
| `num_epochs` | `50` | Number of training epochs |
| `batch_size` | `8` | Batch size |
| `learning_rate` | `1e-3` | Initial learning rate |
| `base_channels` | `64` | U-Net base channel width |
| `optimizer` | `"adam"` | `"adam"` / `"adamw"` / `"sgd"` |
| `scheduler` | `"cosine"` | `"cosine"` / `"step"` / `"none"` |
| `dice_weight` | `0.5` | Dice term weight in combined loss |
| `val_split` | `0.2` | Fraction of data used for validation |

---

## Evaluation

The best checkpoint for each experiment is saved to `checkpoints/{experiment_name}/best.pth`.

To evaluate a saved checkpoint:

```python
from src.config  import Config
from src.dataset import get_dataloaders
from src.model   import UNet
from src.utils   import load_checkpoint
from src.metrics import SegmentationMetrics
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config(loss_type="combined")
_, val_loader = get_dataloaders(cfg)

model = UNet(num_classes=cfg.num_classes).to(device)
load_checkpoint(model, f"checkpoints/{cfg.experiment_name}/best.pth", device=device)
model.eval()

metrics = SegmentationMetrics(cfg.num_classes)
with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images.to(device))
        metrics.update(logits.argmax(1), labels)

results = metrics.compute()
print(f"mIoU:       {results['miou']:.4f}")
print(f"Pixel Acc:  {results['pixel_acc']:.4f}")
```

---

## Model Weights

Pre-trained weights are available at:

> [Google Drive link — fill in after training]

---

## Citation

```bibtex
@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  year      = {2015}
}

@inproceedings{gould2009decomposing,
  title     = {Decomposing a Scene into Geometric and Semantically Consistent Regions},
  author    = {Gould, Stephen and Fulton, Richard and Koller, Daphne},
  booktitle = {ICCV},
  year      = {2009}
}
```
