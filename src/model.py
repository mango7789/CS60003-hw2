"""
U-Net built from scratch — no pretrained weights, no torchvision model imports.

Architecture:
    Encoder: 4 × (DoubleConv → MaxPool)
    Bottleneck: DoubleConv
    Decoder: 4 × (Upsample → concat skip → DoubleConv)
    Head: 1×1 Conv → num_classes logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool; returns both the skip feature and the pooled output."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """Bilinear upsample → concat skip connection → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions from encoder
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net with configurable base channel width.

    Args:
        num_classes:   number of segmentation classes
        in_channels:   input image channels (3 for RGB)
        base_channels: channels in the first encoder block (doubles each level)
    """

    def __init__(self, num_classes: int, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        # Encoder
        self.enc1 = EncoderBlock(in_channels, c)       # skip: c,   out: c
        self.enc2 = EncoderBlock(c,           c * 2)   # skip: 2c,  out: 2c
        self.enc3 = EncoderBlock(c * 2,       c * 4)   # skip: 4c,  out: 4c
        self.enc4 = EncoderBlock(c * 4,       c * 8)   # skip: 8c,  out: 8c

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)    # out: 16c

        # Decoder
        self.dec4 = DecoderBlock(c * 16, c * 8,  c * 8)
        self.dec3 = DecoderBlock(c * 8,  c * 4,  c * 4)
        self.dec2 = DecoderBlock(c * 4,  c * 2,  c * 2)
        self.dec1 = DecoderBlock(c * 2,  c,      c)

        # Segmentation head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)   # (B, num_classes, H, W) — raw logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
