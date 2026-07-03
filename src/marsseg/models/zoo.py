"""Segmentation model zoo + registry (MS2).

``build_model(name, ...)`` returns an ``nn.Module`` mapping images ``(B,3,H,W)`` to per-pixel logits
``(B,C,H,W)`` at the INPUT resolution. Trainable models:
  - ``baseline``        — a small from-scratch U-Net (the H0/H1 yardstick).
  - ``unet``            — smp U-Net (ResNet-34 / EfficientNet-B0 encoder; ImageNet vs scratch, H2).
  - ``deeplabv3plus``   — smp DeepLabV3+ (ResNet-50 encoder).
  - ``segformer``       — transformer (MiT-B0 / B2), logits upsampled to input size.
Foundation references (DINOv3-SAT / SAM) live in ``foundation.py`` (gated).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# baseline: tiny from-scratch U-Net
# --------------------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """A small 2-level U-Net trained from scratch (the baseline)."""

    def __init__(self, num_classes: int = 4, base: int = 16):
        super().__init__()
        self.enc1 = _DoubleConv(3, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# --------------------------------------------------------------------------------------
# SegFormer wrapper (transformers) — upsample logits to input resolution
# --------------------------------------------------------------------------------------
class _SegFormer(nn.Module):
    def __init__(self, num_classes: int, variant: str = "b0", pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            depths = {"b0": [2, 2, 2, 2], "b2": [3, 4, 6, 3]}[variant]
            widths = {"b0": [32, 64, 160, 256], "b2": [64, 128, 320, 512]}[variant]
            cfg = SegformerConfig(
                num_labels=num_classes,
                depths=depths,
                hidden_sizes=widths,
                decoder_hidden_size={"b0": 256, "b2": 768}[variant],
            )
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # (B, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------
SMP_BUILDERS = {"unet": "Unet", "deeplabv3plus": "DeepLabV3Plus"}
DEFAULT_BACKBONE = {"unet": "resnet34", "deeplabv3plus": "resnet50"}


def build_model(
    name: str, num_classes: int = 4, backbone: str | None = None, pretrained: bool = True
) -> nn.Module | None:
    """Build a segmentation model. ``pretrained`` = ImageNet (smp) / ADE (SegFormer) init (H2).

    Foundation names (``dinov3_sat``/``sam``) are gated: they may skip-and-log and return ``None``
    when weights/GPU are absent (DEVPLAN 5.4); callers record ``status="skipped"``, never crash.
    """
    name = name.lower()
    if name == "baseline":
        return TinyUNet(num_classes=num_classes)
    if name in ("dinov3_sat", "sam"):
        from .foundation import build_foundation

        return build_foundation(name, num_classes=num_classes)
    if name in SMP_BUILDERS:
        import segmentation_models_pytorch as smp

        ctor = getattr(smp, SMP_BUILDERS[name])
        return ctor(
            encoder_name=backbone or DEFAULT_BACKBONE[name],
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    if name == "segformer":
        return _SegFormer(num_classes, variant=(backbone or "b0"), pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}")
