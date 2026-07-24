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

import hashlib
import hmac
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import load_env

SEGFORMER_SAFETENSORS = {
    "b0": {
        "revision": "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc",
        "sha256": "3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a",
    },
    "b2": {
        "revision": "d15ed1f9ae92346f6a6067dbb490a62494ae0d28",
        "sha256": "b3ad4dd552f9e1b871f46666f39187414133b861e3d07eda016600230f8a1ad6",
    },
}


# --------------------------------------------------------------------------------------
# baselines: constant majority class + tiny from-scratch U-Net
# --------------------------------------------------------------------------------------
class MajorityClassBaseline(nn.Module):
    """Parameter-free class-0 predictor used as the genuine non-deep baseline."""

    def __init__(self, num_classes: int = 4, predicted_class: int = 0):
        super().__init__()
        if not 0 <= predicted_class < num_classes:
            raise ValueError("predicted_class must be within the configured class range")
        self.num_classes = int(num_classes)
        self.predicted_class = int(predicted_class)

    def forward(self, x):
        logits = x.new_zeros((x.shape[0], self.num_classes, *x.shape[-2:]))
        logits[:, self.predicted_class] = 1.0
        return logits


# --------------------------------------------------------------------------------------
# tiny from-scratch U-Net
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
def _verified_segformer_snapshot(variant: str, revision: str | None) -> Path:
    """Download the pinned safe checkpoint and verify it before model construction."""
    pin = SEGFORMER_SAFETENSORS[variant]
    if revision != pin["revision"]:
        raise ValueError(
            f"pretrained MiT-{variant} requires safetensors revision {pin['revision']}; "
            f"got {revision!r}"
        )

    from huggingface_hub import hf_hub_download

    load_env()
    token = os.environ.get("HF_TOKEN")
    repo_id = f"nvidia/mit-{variant}"
    weights_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            token=token,
        )
    )
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=token,
        )
    )
    if weights_path.parent != config_path.parent:
        raise RuntimeError(f"MiT-{variant} config and weights resolved to different snapshots")
    with weights_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if not hmac.compare_digest(actual, pin["sha256"]):
        raise RuntimeError(
            f"MiT-{variant} safetensors sha256 mismatch: expected {pin['sha256']}, got {actual}"
        )
    return weights_path.parent


class _SegFormer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "b0",
        pretrained: bool = True,
        revision: str | None = None,
    ):
        super().__init__()
        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerModel,
        )

        if variant not in {"b0", "b2"}:
            raise ValueError(f"unsupported SegFormer variant {variant!r}")
        if pretrained:
            # Load only the ImageNet-1k MiT encoder. The segmentation decoder is freshly
            # initialized, keeping H2 comparable with the ImageNet-pretrained CNN encoders.
            snapshot = _verified_segformer_snapshot(variant, revision)
            encoder = SegformerModel.from_pretrained(
                snapshot,
                use_safetensors=True,
                local_files_only=True,
            )
            cfg = encoder.config
            cfg.num_labels = num_classes
            cfg.id2label = {index: str(index) for index in range(num_classes)}
            cfg.label2id = {str(index): index for index in range(num_classes)}
            self.model = SegformerForSemanticSegmentation(cfg)
            self.model.segformer = encoder
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
    name: str,
    num_classes: int = 4,
    backbone: str | None = None,
    pretrained: bool = True,
    sam_checkpoint: str | None = None,
    revision: str | None = None,
) -> nn.Module | None:
    """Build a segmentation model with ImageNet-pretrained encoders when requested.

    Foundation names (``dinov3_sat``/``sam``) are gated: they may skip-and-log and return ``None``
    when weights/GPU are absent (DEVPLAN 5.4); callers record ``status="skipped"``, never crash.
    ``sam_checkpoint`` (config key ``model.sam_checkpoint``, DEVPLAN 6) applies to ``sam`` only.
    """
    name = name.lower()
    if name == "majority":
        return MajorityClassBaseline(num_classes=num_classes)
    if name in {"tiny_unet", "baseline"}:
        return TinyUNet(num_classes=num_classes)
    if name in ("dinov3_sat", "sam"):
        from .foundation import build_foundation

        return build_foundation(
            name,
            num_classes=num_classes,
            sam_checkpoint=sam_checkpoint,
            revision=revision,
        )
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
        return _SegFormer(
            num_classes,
            variant=(backbone or "b0"),
            pretrained=pretrained,
            revision=revision,
        )
    raise ValueError(f"unknown model {name!r}")
