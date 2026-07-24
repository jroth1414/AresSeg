"""Capability detection: discover the hardware/library profile.

Segmentation training is GPU-bound. CPU is fine for the full pipeline + smoke tests; full training
runs on the V100. SAM / DINOv3 foundation models (H5) are gated behind these flags and degrade
gracefully (skip-and-log) when the package or hardware is absent.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass


def _can_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class Capabilities:
    cuda: bool
    gpu_name: str | None
    smp: bool  # segmentation-models-pytorch (U-Net/DeepLab)
    transformers: bool  # SegFormer
    sam: bool  # Meta Segment-Anything (foundation, H5)
    timm: bool  # DINOv3 / ViT backbones (foundation, H5)
    profile: str  # "windows_cpu" | "gpu_full"


def detect() -> Capabilities:
    cuda = has_cuda()
    caps = dict(
        cuda=cuda,
        gpu_name=_gpu_name(),
        smp=_can_import("segmentation_models_pytorch"),
        transformers=_can_import("transformers"),
        sam=_can_import("segment_anything"),
        timm=_can_import("timm"),
    )
    profile = "gpu_full" if cuda else "windows_cpu"
    return Capabilities(profile=profile, **caps)


def as_dict() -> dict:
    return asdict(detect())
