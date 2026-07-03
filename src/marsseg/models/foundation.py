"""Foundation-model references for H5 (gated): DINOv3-SAT frozen backbone + head; SAM zero-shot.

Gating contract (DEVPLAN section 3 / 5.4): these arms NEVER hard-crash the pipeline. Gates are
checked FIRST via ``os.environ`` / imports / files — if any gate fails, ``build_foundation``
logs the reason and returns ``None`` ("skip-and-log"); ``run_experiment.py`` (MS3) converts a
``None`` into ``status="skipped"`` results rows + ``manifest.gpu_stages_skipped``. The gated
``HF_TOKEN`` is read through ``config.require_secret`` only on the actual load path.

- ``dinov3_sat`` — DINOv3 ViT-L/16 pretrained on Earth satellite imagery (SAT-493M; gated HF
  weights, fits the 16 GB V100 unlike ViT-7B). The backbone is FROZEN; only a small conv head is
  trained (H5 "finetuned" variant).
- ``sam`` — Meta SAM ViT-B automatic mask generation (zero-shot). SAM proposes class-AGNOSTIC
  regions; AI4Mars has no prompt/text channel, so the H5 eval (MS4) scores the proposals with the
  region-oracle assignment (each region takes the majority ground-truth class among its valid
  pixels) — explicitly an UPPER BOUND on any zero-shot region labeler using SAM's masks. That
  protocol choice must be frozen in experiments/PREREG.md (MS3) before any test number is computed;
  this module only builds the generator.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.capabilities import has_cuda
from ..utils.config import REPO_ROOT, load_env, require_secret
from ..utils.logging import get_logger

log = get_logger("marsseg.foundation")

DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_DEFAULT = REPO_ROOT / "data" / "weights" / "sam" / "sam_vit_b_01ec64.pth"
FOUNDATION_MODELS = ("dinov3_sat", "sam")


def _sam_importable() -> bool:
    return importlib.util.find_spec("segment_anything") is not None


def gating_reasons(name: str, sam_checkpoint: str | os.PathLike | None = None) -> list[str]:
    """Why ``name`` cannot run here (empty list = all gates pass). Never raises."""
    load_env()  # populate HF_TOKEN from .env if present (does not override real env)
    reasons = []
    if not has_cuda():
        reasons.append("cuda absent (H5 is decided on the gpu_full profile)")
    if name == "dinov3_sat":
        if not os.environ.get("HF_TOKEN"):
            reasons.append("HF_TOKEN missing/empty (gated DINOv3 weights need a HF read token)")
    elif name == "sam":
        if not _sam_importable():
            reasons.append("segment-anything not installed (requirements-extras.txt)")
        ckpt = Path(sam_checkpoint or SAM_CHECKPOINT_DEFAULT)
        if not ckpt.is_file():
            reasons.append(f"SAM checkpoint absent: {ckpt}")
    else:
        raise ValueError(f"unknown foundation model {name!r}; expected one of {FOUNDATION_MODELS}")
    return reasons


class DinoV3SatSegmenter(nn.Module):
    """Frozen ViT backbone + trainable conv head; logits ``(B,C,H,W)`` at input resolution.

    The backbone stays in eval mode permanently (frozen stats); only ``self.head`` trains.
    ``num_prefix_tokens`` = CLS + register tokens to drop before the patch-grid reshape.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_prefix_tokens: int,
        num_classes: int = 4,
        head_dim: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.num_prefix_tokens = num_prefix_tokens
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, head_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, head_dim),
            nn.GELU(),
            nn.Conv2d(head_dim, num_classes, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone never re-enters train mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state  # (B, prefix+N, C)
        feats = tokens[:, self.num_prefix_tokens :, :]
        if feats.shape[1] != gh * gw:
            raise RuntimeError(
                f"token grid mismatch: got {feats.shape[1]} patch tokens for a {gh}x{gw} grid "
                f"(num_prefix_tokens={self.num_prefix_tokens} wrong for this backbone?)"
            )
        feats = feats.transpose(1, 2).reshape(b, -1, gh, gw)
        return F.interpolate(self.head(feats), size=(h, w), mode="bilinear", align_corners=False)


def _load_dinov3(num_classes: int) -> DinoV3SatSegmenter:
    from transformers import AutoModel

    token = require_secret("HF_TOKEN")  # load path only; gating already verified presence
    backbone = AutoModel.from_pretrained(DINOV3_MODEL_ID, token=token)
    cfg = backbone.config
    return DinoV3SatSegmenter(
        backbone,
        hidden_size=cfg.hidden_size,
        patch_size=cfg.patch_size,
        num_prefix_tokens=1 + getattr(cfg, "num_register_tokens", 0),
        num_classes=num_classes,
    )


class SamZeroShotSegmenter(nn.Module):
    """SAM ViT-B automatic mask generator (zero-shot H5 reference; eval-only, never trained).

    Not a logits-producing segmenter: call ``generate(image_hwc_uint8)`` for SAM's region
    proposals; the H5 eval step assigns classes (region-oracle, see module docstring).
    """

    def __init__(self, checkpoint: str | os.PathLike, model_type: str = SAM_MODEL_TYPE):
        super().__init__()
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.generator = SamAutomaticMaskGenerator(self.sam)

    @torch.no_grad()
    def generate(self, image_hwc_uint8) -> list[dict]:
        return self.generator.generate(image_hwc_uint8)

    def forward(self, *_args, **_kw):
        raise NotImplementedError("SAM is zero-shot eval-only: use .generate(image), not forward()")


def build_foundation(
    name: str, num_classes: int = 4, sam_checkpoint: str | os.PathLike | None = None
) -> nn.Module | None:
    """Build a gated foundation model, or skip-and-log: return ``None`` with the reason logged.

    Load-path failures are ALSO skip-and-logged (pending HF gate approval / token without
    gated-repo scope / network error / corrupt checkpoint): the never-hard-crash contract covers
    the whole build, and the logged reason lands in the skipped results row + manifest (MS3).
    """
    name = name.lower()
    reasons = gating_reasons(name, sam_checkpoint=sam_checkpoint)  # raises on unknown name
    if reasons:
        log.warning("foundation arm %r skipped: %s", name, "; ".join(reasons))
        return None
    try:
        if name == "dinov3_sat":
            return _load_dinov3(num_classes)
        model = SamZeroShotSegmenter(sam_checkpoint or SAM_CHECKPOINT_DEFAULT)
        if torch.cuda.is_available():
            # SAM never passes through the Lightning Trainer, so nothing else moves it off CPU
            model = model.to("cuda")
        return model
    except Exception as e:  # contract: gated arms never hard-crash the pipeline
        log.warning("foundation arm %r skipped: load failed: %r", name, e)
        return None
