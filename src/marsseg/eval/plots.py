"""Segmentation overlays for paper figures (MS3). Pure numpy + cv2 — no plotting deps.

Colors come from ``data.ai4mars.CLASS_COLORS`` (RGB); ignore (255) renders black. Files are
written with cv2 (BGR), so channels are flipped at write time.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.ai4mars import CLASS_COLORS, IGNORE_INDEX


def colorize(mask: np.ndarray) -> np.ndarray:
    """(H, W) label map -> (H, W, 3) uint8 RGB; ignore/unknown values are black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        out[mask == cls] = rgb
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized mask onto a grayscale/RGB image; ignore pixels stay unblended."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image.astype(np.uint8)
    color = colorize(mask)
    blend = cv2.addWeighted(image, 1 - alpha, color, alpha, 0)
    keep = mask == IGNORE_INDEX
    blend[keep] = image[keep]
    return blend


def save_triptych(
    image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str | Path, alpha: float = 0.5
) -> Path:
    """Write `image | GT overlay | pred overlay` side by side (RGB -> BGR for cv2)."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    panel = np.concatenate(
        [image.astype(np.uint8), overlay(image, gt, alpha), overlay(image, pred, alpha)], axis=1
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel[:, :, ::-1])
    return out_path
