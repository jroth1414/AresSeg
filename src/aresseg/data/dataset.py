"""SegDataset + by-image splits for AI4Mars (Phase MS1).

Leakage guard: splits are BY IMAGE (a frame never appears in two splits). The ignore value (255)
is preserved through transforms and excluded from loss/metrics downstream (ignore_index). Images
are read grayscale and replicated to 3 channels.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .ai4mars import IGNORE_INDEX


class SegDataset:
    """torch-style Dataset. Each item: {'image': (3,H,W) float32, 'mask': (H,W) int64, 'name': str}.

    Does not subclass torch.utils.data.Dataset at import time so the module imports without torch;
    it is duck-typed (``__len__``/``__getitem__``) and works directly with a DataLoader.
    """

    def __init__(self, records: list[dict], transform, rover: str = "msl"):
        self.records = records
        self.transform = transform
        self.rover = rover

    def __len__(self) -> int:
        return len(self.records)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return np.repeat(img[:, :, None], 3, axis=2)  # (H, W, 3)

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype("int64")

    def __getitem__(self, i: int) -> dict:
        import torch

        rec = self.records[i]
        image = self._read_image(rec["image"])
        mask = self._read_mask(rec["label"])
        out = self.transform(image=image, mask=mask)
        m = out["mask"]
        m = m.long() if hasattr(m, "long") else torch.as_tensor(np.asarray(m), dtype=torch.long)
        return {
            "image": out["image"].float(),
            "mask": m,
            # canonical camera-qualified join key (DEVPLAN 4.3) — a missing name is a hard error,
            # never a positional fallback (cross-model joins would silently misalign)
            "name": rec["name"],
            "rover": self.rover,
        }


def make_splits(records: list[dict], val_frac: float = 0.2, seed: int = 1414) -> dict:
    """Shuffle records and split BY IMAGE into train/val. Returns {'train':[...], 'val':[...]}."""
    recs = list(records)
    random.Random(seed).shuffle(recs)
    n_val = int(round(len(recs) * val_frac))
    return {"train": recs[n_val:], "val": recs[:n_val]}


def class_pixel_counts(records: list[dict], num_classes: int = 4, max_images: int | None = None):
    """Per-class pixel counts over labels (for class-weighted loss). Ignores IGNORE_INDEX."""
    counts = np.zeros(num_classes, dtype="int64")
    for rec in records[: max_images or len(records)]:
        m = cv2.imread(rec["label"], cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        valid = m[m != IGNORE_INDEX]
        if valid.size:
            counts += np.bincount(valid.ravel(), minlength=num_classes)[:num_classes]
    return counts
