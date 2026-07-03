"""Segmentation metrics with exact ignore handling (MS3; DEVPLAN section 6).

Two layers:
- PER-IMAGE integer counts (``per_image_counts``): per-class (inter, union), correct, n_valid.
  These are what ``per_image.parquet`` stores (section 7.4) so the paired bootstrap can recompute
  split-level metrics from SUMMED counts on every resample.
- SPLIT-LEVEL values from summed counts: per-class IoU = micro over images then per class
  (``iou_from_counts``); mIoU = MACRO over the fixed class set S = {classes with split-level
  union > 0} (``fixed_class_set`` / ``macro_miou_from_counts``); pixel_acc over valid pixels.

``ignore_index=255`` pixels are excluded from every numerator and denominator. ``boundary_f1``
(3 px tolerance) is DESCRIPTIVE ONLY — no summable decomposition, never bootstrapped or tested.
"""

from __future__ import annotations

import numpy as np

from ..data.ai4mars import IGNORE_INDEX, NUM_CLASSES


def per_image_counts(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    """Integer counts for one image: {'inter': (C,), 'union': (C,), 'correct': int, 'n_valid': int}.

    ``pred``/``gt`` are (H, W) integer label maps; ``gt`` may contain IGNORE_INDEX (excluded
    everywhere — an ignore pixel contributes to no class's inter/union and not to n_valid).
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    valid = gt != IGNORE_INDEX
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter[c] = np.count_nonzero(p & g)
        union[c] = np.count_nonzero(p | g)
    return {
        "inter": inter,
        "union": union,
        "correct": int(np.count_nonzero((pred == gt) & valid)),
        "n_valid": int(np.count_nonzero(valid)),
    }


def fixed_class_set(union_sums: np.ndarray) -> list[int]:
    """S = classes with split-level union > 0 (DEVPLAN 5.6 step 1 / section 6). Fixed once."""
    return [c for c in range(len(union_sums)) if union_sums[c] > 0]


def iou_from_counts(inter_sum: float, union_sum: float) -> float:
    """Per-class split-level IoU from summed counts; NaN when the class has no union at all."""
    return float(inter_sum) / float(union_sum) if union_sum > 0 else float("nan")


def macro_miou_from_counts(
    inter_sums: np.ndarray, union_sums: np.ndarray, class_set: list[int]
) -> float:
    """MACRO mean over the FIXED set: a fixed-set class with union==0 contributes IoU = 0
    (``empty_class_in_resample: iou_zero``); the denominator stays len(class_set)."""
    if not class_set:
        return float("nan")
    ious = [
        (float(inter_sums[c]) / float(union_sums[c]) if union_sums[c] > 0 else 0.0)
        for c in class_set
    ]
    return float(np.mean(ious))


def pixel_acc_from_counts(correct_sum: float, n_valid_sum: float) -> float:
    return float(correct_sum) / float(n_valid_sum) if n_valid_sum > 0 else float("nan")


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary of a binary mask (mask minus its 4-connected erosion)."""
    from scipy.ndimage import binary_erosion

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = NUM_CLASSES,
    tol_px: int = 3,
) -> float:
    """Mean boundary-F1 over classes present in pred or GT (3 px Euclidean tolerance).

    DESCRIPTIVE ONLY (DEVPLAN section 6): boundaries are computed on each class's binary mask
    with ignore pixels excluded, so ignore contributes to neither precision nor recall. Returns
    NaN when no class is present at all.
    """
    from scipy.ndimage import distance_transform_edt

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    valid = gt != IGNORE_INDEX
    f1s = []
    for c in range(num_classes):
        pm = (pred == c) & valid
        gm = (gt == c) & valid
        if not pm.any() and not gm.any():
            continue  # class absent in both -> contributes nothing for this image
        pb = _boundary(pm)
        gb = _boundary(gm)
        if not pb.any() and not gb.any():
            f1s.append(1.0)  # both masks present but boundary-free (fill the frame) => perfect
            continue
        if not pb.any() or not gb.any():
            f1s.append(0.0)
            continue
        dist_to_gt = distance_transform_edt(~gb)
        dist_to_pred = distance_transform_edt(~pb)
        precision = float(np.mean(dist_to_gt[pb] <= tol_px))
        recall = float(np.mean(dist_to_pred[gb] <= tol_px))
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")
