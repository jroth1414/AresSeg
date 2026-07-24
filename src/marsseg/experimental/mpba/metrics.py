"""Validation diagnostics for the perspective-first MPBA experiment.

These metrics are deliberately separate from the canonical Protocol V3 result machinery.  They
operate on validation predictions and return JSON-friendly summaries for ``experiments/mpba``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

from ...data.ai4mars import CLASSES, NUM_CLASSES
from ...eval.metrics import (
    boundary_f1,
    fixed_class_set,
    iou_from_counts,
    macro_miou_from_counts,
    per_image_counts,
    pixel_acc_from_counts,
)

BIG_ROCK_CLASS = 3
SMALL_COMPONENT_MIN_PIXELS = 16
SMALL_COMPONENT_MAX_PIXELS = 256
SMALL_COMPONENT_HIT_FRACTION = 0.5
ROUTING_ACTIVE_THRESHOLD = 0.10


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_or_none(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def small_big_rock_component_counts(
    pred: Any,
    gt: Any,
    *,
    class_index: int = BIG_ROCK_CLASS,
    min_pixels: int = SMALL_COMPONENT_MIN_PIXELS,
    max_pixels: int = SMALL_COMPONENT_MAX_PIXELS,
    hit_fraction: float = SMALL_COMPONENT_HIT_FRACTION,
) -> dict[str, int]:
    """Count recalled eligible Big Rock components in one image.

    Ground-truth components use 8-connectivity.  Areas are inclusive in ``[min_pixels,
    max_pixels]`` and a component is recalled when at least ``hit_fraction`` of its ground-truth
    pixels are predicted as Big Rock.  Ignore pixels never belong to a ground-truth component.
    """
    prediction = _as_numpy(pred)
    target = _as_numpy(gt)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError(
            f"pred and gt must share shape (H,W), got {prediction.shape} and {target.shape}"
        )
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("component limits must satisfy 0 < min_pixels <= max_pixels")
    if not 0.0 <= float(hit_fraction) <= 1.0:
        raise ValueError("hit_fraction must lie in [0, 1]")

    components, count = connected_components(
        target == int(class_index), structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return {"eligible": 0, "recalled": 0}
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    eligible_ids = np.flatnonzero((sizes >= min_pixels) & (sizes <= max_pixels))
    eligible_ids = eligible_ids[eligible_ids != 0]
    if eligible_ids.size == 0:
        return {"eligible": 0, "recalled": 0}

    predicted_class = prediction == int(class_index)
    hits = np.bincount(components.ravel(), weights=predicted_class.ravel(), minlength=count + 1)
    recalled = np.count_nonzero(hits[eligible_ids] / sizes[eligible_ids] >= hit_fraction)
    return {"eligible": int(eligible_ids.size), "recalled": int(recalled)}


def small_big_rock_component_recall(pred: Any, gt: Any, **kwargs: Any) -> float:
    """Return small-component recall for one image, or NaN when it has no eligible component."""
    counts = small_big_rock_component_counts(pred, gt, **kwargs)
    if counts["eligible"] == 0:
        return float("nan")
    return float(counts["recalled"] / counts["eligible"])


def _routing_axis(weights: np.ndarray, scale_axis: int | None) -> int:
    if scale_axis is not None:
        axis = int(scale_axis) % weights.ndim
    elif weights.ndim == 4:  # (B, S, H, W)
        axis = 1
    elif weights.ndim == 3:  # (S, H, W)
        axis = 0
    elif weights.ndim == 2:  # (B, S), e.g. spatially static weights
        axis = 1
    elif weights.ndim == 1:  # (S,)
        axis = 0
    else:
        raise ValueError(f"routing weights need 1-4 dimensions, got shape {weights.shape}")
    return axis


def _routing_totals(weights: Any, *, scale_axis: int | None = None) -> tuple[np.ndarray, int]:
    value = _as_numpy(weights).astype(np.float64, copy=False)
    if value.size == 0:
        raise ValueError("routing weights cannot be empty")
    axis = _routing_axis(value, scale_axis)
    if not np.isfinite(value).all():
        raise ValueError("routing weights must be finite")
    if np.any(value < -1e-6):
        raise ValueError("routing weights must be non-negative")
    probability_sums = value.sum(axis=axis)
    if not np.allclose(probability_sums, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("routing weights must sum to one along the scale axis")
    reduce_axes = tuple(index for index in range(value.ndim) if index != axis)
    totals = value.sum(axis=reduce_axes) if reduce_axes else value.copy()
    observations = int(value.size // value.shape[axis])
    return np.asarray(totals, dtype=np.float64), observations


def _routing_summary_from_totals(
    totals: np.ndarray,
    observations: int,
    *,
    active_threshold: float = ROUTING_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    if observations <= 0:
        raise ValueError("routing observation count must be positive")
    if not 0.0 <= float(active_threshold) <= 1.0:
        raise ValueError("active_threshold must lie in [0, 1]")
    means = np.asarray(totals, dtype=np.float64) / observations
    active = np.flatnonzero(means > float(active_threshold)).astype(int).tolist()
    return {
        "mean_weights": [float(value) for value in means],
        "active_scale_indices": active,
        "n_active_scales": len(active),
        "active_threshold": float(active_threshold),
        "n_observations": int(observations),
    }


def routing_utilization(
    weights: Any,
    *,
    scale_axis: int | None = None,
    active_threshold: float = ROUTING_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize mean softmax weight per scale and scales exceeding the promotion threshold."""
    totals, observations = _routing_totals(weights, scale_axis=scale_axis)
    return _routing_summary_from_totals(totals, observations, active_threshold=active_threshold)


def cutoff_mae(predicted: Any, target: Any) -> float:
    """Mean absolute cutoff error in fractions of image height."""
    pred = _as_numpy(predicted).astype(np.float64, copy=False).reshape(-1)
    truth = _as_numpy(target).astype(np.float64, copy=False).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predicted and target cutoff shapes differ: {pred.shape} vs {truth.shape}"
        )
    if pred.size == 0:
        return float("nan")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("cutoff values must be finite")
    return float(np.mean(np.abs(pred - truth)))


class MPBAValidationAccumulator:
    """Accumulate all MPBA screening diagnostics over validation batches.

    ``pred`` and ``gt`` passed to :meth:`update` may be one ``(H,W)`` image or aligned
    ``(B,H,W)`` batches.  Routing weights use ``(B,S,H,W)``.  Cutoffs are optional for native,
    static, content-only, and raw-row arms; when supplied both prediction and target are required.
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        class_names: list[str] | tuple[str, ...] = tuple(CLASSES),
        boundary_tolerance_px: int = 3,
        routing_active_threshold: float = ROUTING_ACTIVE_THRESHOLD,
    ):
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.num_classes = int(num_classes)
        self.class_names = tuple(class_names)
        self.boundary_tolerance_px = int(boundary_tolerance_px)
        self.routing_active_threshold = float(routing_active_threshold)
        self.intersections = np.zeros(self.num_classes, dtype=np.int64)
        self.unions = np.zeros(self.num_classes, dtype=np.int64)
        self.correct = 0
        self.n_valid = 0
        self.n_images = 0
        self._boundary_values: list[float] = []
        self._component_eligible = 0
        self._component_recalled = 0
        self._routing_totals: np.ndarray | None = None
        self._routing_observations = 0
        self._cutoff_absolute_error = 0.0
        self._cutoff_n = 0

    def update(
        self,
        pred: Any,
        gt: Any,
        *,
        routing_weights: Any | None = None,
        predicted_cutoff: Any | None = None,
        cutoff_target: Any | None = None,
    ) -> None:
        prediction = _as_numpy(pred)
        target = _as_numpy(gt)
        if prediction.ndim == 2:
            prediction = prediction[None, ...]
        if target.ndim == 2:
            target = target[None, ...]
        if prediction.shape != target.shape or target.ndim != 3:
            raise ValueError(
                "pred and gt must share (B,H,W), got " f"{prediction.shape} and {target.shape}"
            )
        for image_pred, image_gt in zip(prediction, target, strict=True):
            counts = per_image_counts(image_pred, image_gt, num_classes=self.num_classes)
            self.intersections += counts["inter"]
            self.unions += counts["union"]
            self.correct += counts["correct"]
            self.n_valid += counts["n_valid"]
            bf1 = boundary_f1(
                image_pred,
                image_gt,
                num_classes=self.num_classes,
                tol_px=self.boundary_tolerance_px,
            )
            if np.isfinite(bf1):
                self._boundary_values.append(float(bf1))
            components = small_big_rock_component_counts(image_pred, image_gt)
            self._component_eligible += components["eligible"]
            self._component_recalled += components["recalled"]
            self.n_images += 1

        if routing_weights is not None:
            totals, observations = _routing_totals(routing_weights)
            if self._routing_totals is None:
                self._routing_totals = np.zeros_like(totals)
            if totals.shape != self._routing_totals.shape:
                raise ValueError("routing scale count changed between validation batches")
            self._routing_totals += totals
            self._routing_observations += observations

        one_cutoff_missing = (predicted_cutoff is None) != (cutoff_target is None)
        if one_cutoff_missing:
            raise ValueError("predicted_cutoff and cutoff_target must be supplied together")
        if predicted_cutoff is not None:
            predicted_value = _as_numpy(predicted_cutoff).astype(np.float64).reshape(-1)
            target_value = _as_numpy(cutoff_target).astype(np.float64).reshape(-1)
            if predicted_value.shape != target_value.shape:
                raise ValueError("predicted and target cutoff shapes differ")
            if predicted_value.size != prediction.shape[0]:
                raise ValueError("cutoff batch length must match prediction batch length")
            if not np.isfinite(predicted_value).all() or not np.isfinite(target_value).all():
                raise ValueError("cutoff values must be finite")
            self._cutoff_absolute_error += float(np.abs(predicted_value - target_value).sum())
            self._cutoff_n += int(predicted_value.size)

    def compute(self) -> dict[str, Any]:
        """Return a JSON-friendly validation report without mutating accumulator state."""
        class_set = fixed_class_set(self.unions)
        per_class = {
            name: _finite_or_none(iou_from_counts(self.intersections[index], self.unions[index]))
            for index, name in enumerate(self.class_names)
        }
        component_recall = (
            float(self._component_recalled / self._component_eligible)
            if self._component_eligible
            else None
        )
        routing = (
            _routing_summary_from_totals(
                self._routing_totals,
                self._routing_observations,
                active_threshold=self.routing_active_threshold,
            )
            if self._routing_totals is not None
            else None
        )
        return {
            "n_images": int(self.n_images),
            "miou": _finite_or_none(
                macro_miou_from_counts(self.intersections, self.unions, class_set)
            ),
            "per_class_iou": per_class,
            "pixel_accuracy": _finite_or_none(pixel_acc_from_counts(self.correct, self.n_valid)),
            "boundary_f1": (
                float(np.mean(self._boundary_values)) if self._boundary_values else None
            ),
            "small_big_rock_component_recall": component_recall,
            "small_big_rock_components_eligible": int(self._component_eligible),
            "small_big_rock_components_recalled": int(self._component_recalled),
            "cutoff_mae": (
                float(self._cutoff_absolute_error / self._cutoff_n) if self._cutoff_n else None
            ),
            "cutoff_n": int(self._cutoff_n),
            "routing_utilization": routing,
        }


def profile_forward_latency(
    model,
    input_tensor,
    *,
    warmups: int = 50,
    iterations: int = 200,
    use_fp16: bool = True,
) -> dict[str, Any]:
    """Profile batch-resident CUDA forward latency with event timing.

    ``input_tensor`` must already be on CUDA, so host-to-device transfer is excluded.  Every
    measured iteration is synchronized; CUDA events isolate device execution from synchronization
    overhead.  The model's original training/evaluation state is restored.
    """
    import torch

    if not torch.is_tensor(input_tensor) or not input_tensor.is_cuda:
        raise ValueError("input_tensor must be a CUDA tensor already resident on the target GPU")
    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be >= 0 and iterations must be > 0")
    was_training = bool(model.training)
    model.eval()
    timings: list[float] = []
    try:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(use_fp16)):
                for _ in range(warmups):
                    model(input_tensor)
                torch.cuda.synchronize(input_tensor.device)
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(input_tensor)
                    end.record()
                    end.synchronize()
                    timings.append(float(start.elapsed_time(end)))
    finally:
        model.train(was_training)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "p25_ms": float(np.percentile(values, 25)),
        "p75_ms": float(np.percentile(values, 75)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "warmups": int(warmups),
        "iterations": int(iterations),
        "precision": "fp16" if use_fp16 else str(input_tensor.dtype).removeprefix("torch."),
        "batch_size": int(input_tensor.shape[0]),
        "device": str(input_tensor.device),
    }
