#!/usr/bin/env python3
"""Analyze MPBA perspective priors on the MSL training partition only.

This command deliberately exposes no gold/test split option.  It applies the frozen by-image split
and analyzes only the resulting training records, excluding validation labels.  Its JSON report is
restricted to ``experiments/mpba`` and contains vertical class distributions, component-size versus
vertical-position correlations, and 30 m range-mask coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as connected_components
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aresseg.data.ai4mars import CLASSES, IGNORE_INDEX, build_index  # noqa: E402
from aresseg.data.dataset import make_splits  # noqa: E402
from aresseg.experimental.mpba.data import (  # noqa: E402
    apply_matched_range_cohort,
    attach_range_masks,
    cutoff_target_from_mask,
    read_range_mask,
)
from aresseg.utils.config import load_yaml  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="dataset extraction dir or its parent")
    parser.add_argument("--bins", type=int, default=16, help="equal-height vertical bins")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="deterministic train-only prefix for a smoke analysis (default: all training images)",
    )
    parser.add_argument(
        "--out",
        default="experiments/mpba/prior_analysis_seed1414.json",
        help="JSON destination, which must remain under experiments/mpba/",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="write progress to stderr every N images; 0 disables it",
    )
    return parser.parse_args(argv)


def _read_label(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"semantic label must be two-dimensional, got {mask.shape} from {path}")
    values = {int(value) for value in np.unique(mask)}
    allowed = {*range(len(CLASSES)), IGNORE_INDEX}
    if not values <= allowed:
        raise ValueError(f"unexpected semantic values {sorted(values - allowed)} in {path}")
    return mask


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if x.size < 2 or y.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def _record_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["label"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["range_mask"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def analyze_records(records: list[dict[str, Any]], *, bins: int) -> dict[str, Any]:
    """Compute the three train-only MPBA prior families over resolved records."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    vertical_counts = np.zeros((bins, len(CLASSES)), dtype=np.int64)
    component_areas: list[list[int]] = [[] for _ in CLASSES]
    component_centroid_y: list[list[float]] = [[] for _ in CLASSES]
    range_coverages: list[float] = []
    cutoffs: list[float] = []

    for record in records:
        semantic = _read_label(record["label"])
        height, width = semantic.shape
        row_edges = np.linspace(0, height, bins + 1, dtype=np.int64)
        if np.any(np.diff(row_edges) == 0):
            raise ValueError(f"{bins} bins exceed label height {height} for {record['name']}")
        for bin_index, (start, stop) in enumerate(zip(row_edges[:-1], row_edges[1:], strict=True)):
            values = semantic[start:stop]
            values = values[values != IGNORE_INDEX]
            if values.size:
                vertical_counts[bin_index] += np.bincount(values.ravel(), minlength=len(CLASSES))[
                    : len(CLASSES)
                ]

        normalized_denominator = max(height - 1, 1)
        for class_index in range(len(CLASSES)):
            components, count = connected_components(
                semantic == class_index, structure=np.ones((3, 3), dtype=np.uint8)
            )
            if count == 0:
                continue
            areas = np.bincount(components.ravel(), minlength=count + 1)[1:]
            rows, columns = np.nonzero(components)
            component_ids = components[rows, columns]
            row_sums = np.bincount(
                component_ids,
                weights=rows.astype(np.float64),
                minlength=count + 1,
            )[1:]
            centroids = row_sums / areas / normalized_denominator
            component_areas[class_index].extend(int(value) for value in areas)
            component_centroid_y[class_index].extend(float(value) for value in centroids)

        range_mask = read_range_mask(record["range_mask"])
        if range_mask.shape != semantic.shape:
            raise ValueError(
                f"range/semantic shape mismatch for {record['name']}: "
                f"{range_mask.shape} vs {semantic.shape}"
            )
        range_coverages.append(float(np.mean(range_mask != 0)))
        cutoffs.append(cutoff_target_from_mask(range_mask))

    class_totals = vertical_counts.sum(axis=0)
    vertical_rows = []
    for bin_index, counts in enumerate(vertical_counts):
        valid = int(counts.sum())
        vertical_rows.append(
            {
                "bin": bin_index,
                "y_start": float(bin_index / bins),
                "y_stop": float((bin_index + 1) / bins),
                "valid_pixels": valid,
                "class_pixels": {name: int(counts[index]) for index, name in enumerate(CLASSES)},
                "class_share_of_valid_pixels": {
                    name: (float(counts[index] / valid) if valid else None)
                    for index, name in enumerate(CLASSES)
                },
                "fraction_of_class_pixels": {
                    name: (
                        float(counts[index] / class_totals[index]) if class_totals[index] else None
                    )
                    for index, name in enumerate(CLASSES)
                },
            }
        )

    component_report: dict[str, Any] = {}
    for class_index, class_name in enumerate(CLASSES):
        areas = np.asarray(component_areas[class_index], dtype=np.float64)
        centroid_y = np.asarray(component_centroid_y[class_index], dtype=np.float64)
        component_report[class_name] = {
            "area_pixels": _summary(areas),
            "centroid_y": _summary(centroid_y),
            "pearson_centroid_y_vs_log1p_area": _safe_correlation(
                centroid_y, np.log1p(areas), "pearson"
            ),
            "spearman_centroid_y_vs_area": _safe_correlation(centroid_y, areas, "spearman"),
            "components_16_to_256_pixels": int(np.count_nonzero((areas >= 16) & (areas <= 256))),
        }

    coverage_array = np.asarray(range_coverages, dtype=np.float64)
    cutoff_array = np.asarray(cutoffs, dtype=np.float64)
    return {
        "vertical_class_distribution": {
            "n_bins": int(bins),
            "class_pixel_totals": {
                name: int(class_totals[index]) for index, name in enumerate(CLASSES)
            },
            "bins": vertical_rows,
        },
        "component_size_correlations": component_report,
        "range_mask_coverage": {
            "foreground_pixel_fraction": _summary(coverage_array),
            "cutoff_fraction_of_image_height": _summary(cutoff_array),
            "empty_masks": int(np.count_nonzero(coverage_array == 0.0)),
            "full_masks": int(np.count_nonzero(coverage_array == 1.0)),
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")

    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    data_cfg = cfg["data"]
    root = Path(args.root) if args.root else REPO_ROOT / data_cfg["root"]
    indexed_train = build_index(root, rover="msl", camera=data_cfg["camera"])["train"]
    split = make_splits(
        indexed_train,
        val_frac=float(data_cfg["val_frac"]),
        seed=int(data_cfg["split_seed"]),
    )
    split, exclusions = apply_matched_range_cohort(split)
    training_records = split["train"]
    if args.max_images is not None:
        training_records = training_records[: args.max_images]
    # Strict resolution is intentional: privileged supervision must never silently disappear.
    records = attach_range_masks(training_records, strict=True)

    # Progress is intentionally outside analyze_records' pure API.  Resolve all records above so a
    # missing mask fails before a potentially long scan starts.
    if args.progress_every:
        print(
            f"analyzing {len(records)} train-only records (validation excluded)",
            file=sys.stderr,
        )
    analysis = analyze_records(records, bins=int(args.bins))
    report = {
        "analysis": "mpba_perspective_priors",
        "scope": "msl_ncam_training_partition_only",
        "gold_evaluated": False,
        "validation_labels_evaluated": False,
        "split_seed": int(data_cfg["split_seed"]),
        "val_frac": float(data_cfg["val_frac"]),
        "indexed_training_pool_n": len(indexed_train),
        "validation_records_excluded_n": len(split["val"]),
        "matched_cohort_exclusions": exclusions,
        "matched_cohort_excluded_n": len(exclusions),
        "analyzed_records_n": len(records),
        "max_images": args.max_images,
        "record_fingerprint_sha256": _record_fingerprint(records),
        **analysis,
    }

    destination = Path(args.out)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    allowed_root = (REPO_ROOT / "experiments" / "mpba").resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(allowed_root):
        raise ValueError(f"--out must remain under {allowed_root}, got {destination}")
    _atomic_json(destination, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"report={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
