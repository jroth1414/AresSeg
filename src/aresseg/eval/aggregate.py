"""Per-image counts table (7.4) construction + aggregation to results-store rows (MS3).

``image_rows`` renders one image's metrics into the EXACT per_image.parquet schema; ``store_rows``
reduces a per-image table to split-level RESULT_COLUMNS rows (miou / per-class iou from SUMMED
counts; pixel_acc / boundary_f1 descriptive; n). Appending to the store goes through
``utils.results.append_results`` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.ai4mars import CLASSES
from .metrics import iou_from_counts
from .stats import bootstrap_miou_ci, counts_from_per_image

PER_IMAGE_COLUMNS = [
    "run_id",
    "name",
    "split",  # val | test_msl | test_mer
    "scope",  # ALL or a class name
    "metric",  # iou | pixel_acc | boundary_f1
    "value",
    "inter",  # int64; 0 for ALL and for boundary_f1
    "union",  # int64; 0 for ALL and for boundary_f1
    "n_valid",  # int64
]


def image_rows(
    run_id: str, name: str, split: str, counts: dict, bf1: float, classes: list[str] = CLASSES
) -> list[dict]:
    """Rows for ONE image from ``metrics.per_image_counts`` output + its boundary F1."""
    n_valid = counts["n_valid"]
    rows = []
    for c, cls in enumerate(classes):
        inter, union = int(counts["inter"][c]), int(counts["union"][c])
        rows.append(
            {
                "run_id": run_id,
                "name": name,
                "split": split,
                "scope": cls,
                "metric": "iou",
                "value": (inter / union) if union > 0 else float("nan"),
                "inter": inter,
                "union": union,
                "n_valid": n_valid,
            }
        )
    pixel_acc = (counts["correct"] / n_valid) if n_valid > 0 else float("nan")
    for metric, value in (("pixel_acc", pixel_acc), ("boundary_f1", bf1)):
        rows.append(
            {
                "run_id": run_id,
                "name": name,
                "split": split,
                "scope": "ALL",
                "metric": metric,
                "value": value,
                "inter": 0,
                "union": 0,
                "n_valid": n_valid,
            }
        )
    return rows


def per_image_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[PER_IMAGE_COLUMNS]
    df["inter"] = df["inter"].astype("int64")
    df["union"] = df["union"].astype("int64")
    df["n_valid"] = df["n_valid"].astype("int64")
    return df


def store_rows(
    per_image: pd.DataFrame,
    *,
    split: str,
    stratum: str,
    run_id: str,
    model: str,
    backbone: str,
    variant: str,
    profile: str,
    seed: int,
    git_sha: str,
    config_hash: str,
    status: str = "ok",
    classes: list[str] = CLASSES,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    ci_level: float = 0.90,
) -> list[dict]:
    """Split-level result rows, including an image-bootstrap CI for aggregate mIoU."""
    base = {
        "run_id": run_id,
        "model": model,
        "backbone": backbone,
        "variant": variant,
        "status": status,
        "profile": profile,
        "seed": seed,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "ci_low": None,
        "ci_high": None,
    }
    sub = per_image[per_image["split"] == split]
    names, inter, union = counts_from_per_image(per_image, split, classes)
    inter_sums = inter.sum(axis=1, dtype=np.int64)
    union_sums = union.sum(axis=1, dtype=np.int64)
    miou = bootstrap_miou_ci(
        inter,
        union,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        ci_level=ci_level,
    )
    n_images = len(names)
    rows = [
        {
            **base,
            "scope": "ALL",
            "stratum": stratum,
            "metric": "miou",
            "value": miou["miou"],
            "ci_low": miou["ci_low"],
            "ci_high": miou["ci_high"],
        },
        {**base, "scope": "ALL", "stratum": stratum, "metric": "n", "value": float(n_images)},
    ]
    # per-class rows: 'per_class' for standard runs; H4 eval runs keep in_rover/cross_rover so
    # the two splits of one run_id never collide on DEDUP_KEYS (7.1)
    per_class_stratum = "per_class" if stratum == "all" else stratum
    for c, cls in enumerate(classes):
        rows.append(
            {
                **base,
                "scope": cls,
                "stratum": per_class_stratum,
                "metric": "iou",
                "value": iou_from_counts(inter_sums[c], union_sums[c]),
            }
        )
    pa = sub[sub["metric"] == "pixel_acc"]
    if len(pa):  # split-level pixel_acc = sum(correct)/sum(valid); correct_i = value_i * n_valid_i
        correct = float((pa["value"] * pa["n_valid"]).sum())
        rows.append(
            {
                **base,
                "scope": "ALL",
                "stratum": stratum,
                "metric": "pixel_acc",
                "value": correct / float(pa["n_valid"].sum()),
            }
        )
    bf = sub[sub["metric"] == "boundary_f1"]["value"].dropna()
    if len(bf):  # descriptive-only: reported as the mean of per-image values
        rows.append(
            {
                **base,
                "scope": "ALL",
                "stratum": stratum,
                "metric": "boundary_f1",
                "value": float(bf.mean()),
            }
        )
    return rows
