"""Lightweight, appendable, long/tidy metrics store.

The canonical metrics table ``experiments/results_store.parquet`` (+ a ``.csv`` mirror), one metric
value per row. The evaluation aggregation builds on this same schema; ``scripts/merge_results.py``
dedups rows produced on a GPU profile back into it on ``config_hash``. ``status`` records the
skip-and-log contract for gated models ("skipped").
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

RESULTS_STORE_PARQUET = Path("experiments/results_store.parquet")
RESULTS_STORE_CSV = Path("experiments/results_store.csv")

# One metric value per row (long/tidy). Extra columns let GPU-profile rows merge back cleanly.
RESULT_COLUMNS = [
    "run_id",
    "model",  # baseline|unet|deeplabv3plus|segformer|dinov3_sat|sam
    "backbone",  # resnet34|efficientnet-b0|mit-b0|vitl16-sat493m|...
    "variant",  # pretrained|scratch|zeroshot|finetuned
    "scope",  # 'ALL' (overall) or a class name (per-class)
    "stratum",  # all|per_class|in_rover|cross_rover|pretrained|scratch
    "metric",  # miou|iou|pixel_acc|boundary_f1|n
    "value",
    "ci_low",
    "ci_high",
    "status",  # ok|skipped|failed
    "profile",  # windows_cpu|gpu_full
    "seed",
    "git_sha",
    "config_hash",  # dedup key for merging GPU-profile rows
    "total_parameters",
    "trainable_parameters",
]

# Dedup key used by scripts/merge_results.py when merging GPU-profile rows back in.
DEDUP_KEYS = [
    "run_id",
    "model",
    "backbone",
    "variant",
    "scope",
    "stratum",
    "metric",
    "config_hash",
]


def _temporary_sibling(path: Path) -> Path:
    """Create a same-filesystem temporary path suitable for an atomic os.replace."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(name)


def append_results(
    rows: list[dict],
    parquet_path: os.PathLike = RESULTS_STORE_PARQUET,
    csv_path: os.PathLike = RESULTS_STORE_CSV,
) -> Path:
    """Upsert metric rows and atomically replace the parquet and CSV mirrors.

    Re-running a completed arm no longer duplicates its rows: the canonical result identity in
    DEDUP_KEYS is last-write-wins. Both complete files are staged beside their destinations
    before either visible store is replaced.
    """
    incoming = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in incoming.columns:
            incoming[column] = None
    incoming = incoming[RESULT_COLUMNS]

    pq = Path(parquet_path)
    csv = Path(csv_path)
    pq.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    if pq.exists():
        existing = pd.read_parquet(pq)
        for column in RESULT_COLUMNS:
            if column not in existing.columns:
                existing[column] = None
        frame = pd.concat([existing[RESULT_COLUMNS], incoming], ignore_index=True)
    else:
        frame = incoming
    frame = frame.drop_duplicates(subset=DEDUP_KEYS, keep="last", ignore_index=True)

    pq_tmp = _temporary_sibling(pq)
    csv_tmp = _temporary_sibling(csv)
    try:
        frame.to_parquet(pq_tmp, index=False)
        frame.to_csv(csv_tmp, index=False)
        os.replace(pq_tmp, pq)
        os.replace(csv_tmp, csv)
    finally:
        pq_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
    return pq


def read_results(parquet_path: os.PathLike = RESULTS_STORE_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)
