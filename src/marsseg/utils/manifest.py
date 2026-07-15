"""Run manifest — the single self-describing record that makes a run reproducible.

One flexible superset schema for every training/eval run: identity, git state, seed, hardware
profile, package versions, the resolved config and its hash, and the model/backbone/variant/dataset
being run. Callers add run-specific fields via ``**extra``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .capabilities import as_dict as caps_dict

REPO_ROOT = Path(__file__).resolve().parents[3]

# Packages whose versions are worth recording for reproducibility.
_TRACK = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scikit-image",
    "scipy",
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "transformers",
    "timm",
    "albumentations",
    "opencv-python-headless",
    "pillow",
    "segment-anything",
]


def _ver(p: str) -> str | None:
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def _git_output(*args: str) -> str:
    """Run a read-only Git query against this checkout without global config changes."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO_ROOT}", *args],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _git_sha() -> str:
    try:
        return _git_output("rev-parse", "HEAD")
    except Exception:
        return "UNKNOWN"


def _git_dirty() -> bool:
    try:
        return bool(_git_output("status", "--porcelain"))
    except Exception:
        return False


def config_hash(config: dict) -> str:
    """Stable sha256 of the resolved config (a dedup key for the results store)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_manifest(
    run_dir: str | Path,
    config: dict,
    seed: int,
    *,
    profile: str | None = None,
    model: str | None = None,
    backbone: str | None = None,
    variant: str | None = None,
    dataset: str | None = None,
    data_hashes: dict | None = None,
    stages_completed: list[str] | None = None,
    gpu_stages_skipped: list[str] | None = None,
    **extra,
) -> Path:
    """Write ``<run_dir>/manifest.json``. Returns the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    caps = caps_dict()
    m = {
        "run_id": run_dir.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "profile": profile or caps.get("profile", "unknown"),
        "capabilities": caps,
        "packages": {p: _ver(p) for p in _TRACK},
        "config": config,
        "config_hash": config_hash(config),
        "model": model,
        "backbone": backbone,
        "variant": variant,
        "dataset": dataset,
        "data_hashes": data_hashes or {},
        "stages_completed": stages_completed or [],
        "gpu_stages_skipped": gpu_stages_skipped or [],
    }
    m.update(extra)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return out
