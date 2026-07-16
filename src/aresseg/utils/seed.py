"""Deterministic seeding with documented limits.

Project default seed is 1414 (the multi-seed averaging set is [1414, 1415, 1416, 1417, 1418];
the evaluation bootstrap uses a distinct purpose-seed 0). Bit-exact reproducibility is
guaranteed only within identical hardware + library versions; where stochasticity remains we
report means over the seed set and record seed + versions + git SHA in the run manifest.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 1414
SEED_SET = [1414, 1415, 1416, 1417, 1418]
BOOTSTRAP_SEED = 0  # distinct purpose-seed for the eval bootstrap; NOT the project default


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed Python/NumPy/PyTorch and set deterministic flags. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best-effort full determinism; some ops have no deterministic kernel.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    return seed


# Backwards-compatible aliases referenced by some per-phase drafts.
def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    return set_seed(seed, deterministic)


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    return set_seed(seed, deterministic)


def set_all_seeds(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    return set_seed(seed, deterministic)
