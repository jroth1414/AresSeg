"""Hydra-lite config loader: load .env, merge YAML files, apply ``key=value`` CLI overrides.

No heavy dependencies. Behavior across the project comes from YAML under ``configs/`` rather
than hard-coded constants; this module is the single entry point that reads them.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/aresseg/utils/config.py -> parents[3] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env(dotenv_path: str | os.PathLike | None = None) -> None:
    """Load environment variables from ``.env`` (does not override already-set vars)."""
    load_dotenv(dotenv_path or (REPO_ROOT / ".env"), override=False)


def _deep_merge(a: dict, b: Mapping) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(out.get(k), dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce_scalar(raw: str) -> Any:
    """Auto-type a CLI override value. yaml.safe_load handles bools/lists/null/dicts and
    most numbers, but leaves dot-less scientific notation (e.g. ``2e-4``) as a string —
    coerce those to int/float here so ``train.lr=2e-4`` becomes a float."""
    val = yaml.safe_load(raw)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val
    return val


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply overrides like ``["train.lr=1e-3", "model.name=segformer"]`` (auto-typed via YAML)."""
    out = copy.deepcopy(cfg)
    for ov in overrides:
        key, _, raw = ov.partition("=")
        val: Any = _coerce_scalar(raw)  # auto-types ints/floats/bools/lists
        d = out
        *parents, leaf = key.split(".")
        for p in parents:
            nxt = d.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                d[p] = nxt
            d = nxt
        d[leaf] = val
    return out


def require_secret(name: str) -> str:
    """Return the env var ``name`` or raise a clear, actionable error if missing/empty."""
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"Missing required secret env var: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return v


def load_config(
    yaml_path: str | os.PathLike,
    overrides: list[str] | None = None,
    base_paths: list[str] | None = None,
) -> dict:
    """Load .env, deep-merge ``base_paths`` then ``yaml_path``, apply CLI overrides."""
    load_env()
    cfg: dict = {}
    for bp in base_paths or []:
        cfg = _deep_merge(cfg, load_yaml(bp))
    cfg = _deep_merge(cfg, load_yaml(yaml_path))
    return apply_overrides(cfg, overrides or [])
