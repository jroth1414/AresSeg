#!/usr/bin/env python3
"""Validate AI4Mars before any experiment reads training or gold labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from marsseg.data.preflight import inspect_data  # noqa: E402
from marsseg.utils.config import load_yaml  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="dataset extraction dir or its parent")
    parser.add_argument("--expected-msl-train", type=int, default=None)
    parser.add_argument("--expected-msl-test", type=int, default=None)
    parser.add_argument("--expected-mer-test", type=int, default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--allow-missing-archive", action="store_true")
    parser.add_argument("--skip-archive-hash", action="store_true")
    parser.add_argument("--skip-mask-scan", action="store_true")
    parser.add_argument(
        "--json-out",
        default="data/preflight.json",
        help="machine-readable report (written even when validation fails)",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    data_cfg = cfg["data"]
    mer_cfg = cfg["mer"]
    root = Path(args.root) if args.root else REPO_ROOT / data_cfg["root"]
    report = inspect_data(
        root,
        expected_msl_train=(
            args.expected_msl_train
            if args.expected_msl_train is not None
            else data_cfg.get("expected_train_n", 16064)
        ),
        expected_msl_test=(
            args.expected_msl_test
            if args.expected_msl_test is not None
            else data_cfg["expected_test_n"]
        ),
        expected_mer_test=(
            args.expected_mer_test
            if args.expected_mer_test is not None
            else mer_cfg["expected_test_n"]
        ),
        camera=data_cfg["camera"],
        msl_gold_dir=data_cfg["test_gold_dir"],
        mer_gold_dir=mer_cfg["test_gold_dir"],
        val_frac=data_cfg["val_frac"],
        split_seed=data_cfg["split_seed"],
        archive_path=args.archive,
        require_archive=not args.allow_missing_archive,
        hash_archive=not args.skip_archive_hash,
        scan_masks=not args.skip_mask_scan,
    )
    destination = Path(args.json_out)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    _atomic_json(destination, report)
    print(json.dumps(report, indent=2))
    print(f"report={destination}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
