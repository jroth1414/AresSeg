"""Safety tests for the isolated MPBA experiment runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "marsseg_mpba_runner", REPO_ROOT / "scripts" / "run_mpba_experiment.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_runner_confines_all_writes_to_mpba_tree(monkeypatch):
    cfg = {"results": {"root": "experiments/mpba"}}
    canonical = REPO_ROOT / "experiments" / "results_store.parquet"
    with pytest.raises(ValueError, match="MPBA output must be a child"):
        runner._guarded_run_dir(cfg, str(canonical), "unused")
    with pytest.raises(ValueError, match="reserved path"):
        runner._guarded_run_dir(cfg, "manifests", "unused")

    captured = {}

    def record_append(rows, *, parquet_path, csv_path):
        captured.update(rows=rows, parquet=Path(parquet_path), csv=Path(csv_path))

    monkeypatch.setattr(runner, "append_results", record_append)
    runner._append_mpba_results([{"run_id": "guard-test"}])
    assert captured["parquet"] == runner.MPBA_RESULTS_PARQUET
    assert captured["csv"] == runner.MPBA_RESULTS_CSV
    assert captured["parquet"].is_relative_to(runner.MPBA_ROOT)
    assert captured["csv"].is_relative_to(runner.MPBA_ROOT)
    assert captured["parquet"] != canonical


def test_gold_evaluation_is_locked_without_future_protocol_snapshot():
    with pytest.raises(PermissionError, match="gold evaluation is locked"):
        runner._authorize_gold({})


def test_official_screening_settings_are_sealed():
    cfg = runner.load_config(
        REPO_ROOT / "configs" / "mpba" / "unet_native.yaml",
        base_paths=[
            str(REPO_ROOT / "configs" / "data.yaml"),
            str(REPO_ROOT / "configs" / "mpba" / "base.yaml"),
        ],
    )
    runner._validate_arm(cfg)
    cfg["promotion"]["miou_gain_min"] = 0.0
    with pytest.raises(ValueError, match="promotion thresholds"):
        runner._validate_arm(cfg)


def test_data_cap_is_development_only():
    cfg = runner.load_config(
        REPO_ROOT / "configs" / "mpba" / "unet_native.yaml",
        ["data.max_train_images=1"],
        base_paths=[
            str(REPO_ROOT / "configs" / "data.yaml"),
            str(REPO_ROOT / "configs" / "mpba" / "base.yaml"),
        ],
    )
    with pytest.raises(ValueError, match="only with --fast-dev-run"):
        runner._validate_arm(cfg)
    runner._validate_arm(cfg, fast_dev_run=True)
