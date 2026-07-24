#!/usr/bin/env python3
"""Train and validate one perspective-first MPBA screening arm.

The runner is intentionally isolated from Protocol V3:

* validation is the default and the only evaluation used by the current ten-arm screen;
* every writable path is required to live below ``experiments/mpba``;
* results are appended only to the MPBA-local store; and
* gold evaluation is denied unless a future frozen MPBA protocol snapshot and its checksum are
  both present, valid, and explicitly authorize gold scoring.

Example::

    .venv/bin/python scripts/run_mpba_experiment.py \
        --config configs/mpba/unet_raw_y.yaml
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from aresseg.data.ai4mars import IGNORE_INDEX, NUM_CLASSES, build_index  # noqa: E402
from aresseg.data.dataset import SegDataset, class_pixel_counts, make_splits  # noqa: E402
from aresseg.data.preflight import index_fingerprint  # noqa: E402
from aresseg.data.transforms import eval_transform  # noqa: E402
from aresseg.eval import aggregate, metrics  # noqa: E402
from aresseg.experimental.mpba import build_mpba_model  # noqa: E402
from aresseg.experimental.mpba.data import (  # noqa: E402
    MATCHED_COHORT_EXCLUDED_PRODUCTS,
    RangeAwareSegDataset,
    apply_matched_range_cohort,
    attach_range_masks,
)
from aresseg.experimental.mpba.lit import (  # noqa: E402
    MPBADataModule,
    MPBALitModule,
    final_logits,
    output_field,
)
from aresseg.experimental.mpba.metrics import (  # noqa: E402
    MPBAValidationAccumulator,
    profile_forward_latency,
)
from aresseg.models.zoo import build_model  # noqa: E402
from aresseg.utils.capabilities import detect  # noqa: E402
from aresseg.utils.config import load_config  # noqa: E402
from aresseg.utils.logging import get_logger  # noqa: E402
from aresseg.utils.manifest import (  # noqa: E402
    _git_dirty,
    _git_sha,
    config_hash,
    write_manifest,
)
from aresseg.utils.results import append_results  # noqa: E402
from aresseg.utils.seed import set_seed  # noqa: E402

log = get_logger("aresseg.mpba.run")

MPBA_ROOT = REPO_ROOT / "experiments" / "mpba"
MPBA_PROTOCOL_ROOT = MPBA_ROOT / "protocol"
MPBA_RESULTS_PARQUET = MPBA_ROOT / "results_store.parquet"
MPBA_RESULTS_CSV = MPBA_ROOT / "results_store.csv"
ALLOWED_ARMS = {
    "native": (False, "content", "none"),
    "static": (True, "static", "none"),
    "content": (True, "content", "none"),
    "raw_y": (True, "content", "raw_y"),
    "range_cutoff": (True, "content", "range_cutoff"),
}
ALLOWED_BACKBONES = {("unet", "resnet34"), ("segformer", "b0")}
SCREEN_PROTOCOL_VERSION = "mpba-perspective-first-seed1414-v1"
EXPECTED_EXPERIMENT = {
    "family": "mpba_perspective_first",
    "phase": "seed1414_validation_screen",
}
EXPECTED_DATA = {
    "root": "data/raw/ai4mars/ai4mars-dataset-merged-0.6",
    "rover": "msl",
    "camera": "ncam",
    "val_frac": 0.2,
    "size": 512,
    "seed": 1414,
    "split_by": "image",
    "split_seed": 1414,
    "test_gold_dir": "msl/ncam/labels/test/masked-gold-min3-100agree",
    "expected_train_n": 16064,
    "expected_test_n": 322,
}
EXPECTED_CLASS_WEIGHTS = {
    "method": "inverse_freq_normalized",
    "formula": "w_c = median(counts)/counts_c",
    "clip": [0.5, 10.0],
    "computed_on": "train_split",
    "max_images": None,
}
EXPECTED_AUGMENTATION = {
    "hflip_p": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "rbc_p": 0.3,
    "vflip": False,
    "vflip_p": 0.5,
    "scale_crop": False,
    "scale_limit": 0.1,
    "scale_crop_p": 0.3,
}
EXPECTED_TRAIN = {
    "batch_size": 8,
    "num_workers": 8,
    "max_epochs": 50,
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "dice_weight": 1.0,
    "ignore_index": 255,
    "early_stop_patience": 10,
    "grad_clip": 1.0,
}
EXPECTED_EVALUATION = {
    "split": "validation",
    "boundary_tolerance_px": 3,
    "big_rock_class": 3,
    "small_component_min_pixels": 16,
    "small_component_max_pixels": 256,
    "small_component_coverage": 0.5,
}
EXPECTED_PROFILING = {
    "batch_size": 1,
    "image_size": 512,
    "precision": "fp16",
    "warmup_iterations": 50,
    "measured_iterations": 200,
    "required_gpu_substring": "V100",
}
EXPECTED_PROMOTION = {
    "miou_gain_min": 0.003,
    "big_rock_iou_gain_min": 0.01,
    "max_other_class_drop": 0.005,
    "max_median_latency_increase": 0.15,
    "routing_scale_usage_min": 0.10,
    "routing_scales_required": 2,
    "range_cutoff_mae_max": 0.05,
    "raw_y_tie_margin": 0.001,
}
SEGFORMER_REVISION = "25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc"
SEGFORMER_WEIGHTS_SHA256 = "3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a"


def _require_exact(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"official MPBA {label} is sealed to {expected!r}; got {actual!r}")


def _runtime_code_fingerprint() -> str:
    """Hash the additive runner/model/data/metric implementation used by a screen."""
    paths = [Path(__file__).resolve()]
    paths.extend(sorted((REPO_ROOT / "src" / "aresseg" / "experimental" / "mpba").glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _selected_gpu_identity() -> dict[str, str]:
    """Return the UUID/name of the physical GPU selected by CUDA visibility."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    selector = visible or str(torch.cuda.current_device())
    command = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=uuid,name",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not resolve the selected physical GPU UUID") from exc
    if "\n" in output or "," not in output:
        raise RuntimeError(f"ambiguous selected GPU identity: {output!r}")
    uuid, name = (part.strip() for part in output.split(",", 1))
    if not uuid.startswith("GPU-") or not name:
        raise RuntimeError(f"malformed selected GPU identity: {output!r}")
    return {"uuid": uuid, "name": name}


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="one of configs/mpba/*.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra-lite key=value overrides")
    parser.add_argument(
        "--out",
        default=None,
        help="run directory; relative paths are resolved below experiments/mpba",
    )
    parser.add_argument(
        "--eval-split",
        choices=("validation", "none", "gold"),
        default="validation",
        help="defaults to validation; gold is locked behind a future frozen MPBA protocol",
    )
    parser.add_argument(
        "--eval-only",
        metavar="CHECKPOINT",
        default=None,
        help="skip fitting and load a checkpoint into the model reconstructed from config",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="run Lightning's one-batch CPU/GPU development check; never promotion-eligible",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config, arm, gold lock, and output path without reading data or weights",
    )
    return parser.parse_args(argv)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _configured_results_root(cfg: dict) -> Path:
    configured = Path(cfg.get("results", {}).get("root", "experiments/mpba"))
    if not configured.is_absolute():
        configured = REPO_ROOT / configured
    configured = configured.resolve()
    expected = MPBA_ROOT.resolve()
    if configured != expected:
        raise ValueError(
            "MPBA results.root is sealed to experiments/mpba; "
            f"refusing configured path {configured}"
        )
    return expected


def _guarded_run_dir(cfg: dict, requested: str | None, generated_id: str) -> Path:
    root = _configured_results_root(cfg)
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        candidate = root / generated_id
    candidate = candidate.resolve()
    if candidate == root or not _inside(candidate, root):
        raise ValueError(f"MPBA output must be a child of {root}; got {candidate}")
    # Refuse the canonical store and mirror trees even if supplied as a run path.
    reserved = {root / "manifests", root / "protocol"}
    if any(candidate == path or _inside(candidate, path) for path in reserved):
        raise ValueError(f"MPBA run output cannot use reserved path {candidate}")
    return candidate


def _resolve_protocol_file(raw: str, label: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    protocol_root = MPBA_PROTOCOL_ROOT.resolve()
    if not _inside(candidate, protocol_root):
        raise PermissionError(f"{label} must live below {protocol_root}; got {candidate}")
    if not candidate.is_file():
        raise PermissionError(f"gold evaluation locked: missing {label} {candidate}")
    return candidate


def _authorize_gold(cfg: dict) -> dict:
    """Verify a future frozen snapshot + checksum before any gold record can be scored."""
    protocol = cfg.get("mpba_protocol")
    if not isinstance(protocol, dict):
        raise PermissionError(
            "gold evaluation is locked for the perspective-first screen: no future "
            "mpba_protocol snapshot/checksum is configured"
        )
    snapshot_raw = protocol.get("snapshot")
    checksum_raw = protocol.get("checksum")
    if not snapshot_raw or not checksum_raw:
        raise PermissionError(
            "gold evaluation is locked: both mpba_protocol.snapshot and .checksum are required"
        )
    snapshot = _resolve_protocol_file(str(snapshot_raw), "protocol snapshot")
    checksum = _resolve_protocol_file(str(checksum_raw), "protocol checksum")
    expected = checksum.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise PermissionError(f"gold evaluation locked: malformed SHA-256 in {checksum}")
    actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if actual != expected:
        raise PermissionError(
            "gold evaluation locked: MPBA protocol snapshot checksum mismatch "
            f"({actual} != {expected})"
        )
    try:
        document = json.loads(snapshot.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PermissionError("gold evaluation locked: protocol snapshot must be JSON") from exc
    if document.get("protocol") != "mpba" or document.get("frozen") is not True:
        raise PermissionError("gold evaluation locked: snapshot is not a frozen MPBA protocol")
    if document.get("gold_evaluation_authorized") is not True:
        raise PermissionError("gold evaluation locked: snapshot does not authorize gold scoring")
    return {
        "snapshot": str(snapshot.relative_to(REPO_ROOT)),
        "checksum": str(checksum.relative_to(REPO_ROOT)),
        "snapshot_sha256": actual,
    }


def _validate_arm(cfg: dict, *, fast_dev_run: bool = False) -> tuple[dict, dict, dict, str]:
    model_cfg = cfg.get("model", {})
    mpba_cfg = cfg.get("mpba", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    variant = mpba_cfg.get("variant")
    if variant not in ALLOWED_ARMS:
        raise ValueError(f"MPBA variant must be one of {sorted(ALLOWED_ARMS)}; got {variant!r}")
    enabled, router_mode, coordinate_mode = ALLOWED_ARMS[variant]
    actual = (
        bool(mpba_cfg.get("enabled")),
        mpba_cfg.get("router_mode"),
        mpba_cfg.get("coordinate_mode"),
    )
    if actual != (enabled, router_mode, coordinate_mode):
        raise ValueError(
            f"variant {variant!r} requires enabled/router/coordinate "
            f"{(enabled, router_mode, coordinate_mode)}, got {actual}"
        )
    model_key = (model_cfg.get("name"), model_cfg.get("backbone"))
    if model_key not in ALLOWED_BACKBONES:
        raise ValueError(
            "first-cycle MPBA supports only pretrained U-Net/ResNet-34 and "
            f"SegFormer/MiT-B0; got {model_key}"
        )
    if model_cfg.get("pretrained") is not True:
        raise ValueError("first-cycle MPBA configs must use pretrained encoders")
    if model_key == ("unet", "resnet34"):
        _require_exact("model.results_backbone", model_cfg.get("results_backbone"), "resnet34")
        _require_exact("model.revision", model_cfg.get("revision"), None)
    else:
        _require_exact("model.results_backbone", model_cfg.get("results_backbone"), "mit-b0")
        _require_exact("model.revision", model_cfg.get("revision"), SEGFORMER_REVISION)
        _require_exact(
            "model.weights_filename", model_cfg.get("weights_filename"), "model.safetensors"
        )
        _require_exact(
            "model.weights_sha256",
            model_cfg.get("weights_sha256"),
            SEGFORMER_WEIGHTS_SHA256,
        )
    seed = int(cfg.get("data", {}).get("seed", 1414))
    if seed != 1414:
        raise ValueError(f"first-cycle screening seed is fixed at 1414; got {seed}")
    if int(mpba_cfg.get("num_scales", 4)) != 4:
        raise ValueError("MPBA screening requires exactly four encoder scales")
    if int(mpba_cfg.get("projection_channels", 128)) != 128:
        raise ValueError("MPBA screening projection width is fixed at 128 channels")
    if float(mpba_cfg.get("cutoff_loss_weight", 0.1)) != 0.1:
        raise ValueError("MPBA range cutoff loss weight is fixed at 0.1")
    if mpba_cfg.get("strict_range_masks") is not True:
        raise ValueError("MPBA screening requires strict range-mask resolution")
    configured_exclusions = tuple(mpba_cfg.get("matched_cohort_exclusions", ()))
    if configured_exclusions != MATCHED_COHORT_EXCLUDED_PRODUCTS:
        raise ValueError(
            "MPBA matched-cohort exclusions must match the pinned source snapshot: "
            f"{MATCHED_COHORT_EXCLUDED_PRODUCTS}"
        )
    if int(cfg.get("data", {}).get("size", 512)) != 512:
        raise ValueError("MPBA screening input size is fixed at 512")
    _require_exact("experiment", cfg.get("experiment"), EXPECTED_EXPERIMENT)
    for key, expected in EXPECTED_DATA.items():
        _require_exact(f"data.{key}", data_cfg.get(key), expected)
    max_train = data_cfg.get("max_train_images")
    if max_train is not None:
        if not fast_dev_run:
            raise ValueError("data.max_train_images is allowed only with --fast-dev-run")
        if int(max_train) <= 0:
            raise ValueError("data.max_train_images must be positive")
    _require_exact("class_weights", cfg.get("class_weights"), EXPECTED_CLASS_WEIGHTS)
    _require_exact("augmentation", cfg.get("aug"), EXPECTED_AUGMENTATION)
    _require_exact("training settings", train_cfg, EXPECTED_TRAIN)
    _require_exact("evaluation settings", cfg.get("evaluation"), EXPECTED_EVALUATION)
    _require_exact("profiling settings", cfg.get("profiling"), EXPECTED_PROFILING)
    _require_exact("promotion thresholds", cfg.get("promotion"), EXPECTED_PROMOTION)
    _require_exact("results.root", cfg.get("results"), {"root": "experiments/mpba"})
    _require_exact("mpba.range_mask_dir", mpba_cfg.get("range_mask_dir"), None)
    return model_cfg, mpba_cfg, train_cfg, str(variant)


def _screen_protocol(
    cfg: dict,
    *,
    matched_split_fingerprint: str,
    git_sha: str,
    runtime_code_sha256: str,
    gpu_identity: dict[str, str] | None,
) -> tuple[dict[str, Any], str]:
    """Build the arm-independent protocol identity required for ten-run promotion."""
    mpba_cfg = cfg["mpba"]
    payload: dict[str, Any] = {
        "version": SCREEN_PROTOCOL_VERSION,
        "git_sha": git_sha,
        "runtime_code_sha256": runtime_code_sha256,
        "matched_train_validation_sha256": matched_split_fingerprint,
        "gpu": gpu_identity,
        "experiment": cfg["experiment"],
        "data": cfg["data"],
        "class_weights": cfg["class_weights"],
        "augmentation": cfg["aug"],
        "training": cfg["train"],
        "evaluation": cfg["evaluation"],
        "profiling": cfg["profiling"],
        "promotion": cfg["promotion"],
        "adapter_shared": {
            key: mpba_cfg[key]
            for key in (
                "projection_channels",
                "num_scales",
                "cutoff_loss_weight",
                "range_mask_dir",
                "strict_range_masks",
                "matched_cohort_exclusions",
            )
        },
        "model_families": {
            "unet": {"backbone": "resnet34", "pretrained": True},
            "segformer": {
                "backbone": "b0",
                "results_backbone": "mit-b0",
                "pretrained": True,
                "revision": SEGFORMER_REVISION,
                "weights_sha256": SEGFORMER_WEIGHTS_SHA256,
            },
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def _class_weights(records: list[dict], cfg: dict) -> list[float]:
    counts = class_pixel_counts(
        records,
        num_classes=NUM_CLASSES,
        max_images=cfg.get("max_images"),
    ).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    lower, upper = cfg.get("clip", [0.5, 10.0])
    return [float(value) for value in np.clip(np.median(counts) / counts, lower, upper)]


def _build_screening_model(model_cfg: dict, mpba_cfg: dict) -> torch.nn.Module:
    kwargs = dict(
        backbone=model_cfg.get("backbone"),
        num_classes=NUM_CLASSES,
        pretrained=True,
        revision=model_cfg.get("revision"),
    )
    if not mpba_cfg["enabled"]:
        model = build_model(model_cfg["name"], **kwargs)
    else:
        model = build_mpba_model(
            model_cfg["name"],
            **kwargs,
            router_mode=mpba_cfg["router_mode"],
            coordinate_mode=mpba_cfg["coordinate_mode"],
            projection_channels=int(mpba_cfg["projection_channels"]),
        )
    if model is None:
        raise RuntimeError("MPBA screening model construction returned no model")
    return model


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    parameters = list(model.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    )


def _load_checkpoint(module: MPBALitModule, checkpoint: str | Path, device: str) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {checkpoint}")
    module.load_state_dict(state, strict=True)


def _evaluation_loader(
    records: list[dict],
    *,
    size: int,
    batch_size: int,
    num_workers: int,
    include_cutoff_targets: bool,
) -> DataLoader:
    dataset_type = RangeAwareSegDataset if include_cutoff_targets else SegDataset
    dataset = dataset_type(records, eval_transform(size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(
    module: MPBALitModule,
    loader: DataLoader,
    *,
    run_dir: Path,
    run_id: str,
    split: str,
    device: str,
    boundary_tolerance_px: int,
) -> tuple[dict[str, Any], list[dict]]:
    accumulator = MPBAValidationAccumulator(
        boundary_tolerance_px=boundary_tolerance_px,
    )
    prediction_dir = run_dir / "preds" / split
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    module = module.to(device).eval()
    autocast_enabled = device == "cuda"
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device,
                dtype=torch.float16 if autocast_enabled else torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = module.forward_with_aux(images)
                logits = final_logits(output)
            predictions = logits.argmax(1).cpu().numpy().astype(np.uint8)
            targets = batch["mask"].numpy()
            routing_weights = output_field(output, "routing_weights", "route_weights")
            predicted_cutoff = None
            cutoff_target = None
            if module.uses_range_cutoff and "cutoff_target" in batch:
                predicted_cutoff = output_field(output, "predicted_cutoff", "cutoff")
                cutoff_target = batch["cutoff_target"]
            accumulator.update(
                predictions,
                targets,
                routing_weights=routing_weights,
                predicted_cutoff=predicted_cutoff,
                cutoff_target=cutoff_target,
            )
            for index, name in enumerate(batch["name"]):
                prediction = predictions[index]
                target = targets[index]
                saved = prediction.copy()
                saved[target == IGNORE_INDEX] = IGNORE_INDEX
                if not cv2.imwrite(str(prediction_dir / f"{name}.png"), saved):
                    raise OSError(f"failed to write prediction for {name}")
                counts = metrics.per_image_counts(prediction, target)
                boundary = metrics.boundary_f1(
                    prediction,
                    target,
                    tol_px=boundary_tolerance_px,
                )
                rows.extend(aggregate.image_rows(run_id, str(name), split, counts, boundary))
    return accumulator.compute(), rows


def _profile_latency(
    module: MPBALitModule,
    cfg: dict,
    *,
    device: str,
) -> dict[str, Any] | None:
    if device != "cuda":
        return None
    profile_cfg = cfg["profiling"]
    gpu_identity = _selected_gpu_identity()
    gpu_name = gpu_identity["name"]
    required = profile_cfg.get("required_gpu_substring", "V100")
    if required and str(required).lower() not in gpu_name.lower():
        raise RuntimeError(
            f"official MPBA latency must be profiled on {required}; current GPU is {gpu_name}"
        )
    size = int(profile_cfg.get("image_size", 512))
    batch_size = int(profile_cfg.get("batch_size", 1))
    sample = torch.zeros(batch_size, 3, size, size, device=device)
    report = profile_forward_latency(
        module,
        sample,
        warmups=int(profile_cfg.get("warmup_iterations", 50)),
        iterations=int(profile_cfg.get("measured_iterations", 200)),
        use_fp16=profile_cfg.get("precision", "fp16") == "fp16",
    )
    report["gpu_name"] = gpu_name
    report["gpu_uuid"] = gpu_identity["uuid"]
    report["transfer_time_included"] = False
    return report


def _write_per_image(run_dir: Path, rows: list[dict]) -> tuple[Path, Path] | None:
    if not rows:
        return None
    frame = aggregate.per_image_frame(rows)
    parquet = run_dir / "per_image.parquet"
    csv = run_dir / "per_image.csv"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return parquet, csv


def _metric_rows(
    summary: dict[str, Any],
    *,
    run_id: str,
    model_name: str,
    backbone: str,
    variant: str,
    profile: str,
    seed: int,
    git_sha: str,
    resolved_config_hash: str,
    parameter_counts: tuple[int, int],
) -> list[dict]:
    validation = summary.get("validation")
    if not validation:
        return []
    total_parameters, trainable_parameters = parameter_counts
    common = {
        "run_id": run_id,
        "model": model_name,
        "backbone": backbone,
        "variant": variant,
        "stratum": "validation",
        "ci_low": None,
        "ci_high": None,
        "status": "ok",
        "profile": profile,
        "seed": seed,
        "git_sha": git_sha,
        "config_hash": resolved_config_hash,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    rows = [
        {**common, "scope": "ALL", "metric": "miou", "value": validation["miou"]},
        {
            **common,
            "scope": "ALL",
            "metric": "pixel_acc",
            "value": validation["pixel_accuracy"],
        },
        {
            **common,
            "scope": "ALL",
            "metric": "boundary_f1",
            "value": validation["boundary_f1"],
        },
        {
            **common,
            "scope": "big_rock",
            "metric": "small_component_recall",
            "value": validation["small_big_rock_component_recall"],
        },
        {
            **common,
            "scope": "big_rock",
            "metric": "small_components_eligible",
            "value": float(validation["small_big_rock_components_eligible"]),
        },
        {
            **common,
            "scope": "big_rock",
            "metric": "small_components_recalled",
            "value": float(validation["small_big_rock_components_recalled"]),
        },
    ]
    rows.extend(
        {**common, "scope": name, "metric": "iou", "value": value}
        for name, value in validation["per_class_iou"].items()
    )
    if validation.get("cutoff_n", 0):
        rows.append(
            {
                **common,
                "scope": "ALL",
                "metric": "cutoff_mae",
                "value": validation["cutoff_mae"],
            }
        )
    routing = validation.get("routing_utilization")
    if routing:
        rows.extend(
            {
                **common,
                "scope": f"scale_{index}",
                "metric": "routing_mean_weight",
                "value": value,
            }
            for index, value in enumerate(routing["mean_weights"])
        )
        rows.append(
            {
                **common,
                "scope": "ALL",
                "metric": "routing_active_scales",
                "value": float(routing["n_active_scales"]),
            }
        )
    latency = summary.get("latency")
    if latency:
        rows.append(
            {
                **common,
                "scope": "ALL",
                "metric": "latency_median_ms",
                "value": latency["median_ms"],
            }
        )
    peak = summary.get("peak_gpu_memory_bytes")
    if peak is not None:
        rows.append(
            {
                **common,
                "scope": "ALL",
                "metric": "peak_gpu_memory_bytes",
                "value": float(peak),
            }
        )
    return rows


def _append_mpba_results(rows: list[dict]) -> None:
    if not rows:
        return
    MPBA_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = MPBA_ROOT / ".results_store.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            append_results(
                rows,
                parquet_path=MPBA_RESULTS_PARQUET,
                csv_path=MPBA_RESULTS_CSV,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _promotion_check(
    candidate: dict,
    native: dict,
    content: dict,
    thresholds: dict,
    *,
    require_cutoff: bool,
) -> dict[str, Any]:
    candidate_val = candidate["validation"]
    native_val = native["validation"]
    content_val = content["validation"]
    miou_margin = float(thresholds["miou_gain_min"])
    big_rock_margin = float(thresholds["big_rock_iou_gain_min"])
    max_drop = float(thresholds["max_other_class_drop"])
    checks: dict[str, bool] = {
        "miou_over_native": candidate_val["miou"] >= native_val["miou"] + miou_margin,
        "miou_over_content": candidate_val["miou"] >= content_val["miou"] + miou_margin,
        "big_rock_iou": candidate_val["per_class_iou"]["big_rock"]
        >= native_val["per_class_iou"]["big_rock"] + big_rock_margin,
        "other_classes": all(
            candidate_val["per_class_iou"][name] >= native_val["per_class_iou"][name] - max_drop
            for name in ("soil", "bedrock", "sand")
        ),
        "latency": bool(candidate.get("latency") and native.get("latency"))
        and candidate["latency"]["median_ms"]
        <= native["latency"]["median_ms"]
        * (1.0 + float(thresholds["max_median_latency_increase"])),
        "routing_utilization": bool(candidate_val.get("routing_utilization"))
        and candidate_val["routing_utilization"]["n_active_scales"]
        >= int(thresholds["routing_scales_required"]),
    }
    if require_cutoff:
        checks["cutoff_mae"] = candidate_val.get("cutoff_n", 0) > 0 and candidate_val[
            "cutoff_mae"
        ] <= float(thresholds["range_cutoff_mae_max"])
    return {"passed": all(checks.values()), "checks": checks}


def _maybe_write_promotion_assessment() -> Path | None:
    """Write a gate decision from one complete, protocol-compatible ten-run screen."""
    variants = set(ALLOWED_ARMS)
    backbones = {"resnet34", "mit-b0"}
    required = {(backbone, variant) for backbone in backbones for variant in variants}
    groups: dict[str, dict[tuple[str, str], tuple[str, dict]]] = {}
    for path in MPBA_ROOT.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not summary.get("run_complete") or not summary.get("screening_eligible"):
            continue
        signature = summary.get("screen_protocol_signature")
        if not isinstance(signature, str) or len(signature) != 64:
            continue
        manifest_path = path.parent / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("screen_protocol_signature") != signature or manifest.get(
            "run_id"
        ) != summary.get("run_id"):
            continue
        key = (summary.get("backbone"), summary.get("variant"))
        if key[0] not in backbones or key[1] not in variants:
            continue
        timestamp = str(summary.get("timestamp_utc", ""))
        latest = groups.setdefault(signature, {})
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, summary)
    complete_groups = {
        signature: latest for signature, latest in groups.items() if required <= set(latest)
    }
    if not complete_groups:
        return None
    signature, latest = max(
        complete_groups.items(),
        key=lambda item: max(timestamp for timestamp, _summary in item[1].values()),
    )
    summaries = {key: value[1] for key, value in latest.items()}
    thresholds = summaries[("resnet34", "native")]["promotion_thresholds"]
    candidates: dict[str, Any] = {}
    for variant in ("raw_y", "range_cutoff"):
        by_backbone = {}
        for backbone in sorted(backbones):
            by_backbone[backbone] = _promotion_check(
                summaries[(backbone, variant)],
                summaries[(backbone, "native")],
                summaries[(backbone, "content")],
                thresholds,
                require_cutoff=variant == "range_cutoff",
            )
        mean_gain = float(
            np.mean(
                [
                    summaries[(backbone, variant)]["validation"]["miou"]
                    - summaries[(backbone, "native")]["validation"]["miou"]
                    for backbone in sorted(backbones)
                ]
            )
        )
        candidates[variant] = {
            "passed": all(report["passed"] for report in by_backbone.values()),
            "mean_miou_gain_over_native": mean_gain,
            "by_backbone": by_backbone,
        }
    passing = [name for name, report in candidates.items() if report["passed"]]
    selected = None
    if len(passing) == 1:
        selected = passing[0]
    elif len(passing) == 2:
        difference = abs(
            candidates["raw_y"]["mean_miou_gain_over_native"]
            - candidates["range_cutoff"]["mean_miou_gain_over_native"]
        )
        if difference <= float(thresholds["raw_y_tie_margin"]):
            selected = "raw_y"
        else:
            selected = max(passing, key=lambda name: candidates[name]["mean_miou_gain_over_native"])
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "promoted" if selected else "perspective_hypothesis_unsupported",
        "selected_variant": selected,
        "candidates": candidates,
        "thresholds": thresholds,
        "screen_protocol_signature": signature,
        "screen_protocol": summaries[("resnet34", "native")]["screen_protocol"],
        "run_ids": {
            f"{backbone}/{variant}": summaries[(backbone, variant)]["run_id"]
            for backbone, variant in sorted(required)
        },
        "seed": 1414,
        "gold_evaluated": False,
    }
    destination = MPBA_ROOT / "promotion.json"
    _atomic_json(destination, report)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        args.config,
        args.override,
        base_paths=[
            str(REPO_ROOT / "configs" / "data.yaml"),
            str(REPO_ROOT / "configs" / "mpba" / "base.yaml"),
        ],
    )
    model_cfg, mpba_cfg, train_cfg, variant = _validate_arm(cfg, fast_dev_run=args.fast_dev_run)
    protocol_authorization = _authorize_gold(cfg) if args.eval_split == "gold" else None
    seed = int(cfg["data"]["seed"])
    resolved_hash = config_hash(cfg)
    timestamp = datetime.now(UTC)
    run_id = (
        f"{model_cfg['name']}__{model_cfg['results_backbone']}__{variant}"
        f"__validation__seed{seed}__{resolved_hash[:8]}"
        f"__{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    run_dir = _guarded_run_dir(cfg, args.out, run_id)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_dir": str(run_dir),
                    "evaluation_split": args.eval_split,
                    "model": model_cfg["name"],
                    "backbone": model_cfg["results_backbone"],
                    "variant": variant,
                    "gold_authorization": protocol_authorization,
                },
                indent=2,
            )
        )
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )
    log.info("MPBA run_id=%s output=%s", run_id, run_dir)

    git_sha = _git_sha()
    git_dirty_at_start = _git_dirty()
    runtime_code_sha256 = _runtime_code_fingerprint()

    set_seed(seed)
    capabilities = detect()
    device = "cuda" if capabilities.cuda else "cpu"
    data_cfg = cfg["data"]
    data_root = Path(data_cfg["root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    index = build_index(
        data_root,
        rover=data_cfg["rover"],
        camera=data_cfg["camera"],
        test_gold_dir=data_cfg["test_gold_dir"],
    )
    expected_train = int(data_cfg.get("expected_train_n", 16064))
    if len(index["train"]) != expected_train:
        raise AssertionError(f"MSL train count {len(index['train'])} != expected {expected_train}")
    splits = make_splits(
        index["train"],
        val_frac=float(data_cfg["val_frac"]),
        seed=int(data_cfg["split_seed"]),
    )
    splits, cohort_exclusions = apply_matched_range_cohort(splits)
    matched_split_fingerprint = index_fingerprint(splits, data_root)
    full_cohort_counts = {name: len(records) for name, records in splits.items()}
    configured_max_train = data_cfg.get("max_train_images")
    effective_max_train = configured_max_train
    if args.fast_dev_run and effective_max_train is None:
        effective_max_train = int(train_cfg["batch_size"])
    if effective_max_train:
        splits["train"] = splits["train"][: int(effective_max_train)]
        splits["val"] = splits["val"][: min(int(effective_max_train), len(splits["val"]))]
    use_range_cutoff = mpba_cfg["coordinate_mode"] == "range_cutoff"
    range_dir = mpba_cfg.get("range_mask_dir")
    if range_dir and not Path(range_dir).is_absolute():
        range_dir = str(REPO_ROOT / range_dir)
    # Resolve the privileged products for every arm.  This guarantees that native, static,
    # content, raw-y, and cutoff runs use one identical range-complete training cohort.
    splits = {
        name: attach_range_masks(records, range_dir=range_dir, strict=True)
        for name, records in splits.items()
    }
    weights = _class_weights(splits["train"], cfg.get("class_weights", {}))
    model = _build_screening_model(model_cfg, mpba_cfg)
    module = MPBALitModule(
        model,
        num_classes=NUM_CLASSES,
        coordinate_mode=mpba_cfg["coordinate_mode"],
        class_weights=weights,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        dice_weight=float(train_cfg["dice_weight"]),
        cutoff_loss_weight=float(mpba_cfg["cutoff_loss_weight"]),
        ignore_index=int(train_cfg["ignore_index"]),
        max_epochs=int(train_cfg["max_epochs"]),
    )
    parameter_counts = _parameter_counts(module.model)
    data_module = MPBADataModule(
        splits["train"],
        splits["val"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        size=int(data_cfg["size"]),
        aug=cfg.get("aug", {}),
        seed=seed,
        use_range_cutoff=use_range_cutoff,
    )
    best_val_miou = None
    training_metrics_path = None
    stages = ["index_train", "split_train_validation", "class_weights", "build_model"]
    checkpoint_path = Path(args.eval_only).resolve() if args.eval_only else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if checkpoint_path:
        _load_checkpoint(module, checkpoint_path, device)
        data_module.setup("validate")
        stages.append("load_checkpoint")
    else:
        checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best",
            monitor="val_miou",
            mode="max",
            save_top_k=1,
        )
        csv_logger = CSVLogger(save_dir=str(run_dir), name="training_logs", version="")
        trainer = L.Trainer(
            max_epochs=int(train_cfg["max_epochs"]),
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if device == "cuda" else "32-true",
            gradient_clip_val=float(train_cfg["grad_clip"]),
            callbacks=[
                checkpoint,
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(
                    monitor="val_miou",
                    mode="max",
                    patience=int(train_cfg["early_stop_patience"]),
                    min_delta=0.001,
                ),
            ],
            logger=csv_logger,
            enable_progress_bar=False,
            default_root_dir=run_dir,
            fast_dev_run=args.fast_dev_run,
        )
        trainer.fit(module, data_module)
        stages.append("train_fast_dev" if args.fast_dev_run else "train")
        if checkpoint.best_model_score is not None:
            best_val_miou = float(checkpoint.best_model_score)
        if checkpoint.best_model_path:
            checkpoint_path = Path(checkpoint.best_model_path)
            _load_checkpoint(module, checkpoint_path, device)
            stages.append("load_best_checkpoint")
        metrics_source = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_source.is_file():
            training_metrics_path = run_dir / "training_metrics.csv"
            shutil.copy2(metrics_source, training_metrics_path)
            stages.append("training_metrics")

    validation_report = None
    per_image_rows: list[dict] = []
    if args.eval_split != "none":
        if args.eval_split == "validation":
            evaluation_records = splits["val"]
            split_name = "validation"
        else:
            evaluation_records = index["test"]
            split_name = "test_msl"
        if args.fast_dev_run:
            evaluation_records = evaluation_records[
                : max(1, min(int(train_cfg["batch_size"]), len(evaluation_records)))
            ]
        evaluation_loader = _evaluation_loader(
            evaluation_records,
            size=int(data_cfg["size"]),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg.get("num_workers", 0)),
            include_cutoff_targets=(use_range_cutoff and args.eval_split == "validation"),
        )
        validation_report, per_image_rows = _evaluate(
            module,
            evaluation_loader,
            run_dir=run_dir,
            run_id=run_id,
            split=split_name,
            device=device,
            boundary_tolerance_px=int(cfg["evaluation"]["boundary_tolerance_px"]),
        )
        stages.append(f"eval_{split_name}")
    _write_per_image(run_dir, per_image_rows)
    latency = (
        _profile_latency(module, cfg, device=device)
        if args.eval_split != "none" and not args.fast_dev_run
        else None
    )
    if latency:
        stages.append("profile_latency")
    peak_gpu_memory = torch.cuda.max_memory_allocated() if device == "cuda" else None
    code_stable = bool(
        _git_sha() == git_sha
        and _git_dirty() == git_dirty_at_start
        and _runtime_code_fingerprint() == runtime_code_sha256
    )
    gpu_identity = {"uuid": latency["gpu_uuid"], "name": latency["gpu_name"]} if latency else None
    screen_protocol, screen_protocol_signature = _screen_protocol(
        cfg,
        matched_split_fingerprint=matched_split_fingerprint,
        git_sha=git_sha,
        runtime_code_sha256=runtime_code_sha256,
        gpu_identity=gpu_identity,
    )
    full_screen = bool(
        configured_max_train is None
        and not args.fast_dev_run
        and args.eval_only is None
        and args.eval_split == "validation"
    )
    eligible_candidate = bool(
        full_screen
        and len(splits["train"]) == full_cohort_counts["train"]
        and len(splits["val"]) == full_cohort_counts["val"]
        and not git_dirty_at_start
        and code_stable
        and latency
        and latency.get("gpu_uuid")
        and "v100" in str(latency.get("gpu_name", "")).lower()
        and validation_report
        and validation_report.get("n_images") == len(splits["val"])
    )
    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "model": model_cfg["name"],
        "backbone": model_cfg["results_backbone"],
        "variant": variant,
        "seed": seed,
        "evaluation_split": args.eval_split,
        "validation": validation_report,
        "parameters": {
            "total": parameter_counts[0],
            "trainable": parameter_counts[1],
        },
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "latency": latency,
        "best_val_miou": best_val_miou,
        "screening_eligible": False,
        "eligible_candidate": eligible_candidate,
        "run_complete": False,
        "fast_dev_run": bool(args.fast_dev_run),
        "git_sha": git_sha,
        "git_dirty_at_start": git_dirty_at_start,
        "code_stable": code_stable,
        "runtime_code_sha256": runtime_code_sha256,
        "screen_protocol_signature": screen_protocol_signature,
        "screen_protocol": screen_protocol,
        "matched_cohort_exclusions": cohort_exclusions,
        "matched_cohort_counts": full_cohort_counts,
        "promotion_thresholds": cfg["promotion"],
        "gold_evaluated": args.eval_split == "gold",
    }
    _atomic_json(run_dir / "summary.json", summary)
    result_rows = _metric_rows(
        summary,
        run_id=run_id,
        model_name=model_cfg["name"],
        backbone=model_cfg["results_backbone"],
        variant=variant,
        profile=capabilities.profile,
        seed=seed,
        git_sha=git_sha,
        resolved_config_hash=resolved_hash,
        parameter_counts=parameter_counts,
    )
    if not args.fast_dev_run:
        _append_mpba_results(result_rows)
        stages.append("mpba_results_store")
    train_only_fingerprint = index_fingerprint({"train": index["train"]}, data_root)
    manifest_path = write_manifest(
        run_dir,
        cfg,
        seed,
        run_id=run_id,
        profile=capabilities.profile,
        model=model_cfg["name"],
        backbone=model_cfg["results_backbone"],
        variant=variant,
        dataset=f"ai4mars_{data_cfg['rover']}_{data_cfg['camera']}_train_validation",
        data_hashes={
            "train_index_sha256": train_only_fingerprint,
            "matched_train_validation_sha256": matched_split_fingerprint,
        },
        stages_completed=stages,
        evaluation_split=args.eval_split,
        gold_authorization=protocol_authorization,
        mpba_isolated_results=True,
        canonical_results_written=False,
        total_parameters=parameter_counts[0],
        trainable_parameters=parameter_counts[1],
        class_weights=weights,
        data_counts={
            "n_train": len(splits["train"]),
            "n_validation": len(splits["val"]),
        },
        matched_cohort_exclusions=cohort_exclusions,
        matched_cohort_excluded_n=len(cohort_exclusions),
        matched_cohort_full_counts=full_cohort_counts,
        best_checkpoint=str(checkpoint_path) if checkpoint_path else None,
        best_val_miou=best_val_miou,
        training_metrics_path=(
            str(training_metrics_path.relative_to(run_dir)) if training_metrics_path else None
        ),
        summary_path="summary.json",
        peak_gpu_memory_bytes=peak_gpu_memory,
        runtime_code_sha256=runtime_code_sha256,
        screen_protocol_signature=screen_protocol_signature,
        screen_protocol=screen_protocol,
        code_stable=code_stable,
        screening_eligible=eligible_candidate,
    )
    mirror = MPBA_ROOT / "manifests" / run_id
    mirror.mkdir(parents=True, exist_ok=False)
    for source in (
        manifest_path,
        run_dir / "summary.json",
        run_dir / "resolved_config.yaml",
        run_dir / "per_image.parquet",
        run_dir / "per_image.csv",
        run_dir / "training_metrics.csv",
    ):
        if source.is_file():
            shutil.copy2(source, mirror / source.name)
    # Make the mirrored summary final first and the run summary final last.  Promotion scans the
    # latter, so any crash during results/manifest/mirror finalization leaves the run ineligible.
    summary["screening_eligible"] = eligible_candidate
    summary["run_complete"] = True
    _atomic_json(mirror / "summary.json", summary)
    _atomic_json(run_dir / "summary.json", summary)
    promotion = _maybe_write_promotion_assessment()
    if promotion:
        log.info("MPBA promotion decision written to %s", promotion)
    log.info("MPBA run complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
