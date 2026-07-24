"""Train/evaluate one experiment arm end-to-end (MS3; DEVPLAN section 8 CLI).

    python scripts/run_experiment.py --config configs/models/<name>.yaml \
        [--override k=v ...] [--out DIR] [--eval-only CKPT --h4]

Wiring (frozen protocol, DEVPLAN 5.4): seed 1414; camera-aware build_index honoring the pinned
min3 gold dir; by-image splits; ``data.max_train_images`` truncates train (and caps val to the
same N — CPU-smoke practicality; null = full data, untouched) AFTER make_splits; class weights
median/count clipped to [0.5, 10] on the train split; Lightning Trainer with EarlyStopping /
ModelCheckpoint(val_miou, max) / grad-clip 1.0; the BEST checkpoint is evaluated; predictions
contract per 7.4 (preds PNGs + per_image.parquet/csv); manifest + results rows + the committed
mirror under experiments/manifests/<run_id>/.

``--eval-only CKPT --h4`` re-evaluates a trained checkpoint on the MSL gold test (stratum
in_rover) AND the MER gold test (stratum cross_rover) without retraining (5.7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from aresseg.data.ai4mars import IGNORE_INDEX, NUM_CLASSES, build_index  # noqa: E402
from aresseg.data.dataset import SegDataset, class_pixel_counts, make_splits  # noqa: E402
from aresseg.data.preflight import index_fingerprint  # noqa: E402
from aresseg.data.transforms import eval_transform  # noqa: E402
from aresseg.eval import aggregate, metrics  # noqa: E402
from aresseg.utils.capabilities import detect  # noqa: E402
from aresseg.utils.config import load_config  # noqa: E402
from aresseg.utils.logging import get_logger  # noqa: E402
from aresseg.utils.manifest import _git_sha, config_hash, write_manifest  # noqa: E402
from aresseg.utils.results import append_results  # noqa: E402
from aresseg.utils.seed import set_seed  # noqa: E402

log = get_logger("aresseg.run")

FOUNDATION_VARIANTS = {"dinov3_sat": "finetuned", "sam": "region_oracle_upper_bound"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="run dir (default: experiments/<run_id>)")
    ap.add_argument(
        "--eval-only",
        default=None,
        metavar="CKPT",
        help="skip training; evaluate this Lightning checkpoint",
    )
    ap.add_argument(
        "--h4",
        action="store_true",
        help="also evaluate the MER cross-rover gold test (5.7)",
    )
    ap.add_argument(
        "--eval-split",
        choices=("gold", "validation", "none"),
        default="gold",
        help="evaluation target; validation/none are safe for smoke runs and never score gold",
    )
    ap.add_argument("--check-complete", action="store_true", help="find a resumable matching run")
    return ap.parse_args(argv)


def _variant(model_cfg: dict) -> str:
    name = model_cfg["name"]
    if name == "majority":
        return "constant"
    if name in FOUNDATION_VARIANTS:
        return FOUNDATION_VARIANTS[name]
    return "pretrained" if model_cfg.get("pretrained") else "scratch"


def _weights_source(model_cfg: dict) -> str:
    if _variant(model_cfg) == "scratch":
        return "scratch"
    cache_dirs = [Path.home() / ".cache" / "torch" / "hub" / "checkpoints"]
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dirs.append(Path(HF_HUB_CACHE))
    except Exception:
        pass
    return "cache" if any(d.is_dir() and any(d.iterdir()) for d in cache_dirs) else "network"


def _class_weights(train_records: list[dict], cw_cfg: dict) -> list[float]:
    counts = class_pixel_counts(
        train_records, num_classes=NUM_CLASSES, max_images=cw_cfg.get("max_images")
    ).astype("float64")
    counts = np.maximum(counts, 1.0)  # a class absent from the (smoke-sized) scan stays finite
    lo, hi = cw_cfg.get("clip", [0.5, 10.0])
    return [float(w) for w in np.clip(np.median(counts) / counts, lo, hi)]


def _assert_names(records: list[dict], what: str) -> None:
    names = [r["name"] for r in records]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"{what}: names must be non-empty and unique (4.3)")


def _resolved_git_sha() -> str:
    sha = _git_sha()
    if sha != "UNKNOWN":
        return sha
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _runtime_code_fingerprint() -> str:
    digest = hashlib.sha256()
    files = sorted((REPO_ROOT / "src").rglob("*.py"))
    files += sorted((REPO_ROOT / "scripts").glob("*.py"))
    files += sorted((REPO_ROOT / "scripts").glob("*.sh"))
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


PROTOCOL_V3_TRAINING_GIT_SHAS = frozenset(
    {
        "c188b320d700e01c8ffb37330e30f188862ad995",
        "c2c2860f40626413eb95dd9bfec3d492fdde9035",
        "3e52372ce7ea6f923dddec95338384a6dd3693bd",
    }
)
PROTOCOL_V3_TRAINING_CODE_FINGERPRINT = (
    "bebd8b2dae8fb62a087ded6f5334dbf19bfde1a157bad743b17471ba200325d3"
)


def _parameter_counts(model) -> tuple[int, int]:
    params = list(model.parameters())
    return sum(item.numel() for item in params), sum(
        item.numel() for item in params if item.requires_grad
    )


def _add_parameter_context(rows: list[dict], counts: tuple[int, int]) -> list[dict]:
    total, trainable = counts
    for row in rows:
        row["total_parameters"] = total
        row["trainable_parameters"] = trainable
    return rows


def _atomic_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _matching_complete_run(
    cfg: dict,
    *,
    profile: str,
    model_name: str,
    git_sha: str,
    code_fingerprint: str,
    h4: bool = False,
    source_checkpoint: str | None = None,
    allow_protocol_v3_training: bool = False,
) -> Path | None:
    """Return the newest fully materialized matching run, never a marker-file guess."""
    import pandas as pd

    chash = config_hash(cfg)
    try:
        store = pd.read_parquet(REPO_ROOT / "experiments" / "results_store.parquet")
    except Exception:
        csv = REPO_ROOT / "experiments" / "results_store.csv"
        store = pd.read_csv(csv) if csv.is_file() else pd.DataFrame()
    candidates: list[tuple[str, Path]] = []
    for manifest_path in (REPO_ROOT / "experiments").glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = manifest_path.parent
        provenance_matches = (
            manifest.get("git_sha") == git_sha
            and manifest.get("code_fingerprint") == code_fingerprint
        )
        if allow_protocol_v3_training:
            provenance_matches = provenance_matches or (
                manifest.get("git_sha") in PROTOCOL_V3_TRAINING_GIT_SHAS
                and manifest.get("code_fingerprint") == PROTOCOL_V3_TRAINING_CODE_FINGERPRINT
            )
        if (
            manifest.get("config_hash") != chash
            or not provenance_matches
            or manifest.get("profile") != profile
            or manifest.get("model") != model_name
            or manifest.get("evaluation_split") != "gold"
            or "eval_test_msl" not in (manifest.get("stages_completed") or [])
            or not (run_dir / "per_image.parquet").is_file()
            or not (run_dir / "per_image.csv").is_file()
            or h4 != ("eval_test_mer" in (manifest.get("stages_completed") or []))
        ):
            continue
        learned = model_name not in {"majority", "sam"}
        if learned and h4:
            if (
                "load_checkpoint" not in (manifest.get("stages_completed") or [])
                or manifest.get("source_checkpoint") != source_checkpoint
            ):
                continue
        elif learned and (
            "train" not in (manifest.get("stages_completed") or [])
            or not (run_dir / "best.ckpt").is_file()
            or not (run_dir / "training_metrics.csv").is_file()
        ):
            continue
        if (
            store.empty
            or not (
                (store.get("run_id") == manifest.get("run_id"))
                & (store.get("config_hash") == chash)
                & (store.get("status") == "ok")
            ).any()
        ):
            continue
        candidates.append((manifest.get("timestamp_utc", ""), run_dir))
    if not candidates:
        return None
    newest_timestamp = max(timestamp for timestamp, _ in candidates)
    newest = [path for timestamp, path in candidates if timestamp == newest_timestamp]
    if len(newest) != 1:
        raise RuntimeError(f"exact completed-run timestamp tie requires resolution: {newest}")
    return newest[0]


def evaluate_records(
    model,
    records: list[dict],
    *,
    size: int,
    run_dir: Path,
    split: str,
    run_id: str,
    batch_size: int = 8,
    device: str = "cpu",
    num_workers: int = 0,
) -> list[dict]:
    """Best-ckpt evaluation: write preds/<split>/<name>.png + return per-image rows (7.4)."""
    import cv2
    import torch
    from torch.utils.data import DataLoader

    ds = SegDataset(records, eval_transform(size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_dir = run_dir / "preds" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
            masks = batch["mask"].numpy()
            for i, name in enumerate(batch["name"]):
                pred, gt = preds[i], masks[i]
                pred = pred.copy()
                pred[gt == IGNORE_INDEX] = IGNORE_INDEX  # ignore copied from GT (7.4)
                cv2.imwrite(str(pred_dir / f"{name}.png"), pred)
                counts = metrics.per_image_counts(pred, gt)
                bf1 = metrics.boundary_f1(pred, gt)
                rows.extend(aggregate.image_rows(run_id, str(name), split, counts, bf1))
    return rows


def sam_region_oracle(
    sam_model, records: list[dict], *, size: int, run_dir: Path, split: str, run_id: str
) -> list[dict]:
    """SAM zero-shot region-oracle scoring (frozen in PREREG): each proposal takes the majority
    valid-GT class of its pixels; uncovered pixels default to class 0 (soil, the majority class)."""
    import cv2

    pred_dir = run_dir / "preds" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rec in records:
        img = cv2.imread(rec["image"], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(rec["label"], cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        pred = np.zeros((size, size), dtype=np.uint8)  # uncovered -> class 0 (documented)
        for m in sam_model.generate(np.repeat(img[:, :, None], 3, axis=2)):
            seg = m["segmentation"]
            valid = seg & (gt != IGNORE_INDEX)
            if not valid.any():
                continue
            vals, cnts = np.unique(gt[valid], return_counts=True)
            pred[seg] = np.uint8(vals[np.argmax(cnts)])
        pred[gt == IGNORE_INDEX] = IGNORE_INDEX
        cv2.imwrite(str(pred_dir / f"{rec['name']}.png"), pred)
        counts = metrics.per_image_counts(pred, gt)
        rows.extend(
            aggregate.image_rows(run_id, rec["name"], split, counts, metrics.boundary_f1(pred, gt))
        )
    return rows


def _finalize(run_dir: Path, per_image_rows: list[dict], store_rows: list[dict]) -> None:
    """Write per_image parquet/csv + append results + mirror into experiments/manifests/."""
    if per_image_rows:
        df = aggregate.per_image_frame(per_image_rows)
        df.to_parquet(run_dir / "per_image.parquet", index=False)
        df.to_csv(run_dir / "per_image.csv", index=False)
    if store_rows:
        append_results(store_rows)
    mirror = REPO_ROOT / "experiments" / "manifests" / run_dir.name
    mirror.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.json", "per_image.parquet", "per_image.csv", "training_metrics.csv"):
        src = run_dir / f
        if src.is_file():
            _atomic_copy(src, mirror / f)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.h4 and args.eval_split != "gold":
        raise ValueError("--h4 requires --eval-split gold")
    cfg = load_config(args.config, args.override, base_paths=[str(REPO_ROOT / "configs/data.yaml")])
    seed = int(cfg["data"].get("seed", 1414))
    profile = detect().profile
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    model_name = mcfg["name"]
    variant = _variant(mcfg)
    results_backbone = mcfg.get("results_backbone") or "none"
    chash = config_hash(cfg)
    git_sha = _resolved_git_sha()
    code_fingerprint = _runtime_code_fingerprint()
    if args.check_complete:
        completed = _matching_complete_run(
            cfg,
            profile=profile,
            model_name=model_name,
            git_sha=git_sha,
            code_fingerprint=code_fingerprint,
            h4=args.h4,
            source_checkpoint=args.eval_only,
        )
        if completed is None:
            return 1
        print(completed)
        return 0

    set_seed(seed)
    split_scope = (
        "cross_rover" if args.h4 else ("in_rover" if args.eval_split == "gold" else args.eval_split)
    )
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    generated_id = (
        f"{model_name}__{variant}__{results_backbone}__{split_scope}"
        f"__seed{seed}__{profile}__{chash[:8]}__git{git_sha[:8]}"
        f"__code{code_fingerprint[:8]}__{attempt}"
    )
    run_dir = Path(args.out) if args.out else REPO_ROOT / "experiments" / generated_id
    run_id = run_dir.name if args.out else generated_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log.info("run_id=%s profile=%s", run_id, profile)
    base_row = dict(
        run_id=run_id,
        model=model_name,
        backbone=results_backbone,
        variant=variant,
        profile=profile,
        seed=seed,
        git_sha=git_sha,
        config_hash=chash,
    )

    # ---------- data ----------
    root = REPO_ROOT / dcfg["root"]
    index = build_index(root, dcfg["rover"], dcfg["camera"], test_gold_dir=dcfg["test_gold_dir"])
    expected_train = int(dcfg.get("expected_train_n", 16064))
    if len(index["train"]) != expected_train:
        raise AssertionError(f"MSL train count {len(index['train'])} != expected {expected_train}")
    if len(index["test"]) != int(dcfg["expected_test_n"]):
        raise AssertionError(
            f"MSL test count {len(index['test'])} != expected {dcfg['expected_test_n']}"
        )
    msl_index_sha256 = index_fingerprint(index, root)
    splits = make_splits(
        index["train"], val_frac=float(dcfg["val_frac"]), seed=int(dcfg["split_seed"])
    )
    max_train = dcfg.get("max_train_images")
    val_truncated_to = None
    if max_train:  # smoke cap after the stable split; the full source count was checked above
        splits["train"] = splits["train"][: int(max_train)]
        val_truncated_to = min(int(max_train), len(splits["val"]))
        splits["val"] = splits["val"][:val_truncated_to]
    for what, recs in (("train", splits["train"]), ("val", splits["val"]), ("test", index["test"])):
        _assert_names(recs, what)
    if args.eval_split == "gold":
        primary_records, primary_split = index["test"], "test_msl"
    elif args.eval_split == "validation":
        primary_records, primary_split = splits["val"], "validation"
    else:
        primary_records, primary_split = [], None

    # ---------- gated foundation: skip-and-log path ----------
    stages: list[str] = ["index", "splits"]
    skipped: list[str] = []

    def _skip(reason_ctx: str | list[str]) -> int:
        skipped.append(model_name)
        skip_reasons = [reason_ctx] if isinstance(reason_ctx, str) else list(reason_ctx)
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            external_revision=mcfg.get("revision"),
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes={"msl_index_sha256": msl_index_sha256},
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source=None,
            val_truncated_to=val_truncated_to,
            data_counts={
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(index["test"]),
            },
            data_fingerprints={"msl_index_sha256": msl_index_sha256},
            code_fingerprint=code_fingerprint,
            evaluation_split=args.eval_split,
            skip_reasons=skip_reasons,
            git_sha=git_sha,
            total_parameters=None,
            trainable_parameters=None,
        )
        _finalize(
            run_dir,
            [],
            [
                {
                    **base_row,
                    "scope": "ALL",
                    "stratum": "all",
                    "metric": "miou",
                    "value": None,
                    "status": "skipped",
                    "total_parameters": None,
                    "trainable_parameters": None,
                }
            ],
        )
        log.warning(
            "arm %s skipped (%s); status=skipped row appended",
            model_name,
            "; ".join(skip_reasons),
        )
        return 0

    if model_name == "sam":
        from aresseg.models.foundation import (
            build_foundation,
            gating_reasons,
            last_unavailable_reason,
        )

        reasons = gating_reasons("sam", sam_checkpoint=mcfg.get("sam_checkpoint"))
        if reasons:
            return _skip(reasons)
        sam_model = build_foundation("sam", sam_checkpoint=mcfg.get("sam_checkpoint"))
        if sam_model is None:
            return _skip(last_unavailable_reason("sam") or "SAM load unavailable")
        parameter_counts = _parameter_counts(sam_model)
        rows: list[dict] = []
        store: list[dict] = []
        if primary_records:
            rows = sam_region_oracle(
                sam_model,
                primary_records,
                size=int(dcfg["size"]),
                run_dir=run_dir,
                split=primary_split,
                run_id=run_id,
            )
            stages.append(f"eval_{primary_split}")
            if args.eval_split == "gold":
                store = aggregate.store_rows(
                    aggregate.per_image_frame(rows),
                    split="test_msl",
                    stratum="all",
                    status="ok",
                    **base_row,
                )
                _add_parameter_context(store, parameter_counts)
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            external_revision=mcfg.get("revision"),
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes={"msl_index_sha256": msl_index_sha256},
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source="cache",
            val_truncated_to=None,
            data_counts={"n_test": len(index["test"])},
            data_fingerprints={"msl_index_sha256": msl_index_sha256},
            code_fingerprint=code_fingerprint,
            evaluation_split=args.eval_split,
            skip_reasons=[],
            git_sha=git_sha,
            total_parameters=parameter_counts[0],
            trainable_parameters=parameter_counts[1],
            training_metrics_path=None,
        )
        _finalize(run_dir, rows, store)
        return 0

    # ---------- trainable arms (baseline / unet / deeplabv3plus / segformer / dinov3_sat) ----------
    if model_name == "majority":
        from aresseg.models.zoo import build_model

        majority_counts = class_pixel_counts(index["train"], num_classes=NUM_CLASSES)
        majority_class = int(np.argmax(majority_counts))
        stages.append("majority_class_evidence")
        majority_model = build_model(
            "majority",
            num_classes=NUM_CLASSES,
            pretrained=False,
            predicted_class=majority_class,
        )
        parameter_counts = _parameter_counts(majority_model)
        device = "cuda" if profile == "gpu_full" else "cpu"
        per_image_rows: list[dict] = []
        store_rows: list[dict] = []
        if primary_records:
            per_image_rows += evaluate_records(
                majority_model,
                primary_records,
                size=int(dcfg["size"]),
                run_dir=run_dir,
                split=primary_split,
                run_id=run_id,
                batch_size=int(tcfg["batch_size"]),
                device=device,
                num_workers=int(tcfg.get("num_workers", 0)),
            )
            stages.append(f"eval_{primary_split}")
        mer_index_sha256 = None
        if args.h4:
            mer_cfg = cfg["mer"]
            mer = build_index(root, "mer", test_gold_dir=mer_cfg["test_gold_dir"])
            if len(mer["test"]) != int(mer_cfg["expected_test_n"]):
                raise AssertionError(
                    f"MER test count {len(mer['test'])} != expected {mer_cfg['expected_test_n']}"
                )
            _assert_names(mer["test"], "mer_test")
            mer_index_sha256 = index_fingerprint(mer, root)
            per_image_rows += evaluate_records(
                majority_model,
                mer["test"],
                size=int(dcfg["size"]),
                run_dir=run_dir,
                split="test_mer",
                run_id=run_id,
                batch_size=int(tcfg["batch_size"]),
                device=device,
                num_workers=int(tcfg.get("num_workers", 0)),
            )
            stages.append("eval_test_mer")
        if args.eval_split == "gold":
            frame = aggregate.per_image_frame(per_image_rows)
            store_rows += aggregate.store_rows(
                frame,
                split="test_msl",
                stratum="in_rover" if args.h4 else "all",
                status="ok",
                **base_row,
            )
            if args.h4:
                store_rows += aggregate.store_rows(
                    frame,
                    split="test_mer",
                    stratum="cross_rover",
                    status="ok",
                    **base_row,
                )
            _add_parameter_context(store_rows, parameter_counts)
        fingerprints = {"msl_index_sha256": msl_index_sha256}
        if mer_index_sha256:
            fingerprints["mer_index_sha256"] = mer_index_sha256
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            external_revision=mcfg.get("revision"),
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes=fingerprints,
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source="constant",
            val_truncated_to=val_truncated_to,
            data_counts={
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(index["test"]),
            },
            data_fingerprints=fingerprints,
            code_fingerprint=code_fingerprint,
            evaluation_split=args.eval_split,
            skip_reasons=[],
            git_sha=git_sha,
            total_parameters=parameter_counts[0],
            trainable_parameters=parameter_counts[1],
            training_metrics_path=None,
            majority_class=majority_class,
            majority_class_pixel_counts=majority_counts.tolist(),
        )
        _finalize(run_dir, per_image_rows, store_rows)
        return 0

    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from aresseg.train.lit import SegDataModule, SegLitModule

    weights_source = _weights_source(mcfg)
    class_weights = _class_weights(splits["train"], cfg.get("class_weights", {}))
    stages.append("class_weights")
    log.info("class_weights=%s weights_source=%s", class_weights, weights_source)

    lit = SegLitModule(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        backbone=mcfg.get("backbone"),
        pretrained=bool(mcfg.get("pretrained")),
        revision=mcfg.get("revision"),
        class_weights=class_weights,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        dice_weight=float(tcfg["dice_weight"]),
        ignore_index=int(tcfg["ignore_index"]),
        max_epochs=int(tcfg["max_epochs"]),
    )
    if lit.model is None:  # gated dinov3_sat on CPU / missing token -> skip-and-log (5.4)
        if model_name == "dinov3_sat":
            from aresseg.models.foundation import gating_reasons, last_unavailable_reason

            reasons = gating_reasons("dinov3_sat")
            return _skip(
                reasons or [last_unavailable_reason("dinov3_sat") or "DINOv3 load unavailable"]
            )
        return _skip("model construction returned no model")

    parameter_counts = _parameter_counts(lit.model)
    best_val_miou = None
    training_metrics_path = None
    device = "cuda" if profile == "gpu_full" else "cpu"
    if args.eval_only:
        state = torch.load(args.eval_only, map_location=device, weights_only=False)
        lit.load_state_dict(state["state_dict"])
        # recover the training run's val score from the ModelCheckpoint callback state so the
        # 5.7 "highest val_miou" subject rule stays operative on eval-only (--h4) manifests
        for cb_state in (state.get("callbacks") or {}).values():
            if isinstance(cb_state, dict) and cb_state.get("best_model_score") is not None:
                best_val_miou = float(cb_state["best_model_score"])
                break
        stages.append("load_checkpoint")
    else:
        dm = SegDataModule(
            splits["train"],
            splits["val"],
            batch_size=int(tcfg["batch_size"]),
            num_workers=int(tcfg.get("num_workers", 0)),  # loader workers only; verdict-neutral
            size=int(dcfg["size"]),
            aug=cfg.get("aug", {}),
            seed=seed,
        )
        ckpt_cb = ModelCheckpoint(
            dirpath=run_dir, filename="best", monitor="val_miou", mode="max", save_top_k=1
        )
        csv_logger = CSVLogger(save_dir=str(run_dir), name="training_logs", version="")
        trainer = L.Trainer(
            max_epochs=int(tcfg["max_epochs"]),
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if profile == "gpu_full" else "32-true",
            gradient_clip_val=float(tcfg["grad_clip"]),
            callbacks=[
                ckpt_cb,
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(
                    monitor="val_miou",
                    mode="max",
                    patience=int(tcfg["early_stop_patience"]),
                    min_delta=0.001,
                ),
            ],
            logger=csv_logger,
            enable_progress_bar=False,
            default_root_dir=run_dir,
        )
        trainer.fit(lit, dm)
        csv_logger.save()
        metrics_source = Path(csv_logger.log_dir) / "metrics.csv"
        if not metrics_source.is_file():
            raise RuntimeError(f"Lightning CSV metrics missing after fit: {metrics_source}")
        training_metrics_path = _atomic_copy(metrics_source, run_dir / "training_metrics.csv")
        stages.extend(["train", "training_metrics"])
        if ckpt_cb.best_model_score is not None:
            best_val_miou = float(ckpt_cb.best_model_score)
        if ckpt_cb.best_model_path:  # evaluate the BEST checkpoint, not the last epoch
            state = torch.load(ckpt_cb.best_model_path, map_location=device, weights_only=False)
            lit.load_state_dict(state["state_dict"])

    per_image_rows: list[dict] = []
    if args.eval_split != "gold":
        smoke_rows: list[dict] = []
        if args.eval_split == "validation":
            smoke_rows = evaluate_records(
                lit,
                splits["val"],
                size=int(dcfg["size"]),
                run_dir=run_dir,
                split="validation",
                run_id=run_id,
                batch_size=int(tcfg["batch_size"]),
                device=device,
                num_workers=int(tcfg.get("num_workers", 0)),
            )
            stages.append("eval_validation")
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            external_revision=mcfg.get("revision"),
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes={"msl_index_sha256": msl_index_sha256},
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            class_weights=class_weights,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source=weights_source,
            best_val_miou=best_val_miou,
            val_truncated_to=val_truncated_to,
            data_counts={
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(index["test"]),
            },
            data_fingerprints={"msl_index_sha256": msl_index_sha256},
            code_fingerprint=code_fingerprint,
            evaluation_split=args.eval_split,
            skip_reasons=[],
            git_sha=git_sha,
            total_parameters=parameter_counts[0],
            trainable_parameters=parameter_counts[1],
            training_metrics_path=(
                str(training_metrics_path.relative_to(run_dir)) if training_metrics_path else None
            ),
        )
        _finalize(run_dir, smoke_rows, [])
        return 0

    store_rows: list[dict] = []
    msl_stratum = "in_rover" if args.h4 else "all"
    per_image_rows += evaluate_records(
        lit,
        index["test"],
        size=int(dcfg["size"]),
        run_dir=run_dir,
        split="test_msl",
        run_id=run_id,
        batch_size=int(tcfg["batch_size"]),
        device=device,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    stages.append("eval_test_msl")
    if args.h4:
        mer_cfg = cfg["mer"]
        mer = build_index(root, "mer", test_gold_dir=mer_cfg["test_gold_dir"])
        if len(mer["test"]) != int(mer_cfg["expected_test_n"]):
            raise AssertionError(
                f"MER test count {len(mer['test'])} != expected {mer_cfg['expected_test_n']}"
            )
        _assert_names(mer["test"], "mer_test")
        mer_index_sha256 = index_fingerprint(mer, root)
        per_image_rows += evaluate_records(
            lit,
            mer["test"],
            size=int(dcfg["size"]),
            run_dir=run_dir,
            split="test_mer",
            run_id=run_id,
            batch_size=int(tcfg["batch_size"]),
            device=device,
            num_workers=int(tcfg.get("num_workers", 0)),
        )
        stages.append("eval_test_mer")

    frame = aggregate.per_image_frame(per_image_rows)
    store_rows += aggregate.store_rows(
        frame, split="test_msl", stratum=msl_stratum, status="ok", **base_row
    )
    if args.h4:
        store_rows += aggregate.store_rows(
            frame, split="test_mer", stratum="cross_rover", status="ok", **base_row
        )

    _add_parameter_context(store_rows, parameter_counts)
    write_manifest(
        run_dir,
        cfg,
        seed,
        profile=profile,
        model=model_name,
        backbone=results_backbone,
        variant=variant,
        external_revision=mcfg.get("revision"),
        dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
        data_hashes={
            "msl_index_sha256": msl_index_sha256,
            **({"mer_index_sha256": mer_index_sha256} if args.h4 else {}),
        },
        stages_completed=stages,
        gpu_stages_skipped=skipped,
        class_weights=class_weights,
        resolved_test_gold_dir=dcfg["test_gold_dir"],
        weights_source=weights_source,
        best_val_miou=best_val_miou,
        val_truncated_to=val_truncated_to,
        data_counts={
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(index["test"]),
        },
        data_fingerprints={
            "msl_index_sha256": msl_index_sha256,
            **({"mer_index_sha256": mer_index_sha256} if args.h4 else {}),
        },
        code_fingerprint=code_fingerprint,
        evaluation_split=args.eval_split,
        skip_reasons=[],
        git_sha=git_sha,
        total_parameters=parameter_counts[0],
        trainable_parameters=parameter_counts[1],
        training_metrics_path=(
            str(training_metrics_path.relative_to(run_dir)) if training_metrics_path else None
        ),
        source_checkpoint=args.eval_only,
    )
    _finalize(run_dir, per_image_rows, store_rows)
    log.info(
        "run %s complete: %d per-image rows, %d store rows",
        run_id,
        len(per_image_rows),
        len(store_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
