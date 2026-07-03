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
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from marsseg.data.ai4mars import IGNORE_INDEX, NUM_CLASSES, build_index  # noqa: E402
from marsseg.data.dataset import SegDataset, class_pixel_counts, make_splits  # noqa: E402
from marsseg.data.transforms import eval_transform  # noqa: E402
from marsseg.eval import aggregate, metrics  # noqa: E402
from marsseg.utils.capabilities import detect  # noqa: E402
from marsseg.utils.config import load_config  # noqa: E402
from marsseg.utils.logging import get_logger  # noqa: E402
from marsseg.utils.manifest import _git_sha, config_hash, write_manifest  # noqa: E402
from marsseg.utils.results import append_results  # noqa: E402
from marsseg.utils.seed import set_seed  # noqa: E402

log = get_logger("marsseg.run")

FOUNDATION_VARIANTS = {"dinov3_sat": "finetuned", "sam": "zeroshot"}


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
        help="with --eval-only: also evaluate the MER cross-rover gold test (5.7)",
    )
    return ap.parse_args(argv)


def _variant(model_cfg: dict) -> str:
    name = model_cfg["name"]
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
    append_results(store_rows)
    mirror = REPO_ROOT / "experiments" / "manifests" / run_dir.name
    mirror.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.json", "per_image.parquet", "per_image.csv"):
        src = run_dir / f
        if src.is_file():
            shutil.copy2(src, mirror / f)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override, base_paths=[str(REPO_ROOT / "configs/data.yaml")])
    seed = int(cfg["data"].get("seed", 1414))
    set_seed(seed)
    profile = detect().profile
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    model_name = mcfg["name"]
    variant = _variant(mcfg)
    results_backbone = mcfg.get("results_backbone") or "none"
    chash = config_hash(cfg)
    git_sha = _git_sha()
    split_scope = "cross_rover" if args.h4 else "in_rover"
    run_id = (
        f"{model_name}__{variant}__{results_backbone}__{split_scope}"
        f"__seed{seed}__{profile}__{chash[:8]}"
    )
    run_dir = Path(args.out) if args.out else REPO_ROOT / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
    if len(index["test"]) != int(dcfg["expected_test_n"]):
        raise AssertionError(
            f"MSL test count {len(index['test'])} != expected {dcfg['expected_test_n']}"
        )
    splits = make_splits(
        index["train"], val_frac=float(dcfg["val_frac"]), seed=int(dcfg["split_seed"])
    )
    max_train = dcfg.get("max_train_images")
    val_truncated_to = None
    if max_train:  # CPU-smoke cap (5.4/8): first N after make_splits, before the DataModule
        splits["train"] = splits["train"][: int(max_train)]
        val_truncated_to = min(int(max_train), len(splits["val"]))
        splits["val"] = splits["val"][:val_truncated_to]
    for what, recs in (("train", splits["train"]), ("val", splits["val"]), ("test", index["test"])):
        _assert_names(recs, what)

    # ---------- gated foundation: skip-and-log path ----------
    stages: list[str] = ["index", "splits"]
    skipped: list[str] = []

    def _skip(reason_ctx: str) -> int:
        skipped.append(model_name)
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes={
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(index["test"]),
            },
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source=None,
            val_truncated_to=val_truncated_to,
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
                }
            ],
        )
        log.warning("arm %s skipped (%s); status=skipped row appended", model_name, reason_ctx)
        return 0

    if model_name == "sam":
        from marsseg.models.foundation import build_foundation

        sam_model = build_foundation("sam", sam_checkpoint=mcfg.get("sam_checkpoint"))
        if sam_model is None:
            return _skip("gating")
        rows = sam_region_oracle(
            sam_model,
            index["test"],
            size=int(dcfg["size"]),
            run_dir=run_dir,
            split="test_msl",
            run_id=run_id,
        )
        stages += ["eval_test_msl"]
        store = aggregate.store_rows(
            aggregate.per_image_frame(rows),
            split="test_msl",
            stratum="all",
            status="ok",
            **base_row,
        )
        write_manifest(
            run_dir,
            cfg,
            seed,
            profile=profile,
            model=model_name,
            backbone=results_backbone,
            variant=variant,
            dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
            data_hashes={"n_test": len(index["test"])},
            stages_completed=stages,
            gpu_stages_skipped=skipped,
            resolved_test_gold_dir=dcfg["test_gold_dir"],
            weights_source="cache",
            val_truncated_to=None,
        )
        _finalize(run_dir, rows, store)
        return 0

    # ---------- trainable arms (baseline / unet / deeplabv3plus / segformer / dinov3_sat) ----------
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from marsseg.train.lit import SegDataModule, SegLitModule

    weights_source = _weights_source(mcfg)
    class_weights = _class_weights(splits["train"], cfg.get("class_weights", {}))
    stages.append("class_weights")
    log.info("class_weights=%s weights_source=%s", class_weights, weights_source)

    lit = SegLitModule(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        backbone=mcfg.get("backbone"),
        pretrained=bool(mcfg.get("pretrained")),
        class_weights=class_weights,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        dice_weight=float(tcfg["dice_weight"]),
        ignore_index=int(tcfg["ignore_index"]),
        max_epochs=int(tcfg["max_epochs"]),
    )
    if lit.model is None:  # gated dinov3_sat on CPU / missing token -> skip-and-log (5.4)
        return _skip("gating")

    best_val_miou = None
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
        )
        ckpt_cb = ModelCheckpoint(
            dirpath=run_dir, filename="best", monitor="val_miou", mode="max", save_top_k=1
        )
        trainer = L.Trainer(
            max_epochs=int(tcfg["max_epochs"]),
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if profile == "gpu_full" else "32-true",
            gradient_clip_val=float(tcfg["grad_clip"]),
            callbacks=[
                ckpt_cb,
                EarlyStopping(
                    monitor="val_miou",
                    mode="max",
                    patience=int(tcfg["early_stop_patience"]),
                    min_delta=0.001,
                ),
            ],
            logger=False,
            enable_progress_bar=False,
            default_root_dir=run_dir,
        )
        trainer.fit(lit, dm)
        stages.append("train")
        if ckpt_cb.best_model_score is not None:
            best_val_miou = float(ckpt_cb.best_model_score)
        if ckpt_cb.best_model_path:  # evaluate the BEST checkpoint, not the last epoch
            state = torch.load(ckpt_cb.best_model_path, map_location=device, weights_only=False)
            lit.load_state_dict(state["state_dict"])

    per_image_rows: list[dict] = []
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

    write_manifest(
        run_dir,
        cfg,
        seed,
        profile=profile,
        model=model_name,
        backbone=results_backbone,
        variant=variant,
        dataset=f"ai4mars_{dcfg['rover']}_{dcfg['camera']}",
        data_hashes={
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(index["test"]),
        },
        stages_completed=stages,
        gpu_stages_skipped=skipped,
        class_weights=class_weights,
        resolved_test_gold_dir=dcfg["test_gold_dir"],
        weights_source=weights_source,
        best_val_miou=best_val_miou,
        val_truncated_to=val_truncated_to,
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
