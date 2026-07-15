"""Strict, unit-testable AI4Mars dataset preflight checks.

The full preflight validates the exact protocol counts, label/image pairing, mask vocabulary,
split isolation, source archive checksum, and a stable fingerprint of the sorted runtime index.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .ai4mars import DEFAULT_TEST_GOLD_DIR, IMAGE_EXTS, build_index
from .dataset import make_splits

MERGED_ARCHIVE_NAME = "ai4mars-dataset-merged-0.6.zip"
MERGED_ARCHIVE_MD5 = "daf80a86021253292e6c425f97baa5c6"
ALLOWED_MASK_VALUES = frozenset({0, 1, 2, 3, 255})


class DataPreflightError(RuntimeError):
    """Raised with the complete report when one or more data contracts fail."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("AI4Mars preflight failed: " + "; ".join(report["errors"]))


def _resolve_base(root: str | Path) -> Path:
    root = Path(root)
    if (root / "msl").is_dir() or (root / "mer").is_dir():
        return root
    nested = root / "ai4mars-dataset-merged-0.6"
    return nested if nested.is_dir() else root


def _relative(path: str | Path, root: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_digest(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def index_fingerprint(index: dict, root: str | Path) -> str:
    """SHA-256 of sorted identity/path/size rows, independent of extraction location."""
    base = _resolve_base(root)
    rows: list[dict] = []
    for split in sorted(index):
        for record in sorted(index[split], key=lambda item: item["name"]):
            image = Path(record["image"])
            label = Path(record["label"])
            rows.append(
                {
                    "split": split,
                    "name": record["name"],
                    "image": _relative(image, base),
                    "label": _relative(label, base),
                    "image_bytes": image.stat().st_size,
                    "label_bytes": label.stat().st_size,
                }
            )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gold_dir(base: Path, rover: str, camera: str, configured: str | None) -> Path:
    name = Path(configured).name if configured else DEFAULT_TEST_GOLD_DIR
    if rover == "msl":
        return base / "msl" / camera / "labels" / "test" / name
    return base / "mer" / "labels" / "test" / name


def _raw_labels(
    base: Path,
    rover: str,
    camera: str,
    configured_gold_dir: str | None,
) -> dict[str, list[Path]]:
    if rover == "msl":
        train_dir = base / "msl" / camera / "labels" / "train"
        train = sorted(train_dir.glob("*.png")) if train_dir.is_dir() else []
    else:
        train = []
    gold = _gold_dir(base, rover, camera, configured_gold_dir)
    return {
        "train": train,
        "test": sorted(gold.glob("*.png")) if gold.is_dir() else [],
    }


def _image_pool(base: Path, rover: str, camera: str) -> list[Path]:
    if rover == "msl":
        root = base / "msl" / camera / "images"
        root = root / "edr" if (root / "edr").is_dir() else root
        dirs = [root]
    else:
        dirs = [base / "mer" / "images" / "eff", base / "mer" / "images" / "test"]
    images: set[Path] = set()
    extensions = {ext.lower() for ext in IMAGE_EXTS}
    for directory in dirs:
        if directory.is_dir():
            images.update(
                path.resolve()
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
    return sorted(images)


def _pairing_report(
    index: dict,
    raw: dict[str, list[Path]],
    image_pool: list[Path],
) -> dict:
    paired_labels = {
        Path(record["label"]).resolve() for records in index.values() for record in records
    }
    all_raw_labels = {path.resolve() for records in raw.values() for path in records}
    paired_images = [
        Path(record["image"]).resolve() for records in index.values() for record in records
    ]
    duplicate_images = [str(path) for path, count in Counter(paired_images).items() if count > 1]
    missing_paths = [
        str(path)
        for records in index.values()
        for record in records
        for path in (Path(record["image"]), Path(record["label"]))
        if not path.is_file()
    ]
    unmatched = sorted(str(path) for path in all_raw_labels - paired_labels)
    pool = {path.resolve() for path in image_pool}
    return {
        "raw_labels": len(all_raw_labels),
        "paired_labels": len(paired_labels),
        "unmatched_label_count": len(unmatched),
        "unmatched_label_examples": unmatched[:20],
        "duplicate_paired_image_count": len(duplicate_images),
        "duplicate_paired_image_examples": duplicate_images[:20],
        "missing_record_path_count": len(missing_paths),
        "missing_record_path_examples": missing_paths[:20],
        # AI4Mars intentionally includes unlabeled test-pool images. Report, but do not fail, this
        # source-level difference; every protocol label must still have exactly one paired image.
        "image_pool_count": len(pool),
        "unpaired_image_pool_count": len(pool - set(paired_images)),
    }


def _mask_report(label_paths: list[Path], scan_masks: bool) -> dict:
    if not scan_masks:
        return {"scanned": 0, "invalid_count": 0, "invalid_examples": []}
    invalid: list[dict] = []
    for path in label_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            invalid.append({"path": str(path), "reason": "unreadable"})
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        values = {int(value) for value in np.unique(mask)}
        unexpected = sorted(values - ALLOWED_MASK_VALUES)
        if unexpected:
            invalid.append({"path": str(path), "unexpected_values": unexpected})
    return {
        "scanned": len(label_paths),
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
    }


def inspect_data(
    root: str | Path,
    *,
    expected_msl_train: int = 16064,
    expected_msl_test: int = 322,
    expected_mer_test: int = 204,
    camera: str = "ncam",
    msl_gold_dir: str | None = None,
    mer_gold_dir: str | None = None,
    val_frac: float = 0.2,
    split_seed: int = 1414,
    archive_path: str | Path | None = None,
    expected_archive_md5: str | None = MERGED_ARCHIVE_MD5,
    require_archive: bool = True,
    hash_archive: bool = True,
    scan_masks: bool = True,
) -> dict:
    """Inspect data and return a serializable report; errors are accumulated, not short-circuited."""
    base = _resolve_base(root).resolve()
    errors: list[str] = []
    try:
        msl = build_index(base, "msl", camera=camera, test_gold_dir=msl_gold_dir)
        mer = build_index(base, "mer", test_gold_dir=mer_gold_dir)
    except Exception as exc:
        return {
            "ok": False,
            "root": str(base),
            "errors": [f"index construction failed: {exc}"],
        }

    counts = {
        "msl_train": len(msl["train"]),
        "msl_test": len(msl["test"]),
        "mer_train": len(mer["train"]),
        "mer_test": len(mer["test"]),
    }
    expected = {
        "msl_train": int(expected_msl_train),
        "msl_test": int(expected_msl_test),
        "mer_train": 0,
        "mer_test": int(expected_mer_test),
    }
    for key, wanted in expected.items():
        if counts[key] != wanted:
            errors.append(f"{key} count {counts[key]} != expected {wanted}")

    raw_msl = _raw_labels(base, "msl", camera, msl_gold_dir)
    raw_mer = _raw_labels(base, "mer", camera, mer_gold_dir)
    pairing = {
        "msl": _pairing_report(msl, raw_msl, _image_pool(base, "msl", camera)),
        "mer": _pairing_report(mer, raw_mer, _image_pool(base, "mer", camera)),
    }
    for rover, diagnostic in pairing.items():
        if diagnostic["unmatched_label_count"]:
            errors.append(f"{rover} has {diagnostic['unmatched_label_count']} unmatched labels")
        if diagnostic["duplicate_paired_image_count"]:
            errors.append(
                f"{rover} has {diagnostic['duplicate_paired_image_count']} multiply-paired images"
            )
        if diagnostic["missing_record_path_count"]:
            errors.append(f"{rover} has {diagnostic['missing_record_path_count']} missing paths")

    split = make_splits(msl["train"], val_frac=float(val_frac), seed=int(split_seed))
    train_names = {record["name"] for record in split["train"]}
    val_names = {record["name"] for record in split["val"]}
    msl_test_names = {record["name"] for record in msl["test"]}
    mer_test_names = {record["name"] for record in mer["test"]}
    overlaps = {
        "train_val": len(train_names & val_names),
        "train_msl_test": len(train_names & msl_test_names),
        "val_msl_test": len(val_names & msl_test_names),
        "msl_mer_test": len(msl_test_names & mer_test_names),
    }
    if any(overlaps.values()):
        errors.append(f"split identifier overlap detected: {overlaps}")

    labels = [path for records in raw_msl.values() for path in records]
    labels += [path for records in raw_mer.values() for path in records]
    masks = _mask_report(labels, scan_masks)
    if masks["invalid_count"]:
        errors.append(f"{masks['invalid_count']} masks are unreadable or contain invalid values")

    if archive_path is not None:
        archive = Path(archive_path)
    else:
        candidates = [
            base.parent / MERGED_ARCHIVE_NAME,
            base / MERGED_ARCHIVE_NAME,
            Path(root) / MERGED_ARCHIVE_NAME,
        ]
        archive = next(
            (candidate for candidate in candidates if candidate.is_file()), candidates[0]
        )
    archive_present = archive.is_file()
    archive_md5 = file_digest(archive, "md5") if archive_present and hash_archive else None
    if require_archive and not archive_present:
        errors.append(f"source archive missing: {archive}")
    if (
        archive_md5 is not None
        and expected_archive_md5 is not None
        and archive_md5.lower() != expected_archive_md5.lower()
    ):
        errors.append(f"archive MD5 {archive_md5} != expected {expected_archive_md5.lower()}")

    fingerprints = {
        "msl_index_sha256": index_fingerprint(msl, base),
        "mer_index_sha256": index_fingerprint(mer, base),
        "archive_path": str(archive),
        "archive_present": archive_present,
        "archive_bytes": archive.stat().st_size if archive_present else None,
        "archive_md5": archive_md5,
        "expected_archive_md5": expected_archive_md5,
    }
    combined_payload = (fingerprints["msl_index_sha256"] + fingerprints["mer_index_sha256"]).encode(
        "ascii"
    )
    fingerprints["combined_index_sha256"] = hashlib.sha256(combined_payload).hexdigest()

    return {
        "ok": not errors,
        "root": str(base),
        "counts": counts,
        "expected_counts": expected,
        "pairing": pairing,
        "split_sizes": {"train": len(split["train"]), "val": len(split["val"])},
        "split_overlaps": overlaps,
        "masks": masks,
        "fingerprints": fingerprints,
        "errors": errors,
    }


def require_valid_data(*args, **kwargs) -> dict:
    """Return a successful report or raise DataPreflightError with all diagnostics."""
    report = inspect_data(*args, **kwargs)
    if not report["ok"]:
        raise DataPreflightError(report)
    return report
