"""Range-mask data helpers for the experimental MPBA training path.

The Protocol V3 indexer intentionally remains unaware of privileged ``rng-30m`` products.  This
module augments copies of its records and provides a dataset with one extra training target while
preserving the established ``image``/``mask``/``name``/``rover`` batch contract.

AI4Mars names a Navcam image product ``...EDR_....JPG`` and its 30 m mask
``...RNG_....png``.  A non-zero range-mask pixel denotes the top-of-image region beyond 30 m.
The scalar cutoff target is therefore the fraction of leading rows covered by that foreground.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_EDR_TOKEN = re.compile(r"EDR(?=_)", flags=re.IGNORECASE)
_BINARY_VALUES = frozenset({0, 1, 255})
MATCHED_COHORT_EXCLUDED_PRODUCTS = ("NLB_432655207EDR_F0160148NCAM00394M1.JPG",)


def _range_stems(image_stem: str) -> list[str]:
    """Return ordered candidate stems, preferring the AI4Mars EDR -> RNG product mapping."""
    converted, replacements = _EDR_TOKEN.subn("RNG", image_stem, count=1)
    return [converted, image_stem] if replacements and converted != image_stem else [image_stem]


def resolve_range_mask_path(
    record_or_image: Mapping[str, Any] | str | Path,
    *,
    range_dir: str | Path | None = None,
    strict: bool = True,
) -> Path | None:
    """Resolve an EDR image's ``rng-30m`` partner without modifying the sealed index.

    An explicit ``record["range_mask"]`` wins.  Otherwise, the normal AI4Mars layout is inferred
    from ``.../images/edr/<product>.JPG``.  ``range_dir`` can override that inferred directory for
    synthetic fixtures or relocated auxiliary products.  With ``strict=True`` (the training
    default), absence is a hard :class:`FileNotFoundError`; there is no fabricated target or
    positional fallback.
    """
    explicit: str | Path | None = None
    if isinstance(record_or_image, Mapping):
        if "image" not in record_or_image:
            raise KeyError("range-mask resolution requires record['image']")
        image_path = Path(record_or_image["image"])
        explicit = record_or_image.get("range_mask")
    else:
        image_path = Path(record_or_image)

    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path
        if strict:
            raise FileNotFoundError(f"explicit range mask does not exist: {path}")
        return None

    directory = Path(range_dir) if range_dir is not None else image_path.parent.parent / "rng-30m"
    candidates = [directory / f"{stem}.png" for stem in _range_stems(image_path.stem)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if strict:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"no rng-30m partner for {image_path}; tried: {rendered}")
    return None


def attach_range_masks(
    records: Sequence[Mapping[str, Any]],
    *,
    range_dir: str | Path | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Return copied records carrying a resolved ``range_mask`` path.

    The input dictionaries are never mutated.  In non-strict diagnostic use, records without a
    partner are retained without the extra key; MPBA training should always keep the strict
    default.
    """
    augmented: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        path = resolve_range_mask_path(copied, range_dir=range_dir, strict=strict)
        if path is not None:
            copied["range_mask"] = str(path)
        augmented.append(copied)
    return augmented


def apply_matched_range_cohort(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Exclude the one source product without a supplied range mask from every ablation.

    This policy is applied *after* the canonical seed-1414 split so validation membership remains
    unchanged.  It also verifies the pinned source condition instead of silently dropping any
    newly missing record.  Returned records are copies and the exclusions are manifest-ready.
    """
    filtered: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, str]] = []
    excluded = set(MATCHED_COHORT_EXCLUDED_PRODUCTS)
    for split_name, records in splits.items():
        kept: list[dict[str, Any]] = []
        for record in records:
            image_name = Path(record["image"]).name
            if image_name in excluded:
                exclusions.append(
                    {
                        "image": image_name,
                        "record_name": str(record.get("name", Path(image_name).stem)),
                        "split": str(split_name),
                        "reason": "missing_rng_30m_partner_in_source_snapshot",
                    }
                )
            else:
                kept.append(dict(record))
        filtered[str(split_name)] = kept

    found = [item["image"] for item in exclusions]
    if sorted(found) != sorted(MATCHED_COHORT_EXCLUDED_PRODUCTS):
        raise RuntimeError(
            "MPBA matched-cohort source condition changed: expected exclusions "
            f"{list(MATCHED_COHORT_EXCLUDED_PRODUCTS)}, found {found}"
        )
    if any(item["split"] != "train" for item in exclusions):
        raise RuntimeError(f"MPBA matched-cohort exclusion left the training split: {exclusions}")
    return filtered, exclusions


def read_range_mask(path: str | Path) -> np.ndarray:
    """Read and validate one binary range mask as a two-dimensional ``uint8`` array."""
    path = Path(path)
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"range mask must be two-dimensional, got {mask.shape} from {path}")
    values = {int(value) for value in np.unique(mask)}
    if not values <= _BINARY_VALUES:
        raise ValueError(
            f"range mask must use binary values 0/1 or 0/255, got {sorted(values)} from {path}"
        )
    return (mask != 0).astype(np.uint8)


def _as_numpy(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask)


def cutoff_target_from_mask(mask: Any, *, row_coverage: float = 0.5) -> float:
    """Derive the normalized end of a range mask's top-connected foreground band.

    A row counts as foreground when at least ``row_coverage`` of its pixels are non-zero.  The
    cutoff is the number of consecutive foreground rows starting at row zero divided by image
    height.  This gives exactly ``k / H`` for the horizontal bands supplied with AI4Mars, remains
    stable under horizontal flips/resizing, and maps an all-zero mask to ``0.0``.
    """
    value = _as_numpy(mask)
    if value.ndim == 3 and 1 in (value.shape[0], value.shape[-1]):
        value = np.squeeze(value)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"range mask must have non-empty shape (H,W), got {value.shape}")
    if not 0.0 < float(row_coverage) <= 1.0:
        raise ValueError("row_coverage must lie in (0, 1]")
    foreground_rows = np.mean(value != 0, axis=1) >= float(row_coverage)
    background = np.flatnonzero(~foreground_rows)
    leading_rows = int(background[0]) if background.size else int(value.shape[0])
    return float(leading_rows / value.shape[0])


class RangeAwareSegDataset:
    """SegDataset-compatible MPBA dataset with a scalar ``cutoff_target``.

    The semantic and range masks are passed through Albumentations as a joint ``masks`` list, so
    resize and horizontal augmentation stay pixel-aligned.  Set ``return_range_mask=True`` only
    for diagnostics; model inference never consumes the privileged mask.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        transform,
        rover: str = "msl",
        *,
        range_dir: str | Path | None = None,
        strict_range_masks: bool = True,
        return_range_mask: bool = False,
    ):
        self.records = attach_range_masks(records, range_dir=range_dir, strict=strict_range_masks)
        self.transform = transform
        self.rover = rover
        self.strict_range_masks = bool(strict_range_masks)
        self.return_range_mask = bool(return_range_mask)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return np.repeat(image[:, :, None], 3, axis=2)

    @staticmethod
    def _read_semantic_mask(path: str | Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        # Match the sealed dataset's hard join-key behavior.
        name = record["name"]
        image = self._read_image(record["image"])
        semantic_mask = self._read_semantic_mask(record["label"])
        range_path = resolve_range_mask_path(record, strict=self.strict_range_masks)
        if range_path is None:  # possible only for explicitly requested non-strict diagnostics
            raise FileNotFoundError(f"record {name!r} has no range mask")
        range_mask = read_range_mask(range_path)
        if image.shape[:2] != semantic_mask.shape or semantic_mask.shape != range_mask.shape:
            raise ValueError(
                "image, semantic mask, and range mask must share HxW for "
                f"{name}: {image.shape[:2]}, {semantic_mask.shape}, {range_mask.shape}"
            )

        transformed = self.transform(image=image, masks=[semantic_mask, range_mask])
        if "masks" not in transformed or len(transformed["masks"]) != 2:
            raise TypeError("MPBA transform must preserve the two-item `masks` list")
        semantic_out, range_out = transformed["masks"]
        semantic_tensor = torch.as_tensor(semantic_out, dtype=torch.long)
        range_tensor = torch.as_tensor(range_out, dtype=torch.uint8)
        image_tensor = torch.as_tensor(transformed["image"], dtype=torch.float32)
        item: dict[str, Any] = {
            "image": image_tensor,
            "mask": semantic_tensor,
            "name": name,
            "rover": self.rover,
            "cutoff_target": torch.tensor(
                cutoff_target_from_mask(range_tensor), dtype=torch.float32
            ),
        }
        if self.return_range_mask:
            item["range_mask"] = range_tensor.bool()
        return item
