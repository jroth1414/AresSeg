"""AI4Mars dataset constants + (image, label) index builder (Phase MS1).

Label PNGs are single-channel with terrain pixel values 0-3; 255 = NULL/ignore (unlabeled,
rover-self, and >30 m range are merged into 255 in the merged-0.6 release).

Verified merged-0.6 layout (DEVPLAN section 4). MSL is nested PER CAMERA and the protocol is
ncam-only; MER is cross-rover TEST ONLY (no train labels), with JPGs under images/{eff,test}:

    <root>/msl/ncam/images/edr/*.JPG              images (18,127)
    <root>/msl/ncam/labels/train/*.png            train labels (16,064; stem == image stem)
    <root>/msl/ncam/labels/test/<gold>/*.png      gold test labels (322; stem + "_merged")
    <root>/msl/mcam/...                           color mastcam - EXCLUDED from the protocol
    <root>/mer/images/{eff,test}/*.JPG            image pool (gold subset lives in BOTH)
    <root>/mer/labels/test/<gold>/*.png           gold test labels (204; stem + "_<n>_T<n>_merged")

Pairing is done WITHIN one camera subtree (never across cameras), and label stems are normalized
back to image stems via ``label_key``. The pinned gold test dir is min3 (``DEFAULT_TEST_GOLD_DIR``);
it is honored explicitly, never picked by sort order.
"""

from __future__ import annotations

import re
from pathlib import Path

CLASSES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES = 4
IGNORE_INDEX = 255  # NULL / unlabeled / rover-self / >30 m range
# RGB colors for segmentation overlays (paper figures).
CLASS_COLORS = {
    0: (205, 133, 63),  # soil — tan
    1: (112, 128, 144),  # bedrock — slate
    2: (238, 214, 175),  # sand — pale
    3: (139, 0, 0),  # big rock — dark red
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG")

# The pinned protocol test set (DEVPLAN section 5.4). min1/min2/min3 have identical counts, so a
# green count does NOT prove the right set — selection must be explicit.
DEFAULT_TEST_GOLD_DIR = "masked-gold-min3-100agree"

_TRAILING_TOKEN = re.compile(r"_(?:T\d+|\d+)$")


def label_key(stem: str) -> str:
    """Normalize a label filename stem back to its image stem.

    Handles all four naming shapes on disk: bare image stem, ``<stem>_merged`` / ``<stem>_label``,
    and MER's ``<stem>_<digits>_T<digits>_merged`` (trailing ``_<digits>`` / ``_T<digits>`` tokens
    are stripped repeatedly).
    """
    for suffix in ("_merged", "_label"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    while True:
        m = _TRAILING_TOKEN.search(stem)
        if m is None:
            return stem
        stem = stem[: m.start()]


def _index_images(image_dirs: list[Path]) -> dict[str, Path]:
    """Map image stem -> path over the given dirs; later dirs win on stem collisions."""
    images: dict[str, Path] = {}
    for d in image_dirs:
        if not d.is_dir():
            continue
        for ext in IMAGE_EXTS:
            for p in d.glob(f"*{ext}"):
                images[p.stem] = p
    return images


def _match_labels(images: dict[str, Path], label_dir: Path, name_prefix: str) -> list[dict]:
    """Pair labels with images on the raw stem, else on progressively normalized stems.

    Each record carries the canonical camera-qualified ``name`` join key (DEVPLAN section 4.3):
    ``{rover}_{camera_or_pool}_{label_key(image_stem).lower()}``.
    """
    pairs = []
    for lab in sorted(label_dir.glob("*.png")):
        stem = lab.stem
        img = images.get(stem)
        if img is None:  # strip _merged/_label, then trailing _<digits>/_T<digits> tokens
            for suffix in ("_merged", "_label"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            img = images.get(stem)
            while img is None:
                m = _TRAILING_TOKEN.search(stem)
                if m is None:
                    break
                stem = stem[: m.start()]
                img = images.get(stem)
        if img is not None:
            name = f"{name_prefix}_{label_key(img.stem).lower()}"
            pairs.append({"image": str(img), "label": str(lab), "name": name})
    pairs.sort(key=lambda r: r["name"])
    return pairs


def _resolve_base(root: Path) -> Path:
    """Accept either the extraction dir or the nested dataset dir as root."""
    if (root / "msl").is_dir() or (root / "mer").is_dir():
        return root
    nested = root / "ai4mars-dataset-merged-0.6"
    if nested.is_dir():
        return nested
    return root


def _resolve_gold_dir(base: Path, test_root: Path, test_gold_dir: str | None) -> Path:
    """Resolve the pinned gold test dir; explicit selection, never sort order.

    The resolved dir must live under THIS camera's labels/test root, so a pin that points at
    another rover/camera can never silently redirect the test set (the basename — the agreement
    level — is what the pin guarantees).
    """
    if test_gold_dir:
        cand = Path(test_gold_dir)
        for c in (base / cand, test_root / cand.name):
            if c.is_dir() and c.resolve().is_relative_to(test_root.resolve()):
                return c
        raise FileNotFoundError(
            f"pinned test_gold_dir {test_gold_dir!r} not found under {test_root}"
        )
    chosen = test_root / DEFAULT_TEST_GOLD_DIR
    if not chosen.is_dir():
        raise FileNotFoundError(f"default gold dir {DEFAULT_TEST_GOLD_DIR!r} not in {test_root}")
    return chosen


def build_index(
    root: str | Path,
    rover: str = "msl",
    camera: str = "ncam",
    test_gold_dir: str | None = None,
) -> dict:
    """Index (image, label, name) records for a rover. Returns {'train': [...], 'test': [...]}.

    ``rover`` in {'msl' (Curiosity), 'mer' (Opportunity/Spirit)}. For MSL, pairing is scoped to ONE
    camera subtree (default 'ncam' — the protocol camera; 'mcam' is excluded from reported numbers).
    For MER there are no train labels (train == []) and images are searched under images/{eff,test}
    (the gold subset exists in both; 'test' wins, so names are 'mer_test_...'). ``test_gold_dir``
    pins the gold label dir (path relative to the dataset base, or a bare dir name); default is the
    pinned min3 set.
    """
    base = _resolve_base(Path(root))
    out: dict = {"train": [], "test": []}
    if rover == "msl":
        cam_dir = base / "msl" / camera
        img_root = cam_dir / "images"
        # ncam JPGs live under images/edr; mcam JPGs sit directly under images/
        edr = img_root / "edr"
        images = _index_images([edr if edr.is_dir() else img_root])
        prefix = f"msl_{camera}"
        train_lab = cam_dir / "labels" / "train"
        if train_lab.is_dir():
            out["train"] = _match_labels(images, train_lab, prefix)
        test_root = cam_dir / "labels" / "test"
        if test_root.is_dir():
            gold = _resolve_gold_dir(base, test_root, test_gold_dir)
            out["test"] = _match_labels(images, gold, prefix)
        elif test_gold_dir:
            raise FileNotFoundError(f"test_gold_dir pinned but {test_root} does not exist")
    elif rover == "mer":
        rdir = base / "mer"
        # union of the two pools; 'test' (the gold subset) wins stem collisions
        images = _index_images([rdir / "images" / "eff", rdir / "images" / "test"])
        train_lab = rdir / "labels" / "train"
        if train_lab.is_dir():
            out["train"] = _match_labels(images, train_lab, "mer_eff")
        test_root = rdir / "labels" / "test"
        if test_root.is_dir():
            gold = _resolve_gold_dir(base, test_root, test_gold_dir)
            out["test"] = _match_labels(images, gold, "mer_test")
        elif test_gold_dir:
            raise FileNotFoundError(f"test_gold_dir pinned but {test_root} does not exist")
    else:
        raise ValueError(f"unknown rover {rover!r}; expected 'msl' or 'mer'")
    for split, recs in out.items():
        names = [r["name"] for r in recs]
        if len(set(names)) != len(names):  # two labels resolved to one image, or a stem collision
            raise ValueError(f"duplicate canonical names in {rover}/{split} index")
    return out
