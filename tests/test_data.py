"""AI4Mars data-pipeline tests (Phase MS1). No network.

Two layers:
1. Offline synthetic-fixture tests on the REAL nested merged-0.6 layout (msl/<camera>/...,
   mer/images/{eff,test}), including regression fixtures for the two build_index bugs the flat
   fixture missed (camera crossing; MER label-stem shapes) and for min1-vs-min3 gold-dir pinning.
2. Real-layout count asserts (DEVPLAN 4.1/9) that run when the dataset is on disk and skip
   otherwise, so CI stays offline-green: msl ncam 16064/322 (min3), mer []/204, unique
   camera-qualified names, by-image split disjointness.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from aresseg.data import ai4mars
from aresseg.data.ai4mars import build_index, label_key
from aresseg.data.dataset import SegDataset, class_pixel_counts, make_splits
from aresseg.data.preflight import inspect_data
from aresseg.data.transforms import eval_transform, train_transform

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4mars"
requires_data = pytest.mark.skipif(
    not (DATA_ROOT / "ai4mars-dataset-merged-0.6").is_dir(),
    reason="AI4Mars dataset not on disk (offline CI)",
)

# ---------------------------------------------------------------------------
# synthetic fixture: the REAL nested layout in miniature
# ---------------------------------------------------------------------------


def _write_img(path: Path, rng, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rng.integers(0, 255, (size, size), dtype="uint8"))


def _write_mask(path: Path, rng, size=32, ignore_px=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = rng.integers(0, 4, (size, size), dtype="uint8")
    if ignore_px:
        m[0:2, 0:2] = 255  # a 2x2 ignore block survives nearest-neighbor downsampling
    cv2.imwrite(str(path), m)


def _make_tree(root: Path, n_train=6, n_test=2, size=32) -> Path:
    """Miniature merged-0.6 tree: nested cameras, mcam decoy, min1/min3 dirs, MER pools."""
    rng = np.random.default_rng(0)
    base = root / "ai4mars-dataset-merged-0.6"
    ncam = base / "msl" / "ncam"
    # ncam train: label stem == image stem
    for i in range(n_train):
        _write_img(ncam / "images" / "edr" / f"NTRAIN{i}.JPG", rng, size)
        _write_mask(ncam / "labels" / "train" / f"NTRAIN{i}.png", rng, size)
    # ncam gold test: label stem carries _merged; min1 is a DECOY with a different count so a
    # green count proves min3 was actually selected (real min1/min3 counts coincide at 322)
    for i in range(n_test):
        _write_img(ncam / "images" / "edr" / f"NTEST{i}.JPG", rng, size)
        _write_mask(
            ncam / "labels" / "test" / "masked-gold-min3-100agree" / f"NTEST{i}_merged.png",
            rng,
            size,
        )
    _write_mask(
        ncam / "labels" / "test" / "masked-gold-min1-100agree" / "NTEST0_merged.png", rng, size
    )
    # mcam decoy subtree (regression: Bug A paired ncam images with mcam labels => 0 pairs);
    # mcam JPGs sit directly under images/ (no edr/) and it has NO test split
    mcam = base / "msl" / "mcam"
    for i in range(3):
        _write_img(mcam / "images" / f"MCAM{i}.JPG", rng, size)
        _write_mask(mcam / "labels" / "train" / f"MCAM{i}.png", rng, size)
    # MER: JPGs under images/{eff,test} (NOT directly under images/); gold subset in BOTH pools;
    # labels/train exists but is EMPTY; gold labels carry _<digits>_T<digits>_merged
    mer = base / "mer"
    for i in range(4):
        _write_img(mer / "images" / "eff" / f"1n{i}eff0338p1931l0m1.JPG", rng, size)
    for i in range(2):
        _write_img(mer / "images" / "test" / f"1n{i}eff0338p1931l0m1.JPG", rng, size)
        _write_mask(
            mer
            / "labels"
            / "test"
            / "masked-gold-min3-100agree"
            / f"1n{i}eff0338p1931l0m1_16165_T0_merged.png",
            rng,
            size,
        )
    # min1 DECOY on the MER side too (different count), so a green MER count proves min3
    _write_mask(
        mer
        / "labels"
        / "test"
        / "masked-gold-min1-100agree"
        / "1n0eff0338p1931l0m1_16165_T0_merged.png",
        rng,
        size,
    )
    (mer / "labels" / "train").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# label_key
# ---------------------------------------------------------------------------


def test_label_key_shapes():
    assert label_key("abc") == "abc"
    assert label_key("abc_merged") == "abc"
    assert label_key("abc_label") == "abc"
    assert label_key("1n129697839eff0338p1931l0m1_16165_T0_merged") == "1n129697839eff0338p1931l0m1"
    # MSL stems end in alphanumeric tokens (e.g. _04096M1) that must NOT be stripped
    assert (
        label_key("NLA_397586934EDR_F0010008AUT_04096M1") == "NLA_397586934EDR_F0010008AUT_04096M1"
    )


# ---------------------------------------------------------------------------
# synthetic-layout build_index
# ---------------------------------------------------------------------------


def test_build_index_msl_camera_scoped(tmp_path):
    root = _make_tree(tmp_path)
    idx = build_index(root, "msl")
    assert len(idx["train"]) == 6
    assert len(idx["test"]) == 2  # min3 count, not min1's 1
    for rec in idx["train"] + idx["test"]:
        assert set(rec) == {"image", "label", "name"}
        assert "ncam" in Path(rec["image"]).parts and "ncam" in Path(rec["label"]).parts
        assert rec["name"].startswith("msl_ncam_")
    assert all("masked-gold-min3-100agree" in r["label"] for r in idx["test"])
    # pin the EXACT 4.3 name recipe (rover_camera_label_key(image_stem).lower()) to literals
    assert idx["train"][0]["name"] == "msl_ncam_ntrain0"
    assert idx["test"][0]["name"] == "msl_ncam_ntest0"


def test_build_index_never_crosses_cameras(tmp_path):
    """Regression (Bug A): mcam labels must never pair against ncam images or vice versa."""
    root = _make_tree(tmp_path)
    ncam = build_index(root, "msl", camera="ncam")
    mcam = build_index(root, "msl", camera="mcam")
    assert len(ncam["train"]) == 6 and len(mcam["train"]) == 3
    assert mcam["test"] == []  # mcam has no gold test split
    assert all(r["name"].startswith("msl_mcam_") for r in mcam["train"])
    assert not ({r["name"] for r in ncam["train"]} & {r["name"] for r in mcam["train"]})


def test_build_index_mer(tmp_path):
    """Regression (Bug B): MER pools + suffixed label stems; train stays empty."""
    root = _make_tree(tmp_path)
    idx = build_index(root, "mer")
    assert idx["train"] == []
    assert len(idx["test"]) == 2  # min3 count, not the min1 decoy's 1
    for rec in idx["test"]:
        assert rec["name"].startswith("mer_test_")
        assert Path(rec["image"]).parent.name == "test"  # gold pool wins the eff/test union
        assert Path(rec["label"]).name.endswith("_T0_merged.png")
        assert "masked-gold-min3-100agree" in rec["label"]
    # literal name pins: label suffix tokens must be stripped, image stem lowercased
    assert idx["test"][0]["name"] == "mer_test_1n0eff0338p1931l0m1"


def test_build_index_gold_dir_pinning(tmp_path):
    root = _make_tree(tmp_path)
    # explicit bare name and base-relative path both resolve
    min1 = build_index(root, "msl", test_gold_dir="masked-gold-min1-100agree")
    assert len(min1["test"]) == 1
    rel = build_index(root, "msl", test_gold_dir="msl/ncam/labels/test/masked-gold-min3-100agree")
    assert len(rel["test"]) == 2
    assert len(build_index(root, "mer", test_gold_dir="masked-gold-min1-100agree")["test"]) == 1
    # a pinned-but-absent gold dir is a hard error, never a silent fallback
    with pytest.raises(FileNotFoundError):
        build_index(root, "msl", test_gold_dir="masked-gold-min9-100agree")
    # a pin whose PATH points outside this rover/camera scope resolves by basename WITHIN the
    # scope (the agreement level is what the pin guarantees), never to the foreign tree
    cross = build_index(root, "mer", test_gold_dir="msl/ncam/labels/test/masked-gold-min3-100agree")
    assert len(cross["test"]) == 2
    assert all(Path(r["label"]).parts.count("mer") == 1 for r in cross["test"])
    # an explicit pin with no labels/test root at all must raise, not return an empty test set
    with pytest.raises(FileNotFoundError):
        build_index(root, "msl", camera="mcam", test_gold_dir="masked-gold-min3-100agree")


def test_build_index_accepts_both_roots_and_is_deterministic(tmp_path):
    root = _make_tree(tmp_path)
    a = build_index(root, "msl")
    b = build_index(root / "ai4mars-dataset-merged-0.6", "msl")
    assert a == b
    assert a["train"] == sorted(a["train"], key=lambda r: r["name"])


def test_build_index_unknown_rover(tmp_path):
    with pytest.raises(ValueError):
        build_index(_make_tree(tmp_path), "m2020")


# ---------------------------------------------------------------------------
# dataset / splits / class counts (synthetic)
# ---------------------------------------------------------------------------


def test_dataset_item_and_ignore(tmp_path):
    root = _make_tree(tmp_path)
    idx = build_index(root, "msl")
    # size != fixture size, so the mask resize path (nearest, no value blending) is exercised
    ds = SegDataset(idx["train"], eval_transform(size=16))
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 16, 16)
    assert tuple(item["mask"].shape) == (16, 16)
    assert str(item["mask"].dtype) == "torch.int64"
    vals = set(item["mask"].unique().tolist())
    assert vals <= {0, 1, 2, 3, ai4mars.IGNORE_INDEX}
    assert ai4mars.IGNORE_INDEX in vals  # the ignore block survived the resize
    assert item["name"] == idx["train"][0]["name"] and item["rover"] == "msl"


def test_dataset_missing_name_is_hard_error(tmp_path):
    """DEVPLAN 4.3: no positional str(i) fallback — a missing join key must raise."""
    root = _make_tree(tmp_path)
    rec = {k: v for k, v in build_index(root, "msl")["train"][0].items() if k != "name"}
    with pytest.raises(KeyError):
        SegDataset([rec], eval_transform(size=32))[0]


def test_splits_disjoint_by_image(tmp_path):
    root = _make_tree(tmp_path, n_train=10)
    idx = build_index(root, "msl")
    sp = make_splits(idx["train"], val_frac=0.3, seed=1414)
    train_names = {r["name"] for r in sp["train"]}
    val_names = {r["name"] for r in sp["val"]}
    assert train_names.isdisjoint(val_names)
    assert len(sp["train"]) + len(sp["val"]) == len(idx["train"])
    # same seed + same input order => byte-identical split membership (V100 reproducibility)
    assert make_splits(idx["train"], val_frac=0.3, seed=1414) == sp


def test_class_pixel_counts(tmp_path):
    root = _make_tree(tmp_path)
    idx = build_index(root, "msl")
    counts = class_pixel_counts(idx["train"], num_classes=4)
    assert counts.sum() > 0 and len(counts) == 4


def test_train_transform_uses_aug_config():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 1
    transform = train_transform(
        size=8,
        aug={
            "hflip_p": 1.0,
            "rbc_p": 0.0,
            "vflip": False,
            "scale_crop": False,
        },
    )
    flipped = transform(image=image, mask=mask)["mask"]
    assert set(flipped[:, :4].unique().tolist()) == {1}
    assert set(flipped[:, 4:].unique().tolist()) == {0}


def test_preflight_synthetic_contract_and_fingerprint(tmp_path):
    root = _make_tree(tmp_path)
    report = inspect_data(
        root,
        expected_msl_train=6,
        expected_msl_test=2,
        expected_mer_test=2,
        require_archive=False,
    )
    assert report["ok"], report["errors"]
    assert report["counts"] == {
        "msl_train": 6,
        "msl_test": 2,
        "mer_train": 0,
        "mer_test": 2,
    }
    assert report["masks"]["scanned"] == 10
    assert len(report["fingerprints"]["combined_index_sha256"]) == 64
    again = inspect_data(
        root,
        expected_msl_train=6,
        expected_msl_test=2,
        expected_mer_test=2,
        require_archive=False,
        scan_masks=False,
    )
    assert again["fingerprints"]["combined_index_sha256"] == (
        report["fingerprints"]["combined_index_sha256"]
    )


def test_preflight_rejects_orphan_label_and_bad_mask(tmp_path):
    root = _make_tree(tmp_path)
    base = root / "ai4mars-dataset-merged-0.6"
    orphan = base / "msl" / "ncam" / "labels" / "train" / "ORPHAN.png"
    _write_mask(orphan, np.random.default_rng(4))
    bad = base / "msl" / "ncam" / "labels" / "train" / "NTRAIN0.png"
    mask = cv2.imread(str(bad), cv2.IMREAD_UNCHANGED)
    mask[4, 4] = 9
    cv2.imwrite(str(bad), mask)
    report = inspect_data(
        root,
        expected_msl_train=6,
        expected_msl_test=2,
        expected_mer_test=2,
        require_archive=False,
    )
    assert not report["ok"]
    assert report["pairing"]["msl"]["unmatched_label_count"] == 1
    assert report["masks"]["invalid_count"] == 1
    assert any("unmatched labels" in error for error in report["errors"])
    assert any("invalid values" in error for error in report["errors"])


def test_download_reuses_verified_archive_without_network(monkeypatch, tmp_path):
    import hashlib

    from scripts import download_data

    archive = tmp_path / "archive.zip"
    payload = b"already complete"
    archive.write_bytes(payload)
    expected = hashlib.md5(payload).hexdigest()

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("verified archive must not open the network")

    monkeypatch.setattr(download_data.requests, "get", network_forbidden)
    assert download_data.download("https://invalid.test/archive", archive, expected) == archive
    assert archive.read_bytes() == payload


def test_download_bad_cached_file_uses_response_and_revalidates(monkeypatch, tmp_path):
    import hashlib

    from scripts import download_data

    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"bad")
    replacement = b"verified replacement"
    expected = hashlib.md5(replacement).hexdigest()
    seen = {}

    class Response:
        status_code = 200
        headers = {"Content-Length": str(len(replacement))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield replacement

        def raise_for_status(self):
            raise AssertionError("status 200 must not raise")

    def fake_get(_url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(download_data.requests, "get", fake_get)
    assert download_data.download("https://invalid.test/archive", archive, expected) == archive
    assert archive.read_bytes() == replacement
    assert seen["headers"] == {"Range": "bytes=3-"}


# ---------------------------------------------------------------------------
# real-layout asserts (DEVPLAN 4.1 step 4 / MS1 gate) — skip when data absent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def msl_index():
    return build_index(DATA_ROOT, "msl")


@pytest.fixture(scope="module")
def mer_index():
    return build_index(DATA_ROOT, "mer")


@requires_data
def test_real_msl_counts(msl_index):
    assert len(msl_index["train"]) == 16064
    assert len(msl_index["test"]) == 322
    # the pinned min3 gold dir is actually the one in use (counts alone cannot prove this)
    assert all("masked-gold-min3-100agree" in r["label"] for r in msl_index["test"])
    assert all("ncam" in Path(r["image"]).parts for r in msl_index["train"] + msl_index["test"])


@requires_data
def test_real_mer_counts(mer_index):
    assert mer_index["train"] == []
    assert len(mer_index["test"]) == 204
    # min1/min2/min3 all hold 204 files, so pin the dir by PATH, exactly as on the MSL side
    assert all("masked-gold-min3-100agree" in r["label"] for r in mer_index["test"])


@requires_data
def test_real_names_unique_and_qualified(msl_index, mer_index):
    for idx, prefix in ((msl_index, "msl_ncam_"), (mer_index, "mer_test_")):
        for split in ("train", "test"):
            names = [r["name"] for r in idx[split]]
            assert all(names) and len(set(names)) == len(names)
            assert all(n.startswith(prefix) for n in names)


@requires_data
def test_real_splits_disjoint(msl_index):
    sp = make_splits(msl_index["train"], val_frac=0.2, seed=1414)
    assert {r["name"] for r in sp["train"]}.isdisjoint({r["name"] for r in sp["val"]})
    assert len(sp["val"]) == round(16064 * 0.2)


@requires_data
def test_real_item_contract(msl_index):
    item = SegDataset(msl_index["test"][:1], eval_transform(size=512))[0]
    assert tuple(item["image"].shape) == (3, 512, 512)
    assert str(item["image"].dtype) == "torch.float32"
    assert tuple(item["mask"].shape) == (512, 512)
    assert str(item["mask"].dtype) == "torch.int64"
    assert set(item["mask"].unique().tolist()) <= {0, 1, 2, 3, 255}
    assert item["name"].startswith("msl_ncam_") and item["rover"] == "msl"
