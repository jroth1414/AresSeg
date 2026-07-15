# marsseg — Mars terrain semantic segmentation

`marsseg` is a Python 3.11 research pipeline for four-class AI4Mars semantic segmentation
(*soil, bedrock, sand, big rock*). It trains U-Net, DeepLabV3+, SegFormer, Tiny U-Net, and
DINOv3-based systems; evaluates a SAM region-oracle upper bound; and records per-image sufficient
statistics for preregistered paired inference.

The research protocol covers:

- H1: deep systems versus a constant-majority reference.
- H2: pretrained versus scratch initialization within each tested architecture.
- H3: configured SegFormer MiT-B2 systems versus U-Net/ResNet-34 and
  DeepLabV3+/ResNet-50, overall and by class.
- H4: Curiosity-to-MER transfer with a bounded mIoU-drop rule and majority-on-MER reference.
- H5: DINOv3-SAT and the SAM region-oracle upper bound versus the learned Tiny U-Net reference.

See [`DEVPLAN.md`](DEVPLAN.md) for the protocol and [`docs/DEVLOG.md`](docs/DEVLOG.md) for the
dated implementation record.

## Reproducible CPU setup

Create a Python 3.11 environment, then install the hardware profile, the UTF-8 third-party lock,
and the checked-out project separately:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements-cpu.lock.txt
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/check_integrity.py
.venv/bin/python scripts/check_env.py
.venv/bin/pre-commit install --install-hooks
```

On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\pre-commit.exe`. Bounded-range
`requirements.txt` and `requirements-extras.txt` are maintenance inputs; the lock/profile files
are the reproducible installation path. CUDA setup and limitations are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data

Download and validate the merged AI4Mars release before any experiment:

```bash
.venv/bin/python scripts/download_data.py --out data/raw/ai4mars --which merged
.venv/bin/python scripts/check_data.py
```

The preflight verifies the configured 16,064 MSL training labels, 322 MSL gold-test labels, 204
MER gold-test labels, image/label pairing, split disjointness, label values, and data fingerprints.
Raw data and model weights remain untracked.

## Current status

The data/model/training/evaluation implementation and CPU smoke artifact exist. Training metrics
are emitted during fitting and persisted with run artifacts. The historical preregistration is
preserved, with a dated Protocol V2 amendment and fail-closed executable snapshot machinery.

The confirmatory multi-seed GPU sweep has not been completed in the committed results, so H0–H5
remain unresolved/deferred. A final Protocol V2 snapshot must be sealed only after the runtime and
model configs stabilize and before confirmatory analysis. Do not treat the committed one-epoch CPU
smoke result as scientific evidence.
