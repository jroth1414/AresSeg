# Reproducibility profiles

The project targets Python 3.11. Select the hardware profile first, install the UTF-8 reference
lock, and install the checked-out project separately so a lock can never replace local source code.

## CPU

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements-cpu.lock.txt
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/check_integrity.py
.venv/bin/python scripts/check_env.py
```

On Windows, replace `.venv/bin/python` with `.venv\\Scripts\\python.exe`.

## CUDA 12.1

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cuda121.lock.txt
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install -r requirements-extras.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/check_integrity.py
.venv/bin/python scripts/check_env.py
```

The CUDA profile must be installed **with dependencies**: PyTorch 2.4.1's wheel metadata pins its
CUDA and Triton requirements except `nvidia-nvjitlink-cu12`, which the profile pins explicitly to
the resolver-observed `12.9.86`. Those platform dependencies are intentionally absent from the
portable main lock. The CPU profile can use `--no-deps` because the main lock supplies its portable
dependencies afterward.

The locks select Python package versions; bitwise GPU reproduction additionally requires identical
hardware, wheels, driver behavior, downloaded model revisions, and caches. DINOv3 and SAM retain
their upstream license/access requirements. The repository currently has no repository-level
license grant; all rights are reserved by default pending an explicit owner choice.

## Data

```bash
.venv/bin/python scripts/download_data.py --out data/raw/ai4mars --which merged
.venv/bin/python scripts/check_data.py
```

The merged AI4Mars archive is approximately 16.2 GB and is verified against its published MD5.
Raw data, model weights, and predictions remain untracked.
