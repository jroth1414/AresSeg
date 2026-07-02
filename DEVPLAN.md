# DEVPLAN — Terrain-Aware Semantic Segmentation for Mars Rover Drivability (AI4Mars)

**Master development plan for the `marsseg` project (JHU EN.705.742, Advanced Applied ML). Author: John Roth.**

This project builds and rigorously evaluates deep semantic-segmentation models that label Martian
terrain — *soil, bedrock, sand, big rock* — from rover camera images, the core perception task for
**autonomous drivability** assessment. We compare CNN (U-Net, DeepLabV3+) vs transformer (SegFormer)
architectures, measure the value of **ImageNet transfer**, test **cross-rover generalization**
(Curiosity → Opportunity/Spirit), and benchmark a **foundation-model reference** (DINOv3 pretrained on
Earth satellite imagery + SAM zero-shot), all under a leakage-safe, pre-registered,
significance-tested protocol.

---

## 0. COLD-START: READ THIS FIRST

**What this is.** A single, self-contained, mechanically executable plan. Any agent with **no prior
conversation context**, given only this file + the repo, must reach the **same results and the same
H0–H5 verdicts** as any other agent. Every load-bearing constant is pinned below. Do not invent
versions, paths, thresholds, or file names.

**ONE-LINE STATUS (2026-07-01): MS0 and MS1 are DONE (MS1 fixed + re-verified against the real
on-disk layout: 16064/322/204 count asserts green; 27 pytest tests green, ruff+black clean); MS2 is
PARTIAL. RESUME POINT → build `models/foundation.py` (MS2 tail), then `configs/*`, `eval/*`, and
`scripts/{run_experiment,analyze_results,merge_results}.py` + `run_gpu.sh` (MS3 → MS4 → MS5).**

> **✅ RESOLVED (2026-07-01, branch `phase-ms1-fix`).** The two MS1 defects that blocked all training
> are **fixed and re-verified against the on-disk dataset**:
> 1. `build_index` is now camera-aware and ncam-scoped (§4.1): MSL ncam train=16064, test=322 (pinned
>    **min3** by path, not sort order), MER train=[], test=204 — asserted by real-layout tests in
>    `tests/test_data.py` (they skip when the dataset is absent so CI stays offline-green).
> 2. Every record carries the canonical camera-qualified `name` join key (§4.3: `msl_ncam_…`,
>    `mer_test_…`; unique, non-empty, sorted for cross-platform determinism), and
>    `SegDataset.__getitem__` returns `rec["name"]` — a missing name is a hard `KeyError`, never a
>    positional fallback.
>
> Training remains gated only on MS3 (`configs/*`, `eval/*`, `run_experiment.py`) being built.

- **Branches.** `main` is the **canonical branch** and contains everything through this hardened plan
  (MS0–MS2 code + the DEVPLAN); `ms2-models` is kept as a mirror at the same commit. **A fresh checkout
  of `main` matches this plan — no branch switch is needed.** Every commit is authored by **John Roth**
  only (no AI co-authors). Branch new phase work off `main` (e.g. `phase-ms3-eval`).
  `project/ntl-sector-etf-forecasting` is an unrelated archived project — ignore it.
- **DONE (do NOT rebuild):**
  - `src/marsseg/utils/*` (seed, config, manifest, results, tracking, logging, capabilities)
  - `src/marsseg/data/{ai4mars,dataset,transforms}.py` — **§4.1 + §4.3 fixes APPLIED 2026-07-01**
    (camera-aware ncam-scoped `build_index`, `label_key` stem normalizer, pinned min3 gold dir,
    camera-qualified `name` join key; `SegDataset` returns `rec["name"]`, no fallback).
  - `src/marsseg/models/zoo.py` (baseline / unet / deeplabv3plus / segformer)
  - `src/marsseg/train/{loss,lit}.py` (PyTorch Lightning)
  - `scripts/{check_env,download_data}.py`
  - `tests/{test_smoke,test_data,test_models}.py` — **27 tests pass**, including real-layout count
    asserts (16064/322/204, min3-by-path, unique camera-qualified names) that run on this box and
    skip offline, plus regression fixtures for the two §4.1 bugs (camera decoy, MER pools/stems)
    and the min1-vs-min3 discriminator.
  - `.git/hooks/commit-msg`, `.gitignore`, `requirements*.txt`, `.env.example`, `pyproject.toml`
  - AI4Mars dataset already downloaded and extracted locally (see §4).
- **REMAINING in MS2:** `src/marsseg/models/foundation.py` (DINOv3-SAT frozen backbone + head; SAM
  zero-shot; skip-and-log if weights/GPU absent).
- **NOT STARTED:** `configs/*` (empty), `src/marsseg/eval/*` (only `__init__.py`),
  `scripts/{run_experiment,analyze_results,merge_results}.py`, `scripts/run_gpu.sh`,
  `experiments/PREREG.md`, the SAM checkpoint fetch, the paper.

**Build trail:** append every phase's decisions/verification to **`docs/DEVLOG.md`** (newest at
bottom). Update §0 here and README §Status at the end of each phase.

---

## 1. Operating rules (NON-NEGOTIABLE — read before your first commit)

1. **NO AI author or co-author anywhere** — commits, code comments, docs, or the paper. Never add
   `Co-Authored-By:`, `Generated with`, or any Claude/Anthropic attribution. A
   `.git/hooks/commit-msg` hook auto-strips any `Co-?Authored-?By: …(Claude|Anthropic)` trailer — do
   **not** re-add them and do not fight the hook. Author identity for all commits is
   **John Roth <jrothecuador@gmail.com>**; do not change `git config user.*`.
2. **RESEARCH.MD is the immutable course rubric — NEVER edit, stage, reformat, or `git add` it.** Its
   frozen SHA-256 is:
   ```
   181361d246cc0f5e2cde7061ff3c4713815aad233fe61aeda4f8a0d496e84e31
   ```
   Re-verify unchanged at the MS5 gate (`sha256sum RESEARCH.MD` must equal the value above).
3. **`.env` is gitignored and NEVER committed.** Secrets (only `HF_TOKEN`, for gated DINOv3) live
   **only** in `.env`. Code reads them via `marsseg.utils.config.require_secret(name)`, which raises
   an actionable error if missing/empty. Only `.env.example` is committed. (Note: the live `.env`
   currently holds stale `EARTHDATA_TOKEN`/`FRED_API_KEY` from a prior project — purge these and add
   `HF_TOKEN` when doing H5; see §3.)
4. **`experiments/` and `data/` are gitignored.** The ONLY experiment files that may be committed:
   `experiments/results_store.parquet`, `experiments/results_store.csv`,
   `experiments/**/manifest.json`, `experiments/PREREG.md`, `experiments/manifests/**`, and
   `.gitkeep`. `data/` keeps only its skeleton via `.gitkeep`. **Never `git add -f`** a checkpoint
   (`*.pt` / `*.pth` / `*.ckpt` / `*.safetensors`), raw data, or predicted-mask PNGs.
5. **Seed = 1414 for splits + training; the evaluation bootstrap uses purpose-seed 0.** Call
   `marsseg.utils.seed.set_seed(1414)` at the top of every entry point (it sets Python/NumPy/torch
   seeds + `cudnn.deterministic=True`, `benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   `torch.use_deterministic_algorithms(warn_only=True)`). The bootstrap in `eval/stats.py` uses
   `marsseg.utils.seed.BOOTSTRAP_SEED = 0` — **not** 1414. `SEED_SET=[1414,1415,1416,1417,1418]` is an
   OPTIONAL multi-seed robustness appendix, not the reported number.
6. **Work task-by-task.** One local commit per task; the message is **prefixed with the phase/task ID**
   (e.g. `MS3: add eval/metrics.py`). One **branch per phase**, branched off `main`
   (e.g. `phase-ms3-eval`). **Commit LOCALLY only — DO NOT push unless the user explicitly asks.**
   **PAUSE at every phase gate for user review** before starting the next phase.
7. **The local RTX 5070 Ti (Blackwell) is intentionally UNUSED for training.** Do not target it. CPU
   is for the full pipeline + smoke; full training runs on the incoming **V100 (16 GB, Ubuntu)**.

---

## 2. Environment & setup

**Interpreter (pinned).** Base Python **3.11** (pyproject requires `>=3.11,<3.12`):
`C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe`. Venv at **`.venv`** (gitignored).

- **Windows/CPU interpreter:** `.venv/Scripts/python.exe`
- **V100/Ubuntu interpreter:** `.venv/bin/python`

**Setup (Windows / CPU — the dev + smoke profile), from repo root, IN ORDER:**
```powershell
# STEP 0: `main` is the canonical branch and already matches this plan (no switch needed).
# If you are on another branch after cloning, check out main first.
git checkout main

C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # CPU torch, smp, lightning, transformers, albumentations
.\.venv\Scripts\python.exe -m pip install -e .                  # editable install of src/marsseg — REQUIRED before ANY import
.\.venv\Scripts\python.exe scripts/check_env.py                 # MUST end with `core OK` (exit 0)
```
The editable install (`pip install -e .`) is **required**: the package lives under `src/` and every
import (`from marsseg…`) and `scripts/check_env.py` assume `marsseg` is importable.

**Expected `check_env.py` output (Windows/CPU):**
```
profile=windows_cpu
cuda=False gpu=None
smp=True transformers=True sam=... timm=...
--- core package versions ---
  numpy: ...
  ... (all core modules present, none MISSING) ...
core OK
```
**The gate is exactly one thing: the last stdout line is `core OK` and exit code is 0.** Nothing else
is asserted. `sam`/`timm` may be `False` on CPU — the core smoke **excludes** gated foundation packages
(`segment-anything`), so their absence never fails the gate.

**Package versions FLOAT — do not treat any specific minor version as frozen.** `requirements.txt` uses
lower bounds with loose upper bounds (`transformers>=4.40` with no cap; `numpy>=1.26,<2.1`;
`torch>=2.2,<2.5`; `torchvision>=0.17,<0.20`; `segmentation-models-pytorch>=0.3.3`; `torchmetrics>=1.3`),
so a fresh `pip install` may resolve e.g. `numpy 2.0.2`, `transformers 5.x`, `torch 2.4.1+cpu`. This is
fine: verdicts do not depend on minor versions. For **byte-exact** reproduction use
**`requirements.lock.txt`** (the frozen full env). **The lockfile is UTF-16-encoded** — read it with
`encoding="utf-16"`, not utf-8. (Observed-good on this box: `torch 2.4.1+cpu`, `torchvision
0.19.1+cpu`, `segmentation-models-pytorch 0.5.0`, `cuda=False`, `profile=windows_cpu` — illustrative,
not a gate.)

**CPU-vs-V100 profile split.** `marsseg.utils.capabilities.detect().profile` returns
`"gpu_full"` if `torch.cuda.is_available()` else `"windows_cpu"`. All run scripts branch on this:
- `windows_cpu` → **smoke path only** (subset of images via `data.max_train_images`, `train.max_epochs=1`,
  limited batches); GPU-only arms are recorded `status="skipped"` (never crash) and appended to
  `manifest.gpu_stages_skipped`.
- `gpu_full` → **full training** (`max_epochs=50`, AMP `precision="16-mixed"`).

**GPU handoff (V100 / Ubuntu) — `scripts/run_gpu.sh` (TO BUILD, MS4).** It is a **bash** script (NOT
PowerShell) that MUST:
1. `git checkout main` (or the merged phase branch) first, then `python3.11 -m venv .venv`.
2. `.venv/bin/python -m pip install -r requirements.txt`, then **reinstall a CUDA torch build**
   because `requirements.txt` resolves to the CPU wheel:
   `.venv/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`.
3. `.venv/bin/python -m pip install -r requirements-extras.txt` (segment-anything, timm — foundation).
4. `.venv/bin/python -m pip install -e .`.
5. **Pre-cache ImageNet encoder weights** (see network precondition below): the smp `resnet34`/`resnet50`
   ImageNet encoders and the SegFormer ADE checkpoints are fetched from the network on first use. Warm
   the cache here (e.g. `python -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34',
   encoder_weights='imagenet'); smp.DeepLabV3Plus('resnet50', encoder_weights='imagenet')"` and the
   transformers SegFormer load) while the box has network, so training arms don't fail offline later.
6. Download the SAM ViT-B checkpoint (§6) into `data/weights/sam/`.
7. `.venv/bin/python scripts/check_env.py` and assert `profile=gpu_full` (cuda True) before training.
8. Run each `configs/models/*.yaml` and the cross-rover (H4) + foundation (H5) arms via
   `scripts/run_experiment.py`.
9. `.venv/bin/python scripts/merge_results.py …` to dedup GPU rows into
   `experiments/results_store.parquet` on `DEDUP_KEYS` (§7).
Extras are installed **only** on the V100 profile.

**Network precondition for pretrained arms.** `build_model('unet'|'deeplabv3plus', pretrained=True)`
and `build_model('segformer', pretrained=True)` **download weights from the internet on first use**
(smp ImageNet encoders; `nvidia/segformer-*-finetuned-ade-512-512` via `transformers`). The
**scratch** arms (`pretrained=False`) need no network. On an **offline/air-gapped V100**, the pretrained
arms fail unless the encoder-weights cache (`~/.cache/torch/hub`, `~/.cache/huggingface`) is
pre-populated — hence step 5 above. Record in the run manifest whether weights came from cache or
network.

**Gated DINOv3 (H5).** See §3 for the token + license steps. DINOv3 itself needs **no extra pip**
(loads via `transformers`, already core); it needs `HF_TOKEN` + license acceptance. SAM needs the
extras install **and** a downloaded ViT-B checkpoint (§6).

---

## 3. Research question & hypotheses (α = 0.10, Holm-corrected within families)

**RQ.** For Mars terrain segmentation, what drives accuracy — architecture (CNN vs transformer),
encoder transfer (ImageNet vs from-scratch), label-noise handling — and does a model generalize
**across rovers/cameras**?

**ML problem.** Multi-class **semantic segmentation**: map a grayscale **Navcam (ncam)** image
`x ∈ ℝ^{H×W}` (grayscale replicated to 3 channels, ImageNet-normalized) to a per-pixel label
`y ∈ {soil, bedrock, sand, big_rock}^{H×W}`, with an **ignore label (255)** for
unlabeled / rover-self / >30 m-range pixels excluded from loss AND metrics. Loss = class-weighted
cross-entropy + Dice on valid pixels; primary metric = **mean IoU (mIoU)**.

> **Camera scope (settles the §4 nesting question).** The AI4Mars `msl/NOTE.txt` states: *ncam =
> navcam, grayscale, the stereo imagery the rover drives on*; *mcam = mastcam, color, science*. The
> ML problem above is defined on the **grayscale navcam**, and only **ncam** has an expert test split.
> **The primary protocol is ncam-only for MSL.** `mcam` (color, train-only, no gold test) is
> **excluded** from all reported numbers. This is a deliberate scope decision, not an oversight.

Hypotheses are frozen in **`configs/hypotheses.yaml`** (owner: `eval/`; TO BUILD, MS3 — see §5 for the
exact schema) and pre-registered in **`experiments/PREREG.md`** (frozen BEFORE any test-set number is
computed). **Significance threshold: α = 0.10; Holm correction applied within each family.** The
paired-bootstrap statistic, pairing, p-value estimator, and fixed-class-set rules are defined **once**
in §5.6 and are the single source of truth for every hypothesis below.

| ID | Statement | Family | Test | Exact decision rule |
|---|---|---|---|---|
| **H0** | No deep model significantly beats the baseline at mIoU. | — | — | Reported **honestly**: H0 holds iff H1 is NOT rejected. H0 is never "tested" as its own comparison. |
| **H1** | ≥1 deep model (U-Net / DeepLab / SegFormer, best pretrained variant) beats `baseline` on mIoU. | A | paired_bootstrap, one-sided (`greater`), §5.6 | **Reject H0 (support H1)** iff Holm-adjusted p < 0.10 for ≥1 Family-A member (Δ = mIoU_candidate − mIoU_baseline > 0). |
| **H2** | ImageNet-pretrained encoder > identical from-scratch. | B | paired_bootstrap, one-sided (`greater`), §5.6 | **Support** iff Holm-adjusted p < 0.10 for ≥1 Family-B member (Δ = mIoU_pretrained − mIoU_scratch > 0). |
| **H3** | SegFormer (transformer) vs U-Net/DeepLab (CNN) — which wins overall **and per class** (hypothesis: transformer favors large homogeneous soil/sand; CNN favors small `big_rock` boundaries). | C | paired_bootstrap, **two-sided**, overall + per-class strata, §5.6 | Delta orientation is **`segformer − cnn`** everywhere. **Support a direction** for a member iff Holm-adjusted p < 0.10; the reported direction is `sign(observed Δ)`. Report per-class (scope∈{soil,bedrock,sand,big_rock}) IoU deltas separately using the same fixed-class + resample rules (§5.6). |
| **H4** | A model generalizes across rovers: train MSL (Curiosity ncam), test MER (Opportunity/Spirit) with a **bounded mIoU drop**. | D | bootstrap on MER-test images, §5.6 + §5.7 | **Deterministic rule, no p-value** (single-member family ⇒ Holm is a no-op). **Generalizes (support)** iff **both**: (1) point-estimate `drop = mIoU(subject, MSL ncam test) − mIoU(subject, MER test) < 0.15`, AND (2) `cross_rover_ci_low > baseline_on_MER_miou` (point estimate). See §5.7 for the exact mechanics. |
| **H5** | Foundation reference: DINOv3 ViT-L/16 SAT-493M frozen backbone + trained head; SAM zero-shot. | E | paired_bootstrap vs baseline (§5.6); gated | Decided on `gpu_full`. On `windows_cpu`/missing weights → verdict = **DEFERRED (gated, awaiting gpu_full)**; MUST NOT block H1–H4. On gpu_full: per-member Holm within Family E; **see §5.8 for the partial-GPU (mixed ok/skipped) rule.** |

**H5 gated-weights steps (do only when running H5):**
1. Visit `https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m` while logged in and
   **accept the license**. Only the **ViT-L/16 SAT-493M** variant fits the 16 GB V100 (ViT-7B does not).
2. Create an HF **read** token.
3. Put it in `.env` as `HF_TOKEN=…` (gitignored, never committed).
4. Code reads it via `config.require_secret("HF_TOKEN")` **only on the load path** — but
   `foundation.py` must FIRST check `os.environ.get("HF_TOKEN")` and, if absent (or cuda absent, or
   SAM checkpoint absent), **skip-and-log** (append a `status="skipped"` results row, record in
   `manifest.gpu_stages_skipped`, return) rather than raise. Never hard-crash the pipeline on CPU.
5. **MS0 fix task — DONE 2026-07-01:** `.env.example` rewritten (no false "NO credentials required"
   framing; `HF_TOKEN` marked **required-for-H5** with the license-acceptance URL); stale
   `EARTHDATA_TOKEN`/`FRED_API_KEY` purged from the live `.env` (now holds an empty `HF_TOKEN`
   placeholder). The stale **DINOv2→DINOv3** naming in `.env.example`, `README.md`,
   `requirements.txt`, `capabilities.py`, and the proposal was also corrected.

---

## 4. Data acquisition & layout (AI4Mars)

**Source.** NASA AI4Mars (open), Zenodo record **15995036**
(base URL `https://zenodo.org/records/15995036/files`). Two archives:

| which | filename | MD5 | size |
|---|---|---|---|
| merged (**default**) | `ai4mars-dataset-merged-0.6.zip` | `daf80a86021253292e6c425f97baa5c6` | ~16.2 GB |
| unmerged | `ai4mars-labels-unmerged.zip` | `49fc7a969dfddc0c06d0020edda432c2` | ~1.6 GB |

**Acquisition (already downloaded + extracted locally — do NOT re-download):**
```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --out data/raw/ai4mars --which merged
# flags: [--which merged|unmerged|both] [--no-extract] [--no-md5]
```

**Extracted layout (VERIFIED on disk).** The merged zip extracts into a **nested** directory
`data/raw/ai4mars/ai4mars-dataset-merged-0.6/`, and MSL/MER are further **nested per-camera**. This is
the ground truth an implementation must target (subdir names and counts below are confirmed present):
```
data/raw/ai4mars/ai4mars-dataset-merged-0.6/
  info.md  changelog.md  label_keys.json  TODO.md
  msl/            (Curiosity)          NOTE.txt: ncam=navcam grayscale(drive); mcam=mastcam color(science)
    ncam/                              # PRIMARY (grayscale navcam) — used for ALL reported numbers
      images/  edr/*.JPG               (18,127 JPGs)
               mxy/  rng-30m/          (aux; unused)
      labels/
        train/*.png                    (16,064 train labels; stem == image stem)
        test/
          masked-gold-min1-100agree/*.png   (322)
          masked-gold-min2-100agree/*.png   (322)
          masked-gold-min3-100agree/*.png   (322)   <-- PINNED test set
    mcam/                              # color mastcam — EXCLUDED from the protocol (see §3 scope)
      images/*.JPG                     (9,099 JPGs; images directly under images/, no edr/ subdir)
      labels/train/*.png               (9,099; NO test/ split exists for mcam)
  mer/            (Opportunity/Spirit) # cross-rover TEST ONLY (no train labels present)
    images/
      eff/*.JPG                        (16,300 — full MER image pool)
      test/*.JPG                       (204 — the gold-test subset; ALSO present in eff/)
    labels/
      train/                           (EMPTY: 0 pngs — MER has no train labels here)
      test/
        masked-gold-min1-100agree/*.png   (204)
        masked-gold-min2-100agree/*.png   (204)
        masked-gold-min3-100agree/*.png   (204)   <-- PINNED MER test set (H4)
        raw_unmerged/
  m2020/          (Perseverance — unused here)
```
- Labels are single-channel PNGs, pixel values **0–3** (soil/bedrock/sand/big_rock), **255 = NULL/ignore**.
- **Reported dataset = MSL ncam only:** 16,064 train + **322 expert-labeled** gold test images.
- **MER = cross-rover test only** (H4): 204 gold-test images; MER has **no train labels** in this
  release, so it is never trained on — only evaluated with an MSL-trained checkpoint.

### 4.1 build_index camera/pairing fix — **FIXED 2026-07-01** (spec retained for the record)

**The pre-fix `src/marsseg/data/ai4mars.py::build_index` produced an EMPTY MSL training index and an
EMPTY MER index.** The fix below is applied and re-verified (16064/322/204 asserts green). Two
independent bugs:

**Bug A — camera crossing (MSL train = 0).** `build_index(root, "msl")` calls `find_dir(rdir,
"images/edr","images","edr")` and `find_dir(rdir, "labels/train","train")` at the `msl/` level. MSL has
**no** direct `images/`/`labels/` — only `msl/ncam/…` and `msl/mcam/…`. `find_dir`'s `rglob` resolves
the image dir to `msl/ncam/images/edr` but resolves the train-label dir to `msl/mcam/labels/train`
(**`mcam` sorts before `ncam`**). Images come from ncam, labels from mcam ⇒ `_match_labels` finds **0
stem matches ⇒ 0 train pairs**. (The test set coincidentally resolves to ncam and yields 322 — but see
Bug C.) A cold agent following the old plan would believe the dataset is staged when the training index
is empty.

**Bug B — MER image subpath + label-stem shape (MER train = 0 AND test = 0).** `find_dir(rdir,"images/edr",
"images","edr")` resolves the MER image dir to `mer/images` (a dir with **no JPGs directly in it**;
the JPGs live under `mer/images/eff/` and `mer/images/test/`). So `_match_labels` globs `mer/images/*.JPG`
(empty) and pairs nothing. **Even after fixing the image dir**, MER gold labels are named
`<imgstem>_<digits>_T<digits>_merged.png` (e.g. image `1n129697839eff0338p1931l0m1.JPG` ↔ label
`1n129697839eff0338p1931l0m1_16165_T0_merged.png`), and the current `_match_labels` fallback only strips
`_merged`/`_label` — it does **not** strip the intervening `_16165_T0` tokens, so it still matches 0.

**Required fix (verified to recover exact counts).** Rewrite `build_index` to be **camera-aware and
ncam-scoped by default**, pairing images and labels **within the same camera subtree**, and add a
robust label→image stem normalizer:

1. **Iterate camera subtrees, not the rover root.** For `rover="msl"`, use `camera="ncam"` (default);
   image dir = `msl/ncam/images/edr`, train labels = `msl/ncam/labels/train`, test labels =
   `msl/ncam/labels/test/<gold_dir>`. Optionally accept `camera` as a param, but **only ncam is in the
   protocol.** Do NOT union ncam+mcam for reported numbers.
2. **For `rover="mer"`, search images under BOTH `mer/images/eff` and `mer/images/test`** (union the two
   into the image lookup), and read gold labels from `mer/labels/test/<gold_dir>`. MER has **no train
   split** (`out["train"] == []` is expected and correct).
3. **Add a `label_key(stem)` normalizer used by `_match_labels`** that recovers the image stem across
   **all four** naming shapes present on disk (verified: it recovers 16,064 / 322 / 9,099 / 204):
   - strip a trailing `_merged` or `_label`;
   - then repeatedly strip a trailing `_<digits>` **or** `_T<digits>` token.
   Example: `…_16165_T0_merged` → `…_16165_T0` → `…_16165` → `…` (the image id). Match on both the raw
   stem and `label_key(stem)`.
4. **Re-verify counts as an assertion (part of MS1 "DONE"):**
   `len(build_index(DATA_ROOT,"msl")["train"]) == 16064`,
   `len(build_index(DATA_ROOT,"msl")["test"]) == 322` (against the **pinned** gold dir — see Bug C),
   `len(build_index(DATA_ROOT,"mer")["test"]) == 204`, `build_index(DATA_ROOT,"mer")["train"] == []`.
   Add these to `tests/test_data.py` behind a marker that skips when `DATA_ROOT` is absent (so CI stays
   offline-green) but runs on this box.

### 4.2 DATA_ROOT

`DATA_ROOT = data/raw/ai4mars/ai4mars-dataset-merged-0.6` (the nested extracted dir). `configs/data.yaml`
sets `data.root` to this path (§5.5). `data/` is gitignored (skeleton via `.gitkeep`).

### 4.3 Canonical image `name` (join key) — **FIXED 2026-07-01** (spec retained for the record)

The paired bootstrap and per-image tables join rows **across models** on `name`, so `name` must be
**stable, unique within a split, and identical across models**. **Today `build_index` records carry no
`name`, and `SegDataset.__getitem__` falls back to `str(i)` (the DataLoader index) — unusable.** Every
artifact in §7.4 (`preds/<name>.png`, `per_image.parquet`, the cross-model join) depends on this, so it
is a **hard prerequisite of the first run, NOT deferred hardening**.

**Fix:** `build_index` sets, for each record,
```
rec["name"] = f"{rover}_{camera}_{label_key(Path(rec['image']).stem).lower()}"
```
i.e. **camera-qualified** (`msl_ncam_…`, `mer_test_…`) so a stem can never collide across cameras/pools.
(Verified: ncam and mcam stems have zero overlap today, but the camera prefix makes uniqueness
structural, not incidental.) The **same recipe MUST be applied identically in `build_index` for every
model** so cross-model joins match. `SegDataset.__getitem__` returns `rec["name"]` (no `str(i)`
fallback). `run_experiment.py` asserts, per split: names are **non-empty, unique**
(`len(set(names))==len(names)`), and — when comparing two runs — **identical name sets** (raise
otherwise). Add a test asserting all `build_index` records carry a non-empty unique `name`.

### 4.4 Index + split API (BUILT, MS1 — with the §4.1/§4.3 fixes)

- `marsseg.data.ai4mars.build_index(root, rover="msl", camera="ncam") -> {"train":[{image,label,name}…],
  "test":[…]}` (post-fix). For MER, `train == []` and `test` = 204 gold pairs.
- `marsseg.data.dataset.make_splits(records, val_frac=0.2, seed=1414) -> {"train":…, "val":…}` splits
  **BY IMAGE** (a frame never crosses splits).
- `marsseg.data.dataset.class_pixel_counts(records, num_classes=4, max_images=None)` — per-class pixel
  counts, ignores 255. **NOTE:** this `max_images` is a *label-scan* cap for class-weight computation;
  it is **NOT** the training-subset control. The CPU-smoke subset control is a **separate** key
  `data.max_train_images` consumed by `run_experiment.py` (§5.5 / §8) — do not conflate them.
- `marsseg.data.dataset.SegDataset` item = `{"image": (3,H,W) float32, "mask": (H,W) int64 with
  255=ignore, "name": str (camera-qualified, §4.3), "rover": str}`.

---

## 5. Method & frozen protocol constants

**Trainer = PyTorch Lightning (ADOPTED).** `lightning>=2.2,<2.6` + `torchmetrics>=1.3`. There is **no
`train/trainer.py`** and **no raw training loop**. `SegLitModule` in `train/lit.py` **is** the trainer;
Lightning's `Trainer` provides the loop/AMP/DDP/checkpoint/early-stop. `run_experiment.py` wires
`SegLitModule` + `SegDataModule` + `L.Trainer` (callbacks below). **Do not** create
`models/{base,unet,smp_models,segformer,registry}.py` — the entire registry is the single file
`models/zoo.py`, entry point `build_model(name, num_classes=4, backbone=None, pretrained=True)`.

### 5.1 Model registry (`models/zoo.py`, BUILT)

`build_model(...) -> nn.Module` mapping `(B,3,H,W)` → **`(B,4,H,W)` logits at input resolution**:

| `model` id | class / lib | default backbone | pretrained source | params |
|---|---|---|---|---|
| `baseline` | `TinyUNet` (from scratch, `base=16`) | — (`none`) | never (H0/H1 yardstick) | ~117 k |
| `unet` | smp `Unet` | `resnet34` | `encoder_weights="imagenet"` if pretrained (network on first use) | 24.4 M |
| `deeplabv3plus` | smp `DeepLabV3Plus` | `resnet50` | `encoder_weights="imagenet"` if pretrained (network on first use) | 26.7 M |
| `segformer` | `transformers` MiT (`_SegFormer`) | `"b0"` \| `"b2"` | pretrained loads `nvidia/segformer-{b}-finetuned-ade-512-512` (network), upsamples logits to input res | b0: 3.7 M |

`pretrained=False` builds the scratch variant (random init) for H2 and needs **no network**. See §2
"Network precondition" for the pretrained arms. Foundation models (`dinov3_sat`, `sam`) live in
`models/foundation.py` (TO BUILD).

**Backbone naming contract (results-store label).** `build_model` takes SegFormer backbone as `"b0"`/`"b2"`,
but the results-store `backbone` column uses **`mit-b0`/`mit-b2`**. Each `configs/models/*.yaml` carries
both `model.backbone` (the zoo build arg, e.g. `b0`) and `model.results_backbone` (the store label,
e.g. `mit-b0`); `run_experiment.py` writes `results_backbone` into the store. For smp models
`backbone == results_backbone` (`resnet34`, `resnet50`, `efficientnet-b0`); for `baseline`, `backbone=none`.

### 5.2 Lightning module (`train/lit.py`, BUILT) — frozen hyperparameters

`SegLitModule` hparams (all pinned in code): `model_name`, `num_classes=4`, `backbone`,
`pretrained`, `class_weights`, **`lr=3e-4`**, **`weight_decay=1e-4`**, `dice_weight=1.0`,
`ignore_index=255`, `max_epochs=50`.
- Optimizer: **`AdamW(lr=3e-4, weight_decay=1e-4)`**; scheduler **`CosineAnnealingLR(T_max=max_epochs)`**.
- Metrics: `torchmetrics.MulticlassJaccardIndex(ignore_index=255)`, per-class + macro; logs
  `train_loss`, `val_loss`, `val_miou`, `val_iou_{class}`.

`SegDataModule` (BUILT): `SegDataModule(train_records, val_records, test_records=None, batch_size=8,
num_workers=0, size=512)` — wraps `SegDataset`, deterministic loaders (`torch.Generator().manual_seed(1414)`).

### 5.3 Loss (`train/loss.py`, BUILT)

`CombinedLoss(class_weights, ignore_index=255, dice_weight=1.0) = CE(weight, ignore_index) +
dice_weight * DiceLoss(ignore_index=255)`, both over **valid pixels only** (255 never contributes).

### 5.4 EVERY protocol knob frozen to a value

| Knob | Frozen value | Where |
|---|---|---|
| Camera / rover scope | **MSL `ncam` only** for all reported numbers; `mcam` excluded; `mer` = cross-rover test only | `build_index(camera="ncam")` / §3 |
| Image size | **512×512** (`Resize(512,512)`, train + eval) | `configs/data.yaml: data.size` / `SegDataModule(size=512)` |
| Split | **by image**, `val_frac=0.2`, `split_seed=1414` | `make_splits(val_frac=0.2, seed=1414)` |
| Test set (MSL) | **`msl/ncam/labels/test/masked-gold-min3-100agree`** expert gold; assert `len(index["test"]) == 322` | `configs/data.yaml: data.test_gold_dir` |
| Test set (MER, H4) | **`mer/labels/test/masked-gold-min3-100agree`**; assert `len(index["test"]) == 204`; images from `mer/images/{eff,test}` | `configs/data.yaml: mer.test_gold_dir` |
| Gold-dir selection code fix | Current `build_index` hard-picks `sorted(glob("masked-gold-*"))[0]` = **min1** (comment: "most permissive"). The pinned protocol is **min3**. The 322/204 counts are **identical across min1/min2/min3**, so a green 322 does **NOT** prove min3 is in use — it is coincidental. **MS1 fix: `build_index` MUST honor `test_gold_dir` (min3), not `[0]`.** Record the resolved gold dir + count in `manifest.extra.resolved_test_gold_dir`. | `build_index` / `configs/data.yaml` |
| Epochs | `max_epochs=50` (= cosine `T_max`) | `SegLitModule` / `configs/models/*.yaml: train.max_epochs` |
| Early stop | `EarlyStopping(monitor="val_miou", mode="max", patience=10, min_delta=0.001)` | `run_experiment.py` callback |
| Checkpoint | `ModelCheckpoint(monitor="val_miou", mode="max", save_top_k=1)`; **evaluate the BEST ckpt** (not last) | `run_experiment.py` callback |
| Grad clip | `gradient_clip_val=1.0` | `L.Trainer` |
| Precision | `"16-mixed"` on `gpu_full`, `"32-true"` on `windows_cpu` (affects speed only, not the verdict) | `run_experiment.py` |
| LR / opt / sched | `AdamW(3e-4, wd=1e-4)` + `CosineAnnealingLR(T_max=50)` | `train/lit.py` (frozen) |
| Class weights | **`w_c = median(counts)/counts_c`, clipped to `[0.5, 10.0]`**, computed on the **train split only** (post-`make_splits`, never val/test), `max_images=null` (full scan); record the 4-vector in `manifest.extra.class_weights` | `configs/data.yaml: class_weights` |
| Augmentation | train: `Resize(512)` + `HorizontalFlip(p=0.5)` + `RandomBrightnessContrast(0.2,0.2,p=0.3)`; eval: `Resize(512)`. **NO vertical flip, NO scale/crop.** `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` | `data/transforms.py` (frozen) |
| Batch size | **8** | `SegDataModule(batch_size=8)` |
| CPU-smoke subset | `data.max_train_images` (int or null) — **consumed by `run_experiment.py`**, which truncates `train_records` to the first N **after** `make_splits` and **before** building the DataModule (see §8). `null` = full data. This key is NOT read by any BUILT function; wiring it is an MS3 task. | `configs/data.yaml` / `run_experiment.py` |
| Bootstrap | `n_resamples=10000`, `seed=BOOTSTRAP_SEED=0`, unit = **image**, resample WITH replacement, recompute split-level metric from **summed inter/union counts**; CI = **percentile**, `ci_level=0.90`. **Full mechanics: §5.6.** | `eval/stats.py` / `configs/hypotheses.yaml: stats` |
| McNemar | unit = pixel, valid pixels only (`mask!=255`), Edwards continuity `(|b−c|−1)²/(b+c)`; **secondary/descriptive only — never overrides the image-level bootstrap verdict** | `eval/stats.py` |
| H4 mechanics | **§5.7** (single procedure, no p-value) | `configs/hypotheses.yaml: H4` |
| Foundation gating on CPU | `windows_cpu` OR `caps.sam False` OR `HF_TOKEN` missing OR SAM ckpt absent → append `status="skipped"`, `value=None` rows for `{dinov3_sat, finetuned}` and `{sam, zeroshot}`; set `manifest.gpu_stages_skipped`; never raise. **Partial-GPU rule: §5.8.** | `models/foundation.py` / `run_experiment.py` |
| Descriptive-only metrics | `boundary_f1` and `pixel_acc` are **leaderboard/reporting only**; **NO hypothesis is tested on them.** All bootstraps operate solely on `iou`-derived macro-mIoU (H1/H2/H4) and per-class `iou` (H3). | §5.6 / §6 / §7.4 |
| Seed policy | training/splits = **1414** (single-seed primary leaderboard); bootstrap = **0**; multi-seed `[1414..1418]` optional appendix | `utils/seed.py` |

### 5.5 `configs/` schema (TO BUILD, MS3 — none exist yet)

Loaded via `utils.config.load_config(yaml_path, overrides, base_paths)` with dotted `key=value`
overrides (auto-typed; e.g. `train.lr=2e-4` → float, `data.max_train_images=64` → int).

- **`configs/data.yaml`** (paths are relative to repo root; gold dirs include the **camera segment**):
  ```yaml
  data:
    root: data/raw/ai4mars/ai4mars-dataset-merged-0.6
    rover: msl
    camera: ncam                 # PRIMARY grayscale navcam; mcam is excluded
    val_frac: 0.2
    size: 512
    seed: 1414
    split_by: image
    split_seed: 1414
    test_gold_dir: msl/ncam/labels/test/masked-gold-min3-100agree   # exact on-disk path incl. camera
    expected_test_n: 322
    max_train_images: null       # CPU-smoke subset cap; consumed by run_experiment.py (§8); null = full
  mer:
    rover: mer
    image_dirs: [mer/images/eff, mer/images/test]   # union; JPGs are NOT directly under mer/images/
    test_gold_dir: mer/labels/test/masked-gold-min3-100agree
    expected_test_n: 204
    has_train: false             # MER has no train labels in this release
  class_weights:
    method: inverse_freq_normalized
    formula: "w_c = median(counts)/counts_c"
    clip: [0.5, 10.0]
    computed_on: train_split
    max_images: null             # label-scan cap for class_pixel_counts; NOT the training subset
  aug: {hflip_p: 0.5, brightness_limit: 0.2, contrast_limit: 0.2, rbc_p: 0.3,
        vflip: false, scale_crop: false}
  ```
- **`configs/models/<name>.yaml`**:
  ```yaml
  model: {name: unet, backbone: resnet34, results_backbone: resnet34, pretrained: true}
  train: {batch_size: 8, max_epochs: 50, lr: 3e-4, weight_decay: 1e-4, dice_weight: 1.0,
          ignore_index: 255, early_stop_patience: 10, grad_clip: 1.0}
  ```
  Enumerated model configs: `baseline` (no backbone); `unet` (resnet34) {pretrained, scratch};
  `deeplabv3plus` (resnet50) {pretrained, scratch}; `segformer` (b0→mit-b0, b2→mit-b2)
  {pretrained, scratch}; optional `unet` (efficientnet-b0) for H2/H3.
- **`configs/hypotheses.yaml`** (owner: `eval/`) — **every threshold/path/count below is CONCRETE (no
  placeholders); `test_eval.py` asserts this (§9).**
  ```yaml
  alpha: 0.10
  correction: holm
  ci_level: 0.90
  primary_metric: miou
  per_class_metric: iou
  descriptive_only: [pixel_acc, boundary_f1]   # NEVER tested
  stats:
    n_resamples: 10000
    resampling_unit: image
    ci_method: percentile
    bootstrap_seed: 0
    rng: numpy_default_rng          # np.random.default_rng(0); one draw advanced per replicate (§5.6)
    seed_reset_per_comparison: true # each comparison re-seeds default_rng(0); deterministic + independent
    p_estimator: plus_one           # p = (1 + #{...}) / (n_resamples + 1)  (§5.6)
    fixed_class_set: full_split_present   # classes with union>0 over the FULL split, fixed before bootstrap
    empty_class_in_resample: iou_zero     # a fixed-set class with union==0 in a resample contributes IoU=0
    mcnemar: {unit: pixel, scope: valid_pixels_only, correction: continuity, report: secondary_only}
  canonical_run_selection:            # how a comparison id resolves to ONE run_id (§5.9)
    filter: {seed: 1414, profile: gpu_full, status: ok}
    tie_break: max_manifest_timestamp_utc   # if >1 remain for a (model,backbone,variant,stratum), take newest; if still tied, raise
  families:
    A: {members: [baseline_vs_unet, baseline_vs_deeplabv3plus, baseline_vs_segformer]}
    B: {members: [unet_pretrained_vs_scratch, deeplabv3plus_pretrained_vs_scratch, segformer_pretrained_vs_scratch]}
    C: {members: [segformer_vs_unet, segformer_vs_deeplabv3plus]}   # delta orientation: segformer - cnn
    D: {members: [best_in_rover_vs_cross_rover]}
    E: {members: [dinov3_sat_vs_baseline, sam_zeroshot_vs_baseline]}
  hypotheses:
    H1: {family: A, test: paired_bootstrap, statistic: delta_miou, tail: greater,
         metric: miou, decision_rule: "reject_H0 if holm_p < 0.10 for >=1 member (delta>0)"}
    H2: {family: B, test: paired_bootstrap, statistic: delta_miou, tail: greater, metric: miou,
         decision_rule: "support if holm_p < 0.10 for >=1 member (delta>0)"}
    H3: {family: C, test: paired_bootstrap, tail: two_sided, strata: [all, per_class], metric: miou,
         delta_orientation: "segformer_minus_cnn", per_class_metric: iou,
         decision_rule: "support a direction (sign(observed delta)) iff holm_p < 0.10 for that member"}
    H4: {family: D, test: threshold_plus_ci, statistic: miou_drop, drop_threshold: 0.15,
         emits_p_value: false, decision_rule: "support iff drop < 0.15 AND cross_rover_ci_low > baseline_on_MER_miou"}
    H5: {family: E, gated: true, decided_on_profile: gpu_full, on_missing_gpu: deferred,
         partial_gpu_rule: "holm over ok members only; support iff >=1 ok member holm_p<0.10 (delta>0); reject if all ok fail; deferred if zero ok members"}
  ```
  Each comparison id resolves to exactly **two concrete `run_id`s** via §5.9, then loads each run's
  `per_image.parquet` for the paired bootstrap. H0 has no config entry — it is reported honestly from H1.
  **The `tail: greater` / `statistic: "miou_in_rover - miou_cross_rover"` bootstrap-significance framing
  that older drafts had under H4 is DELETED** — H4 is a deterministic threshold+CI rule with no p-value
  (§5.7).

### 5.6 Paired-bootstrap statistic, pairing, and p-value — THE single source of truth

This procedure defines H1, H2, H3, and H5. (H4 uses §5.7.) `eval/stats.py::paired_bootstrap` MUST
implement exactly this; two independent implementations of this spec must produce identical numbers.

1. **Fixed class set.** Before bootstrapping, compute the split-level per-class union over ALL test
   images. The macro mean is taken over the **fixed set `S` = {classes with total union > 0 over the
   full split}** (computed once). This set does NOT change between replicates.
2. **RNG.** `rng = numpy.random.default_rng(0)` (i.e. `BOOTSTRAP_SEED=0`), **re-seeded to `default_rng(0)`
   at the start of each comparison** (`seed_reset_per_comparison: true`) so every comparison is
   deterministic and order-independent.
3. **Per replicate (10,000 total):** draw **one** bootstrap sample of image names —
   `idx = rng.integers(0, N, size=N)` where `N` = #images in the split (with replacement). **The SAME
   `idx` (same name multiset) is applied to BOTH models' per-image rows** (this is what makes it
   *paired*; never resample the two models independently).
4. **Recompute each model's macro-mIoU on that resample from SUMMED counts:** for each class `c ∈ S`,
   `sum_inter_c = Σ_over_sampled_images inter_c`, `sum_union_c = Σ union_c` (with multiplicity). Per-class
   IoU `= sum_inter_c / sum_union_c`; if `sum_union_c == 0` in this resample, that class contributes
   **IoU = 0** (`empty_class_in_resample: iou_zero`) — the macro denominator stays `|S|`, fixed.
   `macro_mIoU = mean over S`.
5. **Delta per replicate:** `delta_b = mIoU_candidate_b − mIoU_baseline_b` (orientation is
   candidate−baseline for A/B/E; **segformer−cnn** for C, per §5.5).
6. **p-value (plus-one estimator).**
   - one-sided `greater` (A/B/E): `p = (1 + #{delta_b <= 0}) / (n_resamples + 1)`.
   - two-sided (C): `p = 2 * min( (1+#{delta_b<=0}), (1+#{delta_b>=0}) ) / (n_resamples + 1)`, clipped to 1.
7. **CI:** percentile CI of the `delta_b` distribution at `ci_level=0.90` → `[5th, 95th]` percentiles.
   (The reported point `delta` is the observed split-level delta, not the bootstrap mean.)
8. **Per-class strata (H3):** identical procedure with `metric=iou` on a single class `c`; **reuse the
   SAME resample indices `idx` per replicate as the overall run** (draw once per replicate, reuse across
   the overall + per-class statistics of that comparison) so strata are mutually consistent.
9. **Holm within family:** collect each family's member p-values, apply Holm at α=0.10, compare
   `holm_p < 0.10`.

### 5.7 H4 mechanics (single procedure — no p-value)

- **Subject** = the single in-rover model with the highest `val_miou` among the trained MSL models;
  **reuse its H1 checkpoint (no retraining).**
- Evaluate the subject on: (i) the **MSL ncam gold test** (322) → `mIoU_in_rover` point estimate; and
  (ii) the **MER gold test** (204) → `mIoU_cross_rover` point estimate.
- `drop = mIoU_in_rover − mIoU_cross_rover` (**point estimate**, no bootstrap on the drop).
- Bootstrap the **MER-test images only** (`default_rng(0)`, n=10000, resample the 204 images with
  replacement, recompute macro-mIoU from summed counts over the fixed class set of the MER split) →
  90% percentile CI on `mIoU_cross_rover` → `cross_rover_ci_low` (5th pct).
- `baseline_on_MER_miou` = the `baseline` model's MER-test macro-mIoU **point estimate** (single scalar,
  not CI-bounded).
- **SUPPORT H4 iff `drop < 0.15` AND `cross_rover_ci_low > baseline_on_MER_miou`.** No p-value; Family D
  has one member so Holm is a no-op. This rule is stated identically in §3, §5.5, §5.7, and §10.

### 5.8 H5 partial-GPU (Family E mixed ok/skipped) rule

Family E has two members (`dinov3_sat_vs_baseline`, `sam_zeroshot_vs_baseline`). On `windows_cpu` both
are `status="skipped"` ⇒ **H5 = DEFERRED** (never blocks H1–H4). On a run where only some members
completed (e.g. SAM ran but DINOv3 skipped):
- **Holm runs over ONLY the members with `status=="ok"`.** Skipped members are reported
  `verdict=deferred` individually.
- **H5 overall = support** iff ≥1 ok member has `holm_p < 0.10` (Δ>0); **reject** if all ok members fail;
  **deferred** if there are zero ok members.

### 5.9 Canonical-run selection (comparison id → one run_id)

The results store may contain multiple runs for a `(model, backbone, variant, stratum)` tuple
(re-runs, the optional multi-seed appendix `[1414..1418]`, CPU-smoke rows). `verdict.py` MUST resolve
each comparison member to **exactly one** run_id deterministically:
1. Filter rows to `seed == 1414` **AND** `profile == "gpu_full"` **AND** `status == "ok"`.
2. If >1 row remains for a `(model, backbone, variant, stratum)` tuple, take the run whose
   `manifest.timestamp_utc` is newest; **if still tied, raise** (do not guess).
3. Load `experiments/manifests/<run_id>/per_image.parquet` for the two resolved run_ids and run §5.6.

---

## 6. Metric definitions (precise, with ignore handling)

All metrics computed in `eval/metrics.py` (TO BUILD, MS3). **`ignore_index=255` pixels are excluded
from every numerator and denominator.** The `value` stored for `miou`/`iou`/`pixel_acc` uses the
**split-level** formulas below; the per-image table (§7) stores **integer counts** so the bootstrap
recomputes these same split-level formulas on each resample (§5.6).

- **Per-class IoU** = `(Σ_images intersection_c) / (Σ_images union_c)`, i.e. **micro over images, then
  per class** (accumulate confusion counts over the split, not per-image ratios). Classes averaged in
  the macro mean = the **fixed set `S`** (§5.6, step 1): classes with total split-level union > 0. A
  class not in `S` is **excluded** from the macro mean (do NOT count it as 0).
- **mIoU** = **MACRO** over `S` of the split-level per-class IoU above (matches
  `MulticlassJaccardIndex(average="macro")` used for `val_miou`).
- **pixel_acc** = `(Σ correct non-ignore pixels) / (Σ non-ignore pixels)` over the split. **Descriptive
  only** — no hypothesis is tested on it.
- **boundary_F1** = mean over `S` of the F1 between predicted and GT class boundaries within a
  **tolerance of 3 px** (Euclidean, via `scipy.ndimage.distance_transform_edt` / `skimage`). Boundaries
  are computed on each class's binary mask excluding ignore; ignore pixels contribute to neither
  precision nor recall. **Descriptive only** — it has no summable (inter, union) decomposition, so it is
  **NEVER bootstrapped or tested**; its per-image `value` is stored for reporting and its `inter`/`union`
  columns are 0 (§7.4).

**SAM checkpoint (H5, GPU only).** Download `sam_vit_b_01ec64.pth` from
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` into `data/weights/sam/` and
reference it via config key `model.sam_checkpoint`. Checkpoints are gitignored (`*.pth`). If the SAM
import OR the checkpoint is absent, `foundation.py` skip-and-logs (`status="skipped"`).

---

## 7. Artifact contracts (authoritative)

### 7.1 Results-store contract (`utils/results.py`, BUILT) — `experiments/results_store.parquet` (+ `.csv`)

Long/tidy, **one metric value per row**. Append **only** via `utils.results.append_results(rows)`
(fills missing columns with `None`, reorders to `RESULT_COLUMNS`, writes both parquet + csv).

`RESULT_COLUMNS = [run_id, model, backbone, variant, scope, stratum, metric, value, ci_low, ci_high,
status, profile, seed, git_sha, config_hash]`. Allowed values:

| column | allowed values |
|---|---|
| `model` | `baseline, unet, deeplabv3plus, segformer, dinov3_sat, sam` |
| `backbone` | `resnet34, efficientnet-b0, resnet50, mit-b0, mit-b2, vitl16-sat493m, vit-b, none` |
| `variant` | `pretrained, scratch, zeroshot, finetuned` |
| `scope` | `ALL` (overall) or a class name `soil, bedrock, sand, big_rock` |
| `stratum` | `all, per_class, in_rover, cross_rover, pretrained, scratch` |
| `metric` | `miou, iou, pixel_acc, boundary_f1, n` |
| `status` | `ok, skipped, failed` |
| `profile` | `windows_cpu, gpu_full` |

`ci_low`/`ci_high` are populated **only** by the analyze/bootstrap step (`None` on raw run rows).
**Use the exact strings** (e.g. `deeplabv3plus` not `deeplab`; `cross_rover` not `crossrover`; `miou`
not `mIoU`), or `DEDUP_KEYS` will fail to dedup and analysis filters will miss.

`DEDUP_KEYS = [run_id, model, backbone, variant, scope, stratum, metric, config_hash]` — used by
`merge_results.py` to dedup GPU rows. (Note: because `run_id` is in `DEDUP_KEYS`, the results store may
legitimately hold multiple runs per `(model,variant)`; picking *the* canonical run for a verdict uses
§5.9, not `DEDUP_KEYS`.)

### 7.2 Manifest contract (`utils/manifest.py`, BUILT) — `experiments/<run_id>/manifest.json`

Write **only** via `utils.manifest.write_manifest(run_dir, config, seed, *, profile, model, backbone,
variant, dataset, data_hashes, stages_completed, gpu_stages_skipped, **extra)`. Superset fields:
`run_id (= run_dir.name), timestamp_utc, git_sha, git_dirty, python, platform, seed, profile,
capabilities{}, packages{}, config, config_hash (sha256 of resolved config — the dedup key), model,
backbone, variant, dataset, data_hashes, stages_completed, gpu_stages_skipped, **extra`.
`run_experiment.py` MUST set `model/backbone/variant/dataset`, populate
`stages_completed`/`gpu_stages_skipped` so `status="skipped"` rows are reconstructable, and record
`extra.class_weights` and **`extra.resolved_test_gold_dir`** (the actual gold dir + count used, so the
min1-vs-min3 selection is auditable) and **`extra.weights_source`** (`cache`|`network`|`scratch`).
Fixed constants: `CLASSES=["soil","bedrock","sand","big_rock"]`, `NUM_CLASSES=4`, `IGNORE_INDEX=255`.
`verdict.py` reads `timestamp_utc` for the §5.9 tie-break.

### 7.3 `run_id` convention & run-dir layout

```
run_id = f"{model}__{variant}__{backbone_or_none}__{split_scope}__seed{seed}__{profile}__{config_hash[:8]}"
# e.g. unet__pretrained__resnet34__in_rover__seed1414__gpu_full__1a2b3c4d
```
`split_scope ∈ {in_rover, cross_rover}`. The **directory name IS the run_id** (`manifest.write_manifest`
derives `run_id = run_dir.name`). **`config_hash[:8]` is emitted by `run_experiment.py` at runtime**
(sha256 of the resolved config) and is **NOT knowable in advance** — a cold agent can pre-name only the
deterministic prefix `{model}__{variant}__{backbone}__{split_scope}__seed1414__{profile}`; the trailing
hash is filled by code. Do not hand-author a full run_id.

Run-dir layout `experiments/<run_id>/`:
```
manifest.json          # committed (gitignore-whitelisted)
per_image.parquet      # per-image counts table (see 7.4)  — GPU-run-local; committed COPY under manifests/
per_image.csv          # mirror
preds/<split>/<name>.png   # predicted masks — GITIGNORED (large, regenerable)
train.log
best.ckpt              # GITIGNORED (*.ckpt)
```
**Committed mirror** (so bootstrap is reproducible in-repo):
`experiments/manifests/<run_id>/{manifest.json, per_image.parquet, per_image.csv}`.
`run_experiment.py` MUST copy `manifest.json` AND `per_image.parquet`/`.csv` into
`experiments/manifests/<run_id>/` at the end of every run. `.gitignore` already un-ignores
`experiments/manifests/**`; add these two lines under the experiments block to be explicit:
```
!/experiments/manifests/**/per_image.parquet
!/experiments/manifests/**/per_image.csv
```

### 7.4 Predictions contract (segmentation) — DEFINED here (TO BUILD, MS3)

Every run at eval time MUST emit BOTH:

**(A) Per-image predicted masks:** `experiments/<run_id>/preds/<split>/<name>.png`, single-channel
**uint8**, values `0..3` for the 4 classes and **255 for ignore** (ignore copied from the GT mask so
eval never scores ignore pixels); same H×W as the resized eval input (`size=512`); no color palette.
`<split> ∈ {val, test_msl, test_mer}`; `<name>` = the canonical camera-qualified image name (§4.3).
**Gitignored** (local, regenerable).

**(B) Per-image counts table** `experiments/<run_id>/per_image.parquet` (+ `.csv` mirror), columns
**exactly**:

| column | type | notes |
|---|---|---|
| `run_id` | str | |
| `name` | str | JOIN KEY (canonical camera-qualified image name, §4.3) |
| `split` | str | `val` \| `test_msl` \| `test_mer` |
| `scope` | str | `ALL` or `soil` \| `bedrock` \| `sand` \| `big_rock` |
| `metric` | str | `iou` \| `pixel_acc` \| `boundary_f1` |
| `value` | float | NaN if class absent in both pred and GT for that image |
| `inter` | int64 | per-image intersection pixel count for scope-class (0 for `ALL` and for `boundary_f1`) |
| `union` | int64 | per-image union pixel count (0 for `ALL` and for `boundary_f1`) |
| `n_valid` | int64 | count of non-ignore pixels in the image |

The `(inter, union, n_valid)` **integer counts are REQUIRED** for `iou`/`pixel_acc` so the paired
bootstrap resamples images and recomputes dataset-level IoU from **summed counts**, not by averaging
per-image ratios. **`boundary_f1` carries `inter=union=0`** and is descriptive-only (§6) — it is stored
for reporting and never resampled. `run_experiment.py` writes both artifacts; `eval/stats.py` reads
**only** `per_image.parquet` (never re-derives from PNGs). For McNemar, `analyze_results.py` recomputes
the 2×2 discordant counts (`b = A_correct & B_wrong`, `c = A_wrong & B_correct`) over valid pixels from
the two runs' `preds/` dirs when both are present.

**"Contract-valid predictions"** ≡ `manifest.json` present AND `preds/` has ≥1 file AND
`per_image.parquet` present with the columns above AND `per_image` covers every image in `preds/`.

---

## 8. Canonical repo structure (matches reality; TODOs marked)

```
src/marsseg/
  data/    ai4mars.py  (CLASSES/NUM_CLASSES=4/IGNORE_INDEX=255/CLASS_COLORS + label_key +
                        build_index(root, rover, camera, test_gold_dir) — camera-aware, min3-pinned)     [BUILT MS1 — §4.1/§4.3 FIXED]
           dataset.py  (SegDataset item {image (3,H,W) f32, mask (H,W) i64 255=ignore, name, rover};
                        make_splits by-image; class_pixel_counts; name is a hard key)                    [BUILT MS1 — §4.3 FIXED]
           transforms.py (albumentations; hflip + photometric only, NO vflip/scale/crop; ImageNet norm) [BUILT MS1]
  models/  zoo.py       (build_model registry: baseline/unet/deeplabv3plus/segformer — SINGLE entry point) [BUILT MS2]
           foundation.py (DINOv3 ViT-L/16 SAT frozen backbone + head; SAM ViT-B zero-shot; skip-and-log) [TO BUILD MS2]
  train/   loss.py      (DiceLoss + CombinedLoss = CE + Dice, ignore_index=255)                           [BUILT MS2]
           lit.py       (SegLitModule + SegDataModule — PyTorch Lightning; NO trainer.py)                 [BUILT MS2]
  eval/    metrics.py   (miou/per_class_iou/pixel_acc/boundary_f1, ignore=255)                            [TO BUILD MS3]
           stats.py     (paired_bootstrap per §5.6; H4 per §5.7; mcnemar descriptive)                     [TO BUILD MS3]
           prereg.py    (freeze/verify experiments/PREREG.md before test-set numbers)                     [TO BUILD MS3]
           aggregate.py (per-image counts -> RESULT_COLUMNS rows via append_results)                      [TO BUILD MS3]
           verdict.py   (canonical-run selection §5.9 + hypotheses.yaml + Holm(alpha=0.10) -> H0-H5)      [TO BUILD MS3]
           plots.py     (overlays using data.ai4mars.CLASS_COLORS)                                        [TO BUILD MS3]
  utils/   seed.py config.py manifest.py results.py tracking.py logging.py capabilities.py                [BUILT — REUSE]
configs/   data.yaml  models/*.yaml  hypotheses.yaml                                                      [TO BUILD MS3 — dir is EMPTY]
scripts/   download_data.py  check_env.py                                                                 [BUILT]
           run_experiment.py  analyze_results.py  merge_results.py  run_gpu.sh                            [TO BUILD MS3/MS4]
experiments/  PREREG.md  results_store.{parquet,csv}  <run_id>/manifest.json  manifests/**  (gitignore-whitelisted)
tests/  test_smoke.py test_data.py test_models.py  (+ TO ADD: test_eval.py, contract tests, real-layout count asserts) [15 pass]
paper/  docs/DEVLOG.md
requirements.txt (core, CPU incl. lightning + torchmetrics)  requirements-extras.txt (segment-anything, timm — GPU/H5)
.env.example  pyproject.toml  RESEARCH.MD (immutable rubric)
```
**Do NOT create** `models/{base,unet,smp_models,segformer,registry}.py` or `train/trainer.py` — they
would fork `zoo.py`/`lit.py` and break `from marsseg.models.zoo import build_model`.

### Scripts to build (exact CLIs)

- `scripts/run_experiment.py --config configs/models/<name>.yaml [--override k=v …] --out experiments/<run_id>`
  → `set_seed(1414)`; load config (merging `configs/data.yaml`); `build_index` (camera-aware, §4.1) +
  `make_splits`; **if `data.max_train_images` is set, truncate `train_records` to the first N after
  `make_splits` and before building `SegDataModule`** (this is where the CPU-smoke cap lives — no BUILT
  function does it); assert `name`s unique per split (§4.3); compute class weights on the train split;
  branch on `profile`; train via Lightning (callbacks in §5.4); evaluate the **BEST** ckpt on the pinned
  MSL ncam gold test (min3, §5.4) and, for the H4 subject, on the MER gold test; write `manifest.json`
  (with `extra.resolved_test_gold_dir`, `extra.class_weights`, `extra.weights_source`) + predictions
  (§7.4) + append results rows; copy manifest + per_image into `experiments/manifests/<run_id>/`.
- `scripts/analyze_results.py --store experiments/results_store.parquet --hypotheses configs/hypotheses.yaml --out experiments/analysis/`
  → resolve each comparison to one run per member (§5.9); aggregate + paired-bootstrap (§5.6) / H4
  (§5.7) / McNemar (descriptive) + per-family Holm (α=0.10) → write `experiments/manifests/verdicts.json`
  and `experiments/manifests/leaderboard.csv`; render H0–H5.
- `scripts/merge_results.py --incoming <gpu_store.parquet> --into experiments/results_store.parquet`
  → dedup on `DEDUP_KEYS`.
- `scripts/run_gpu.sh` → V100 turnkey (§2).

**`verdicts.json` shape:** `{alpha:0.10, correction:"holm", generated_git_sha, families:{A:{members:
[{comparison, delta, ci_low, ci_high, raw_p, holm_p, decision}]}, …}, hypotheses:{H0..H5:{decision:
support|reject|deferred, evidence}}}`. **`leaderboard.csv`:** `model, backbone, variant, stratum, miou,
ci_low, ci_high`. `prereg.py` freezes `experiments/PREREG.md` (hypotheses + families + thresholds +
seed) **before** any test-set numbers, and it is committed (gitignore-whitelisted).

---

## 9. Phases & gates (each gate = an exact runnable command + pass signal)

Interpreter is `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (V100). Pytest auto-excludes
`integration`/`network` tests via `pyproject addopts = "-q -m 'not integration and not network'"`.
**PAUSE for user review at each gate before starting the next phase.**

> **Executability note (read before MS4).** `scripts/run_experiment.py`, `configs/*`, and `eval/*` **do
> not exist yet** (`scripts/` holds only `check_env.py` + `download_data.py`; `configs/` is empty;
> `src/marsseg/eval/` is `__init__.py` only). The §4.1/§4.3 code fixes LANDED 2026-07-01, so training
> is now gated **only on MS3 being built**. "Train one U-Net" is a build task, not a runnable command,
> until then. The MS4 command below is written for after MS3 completes — it is not runnable today.

| Phase | Status | Tasks | Gate command → pass signal |
|---|---|---|---|
| **MS0 — setup** | **DONE** (incl. the §3 `.env.example`/DINOv3 fix, 2026-07-01) | scaffold, utils, CI, `check_env`, commit-msg hook, gitignore, `.env.example` | `.venv/Scripts/python.exe scripts/check_env.py` → last line `core OK`, prints `profile=windows_cpu`, exit 0; `.venv/Scripts/python.exe -m pytest tests/test_smoke.py -q` → all pass; `.venv/Scripts/python.exe -m ruff check .` → `All checks passed!`; `.venv/Scripts/python.exe -m black --check .` → clean. |
| **MS1 — data** | **DONE (fixed + re-verified 2026-07-01)** — §4.1 camera/pairing + §4.3 `name` fixes applied; real-layout count asserts green on this box | apply §4.1 camera/pairing fix + §4.3 `name` fix; add real-layout count asserts | `.venv/Scripts/python.exe -m pytest tests/test_data.py -q` → all pass, **including** new asserts: `build_index(DATA_ROOT,"msl")` → `len(train)==16064` and `len(test)==322` (against **min3**); `build_index(DATA_ROOT,"mer")` → `len(test)==204`, `train==[]`; every record has a non-empty unique camera-qualified `name`; `make_splits(val_frac=0.2, seed=1414)` → `set(train_names) & set(val_names) == set()`. Item contract = `{image FloatTensor (3,H,W), mask LongTensor (H,W) in {0,1,2,3,255}, name, rover}`. |
| **MS2 — models** | **PARTIAL** (zoo/loss/lit + 15 tests DONE; `foundation.py` remains; `run_experiment.py` smoke is MS3-gated) | build `models/foundation.py`; add contract test | (A) `.venv/Scripts/python.exe -m pytest tests/test_models.py -q` → pass (forward shapes `(B,4,H,W)`; unknown-model `ValueError`; combined-loss-ignore backward finite; Lightning `fast_dev_run`). (B) `build_model("dinov3_sat"|"sam", …)` returns a module OR skip-and-logs (`status="skipped"`, no crash) when weights/GPU absent. (C) **BLOCKED until MS3** authors `configs/models/baseline.yaml` + `scripts/run_experiment.py`: then `run_experiment.py --config configs/models/baseline.yaml --override data.max_train_images=64 train.max_epochs=1` writes a **contract-valid** run (§7.4) + ≥1 results row with `status ∈ {ok,skipped}`. |
| **MS3 — eval** | **NOT STARTED** | `eval/{metrics,stats,prereg,aggregate,verdict,plots}.py` + `configs/*` + `scripts/{run_experiment,analyze_results}.py` + `PREREG.md` | `.venv/Scripts/python.exe -m pytest tests/test_eval.py -q` → pass, including: metrics + paired-bootstrap (§5.6) + verdict units; **`test_eval.py` loads `configs/hypotheses.yaml` and asserts every family/member/threshold/path is CONCRETE — no placeholder strings (`<pin…>`), `drop_threshold==0.15`, `mer.expected_test_n==204`, `data.expected_test_n==322`).** Then `.venv/Scripts/python.exe scripts/analyze_results.py --store experiments/results_store.parquet --hypotheses configs/hypotheses.yaml --out experiments/analysis/` writes `experiments/manifests/verdicts.json` with a decision in `{support, reject, deferred}` for **every** H0..H5, `ci_low ≤ value ≤ ci_high` where CIs exist, and **H5 = deferred on windows_cpu** (does NOT block H1–H4). |
| **MS4 — runs** | **NOT STARTED** | CPU smoke; V100 full training + cross-rover (H4) + foundation (H5); merge-back | CPU smoke: `.venv/Scripts/python.exe scripts/run_experiment.py --config configs/models/baseline.yaml --override data.max_train_images=64 train.max_epochs=1` exits 0, writes contract-valid run (the override MUST actually subset to 64 train images — §5.4/§8). V100: `bash scripts/run_gpu.sh` exits 0. Merge: `.venv/bin/python scripts/merge_results.py --incoming <gpu_store.parquet> --into experiments/results_store.parquet` → store has a `miou`/`stratum=all`/`status=ok` row for each of {baseline,unet,deeplabv3plus,segformer} + a `cross_rover` row (H4), and `df.duplicated(subset=DEDUP_KEYS).sum()==0`. |
| **MS5 — paper** | **NOT STARTED** | rubric-aligned paper + overlay figures + results binding + licenses; deliverables | `cd paper && latexmk -pdf main.tex` → exit 0, `paper/main.pdf` newer than `main.tex`; `sha256sum RESEARCH.MD` == `181361d246cc0f5e2cde7061ff3c4713815aad233fe61aeda4f8a0d496e84e31` (byte-exact); `grep -E 'H0|H1|H2|H3|H4|H5' paper/main.tex` → each hypothesis present with a decision word matching `verdicts.json`; paper includes a **Data & Model Licenses** subsection (§11). |

---

## 10. Hypothesis → evidence decision table (results_store row patterns → verdicts)

`verdict.py` resolves each comparison member to exactly one run (§5.9), selects rows by
`(model, backbone, variant, scope, stratum, metric)`, joins the two runs' `per_image.parquet` on the
camera-qualified `name` for the paired bootstrap (§5.6, seed 0, n=10000), then applies Holm within the
family. **All bootstraps are on `iou`/macro-mIoU only; `pixel_acc`/`boundary_f1` are never tested (§6).**

| Verdict | Rows compared (candidate vs baseline) | Decision |
|---|---|---|
| **H1 (Family A)** | `{model∈{unet,deeplabv3plus,segformer}, variant=pretrained, scope=ALL, stratum=all, metric=miou}` vs `{model=baseline, scope=ALL, stratum=all, metric=miou}` | **Reject H0** iff Holm-p < 0.10 for ≥1 member (Δ = candidate−baseline > 0). Else **H0 holds (honest)**. |
| **H2 (Family B)** | for each of unet/deeplabv3plus/segformer: `variant=pretrained` vs `variant=scratch` (same model/backbone, scope=ALL, metric=miou) | **Support** iff Holm-p < 0.10 for ≥1 member (Δ = pretrained−scratch > 0). |
| **H3 (Family C)** | `segformer` vs `unet` and `segformer` vs `deeplabv3plus`, scope=ALL (overall) AND scope∈{soil,bedrock,sand,big_rock} with metric=iou (per-class stratum); **Δ = segformer − cnn** | **Support** a direction (`sign(observed Δ)`) iff Holm-p < 0.10 (two-sided, §5.6); report per-class winners with the same fixed-class + resample rules. |
| **H4 (Family D)** | subject = highest-`val_miou` model; `stratum=in_rover, scope=ALL, metric=miou` (MSL ncam test) vs `stratum=cross_rover` (MER test); plus `{model=baseline, stratum=cross_rover}` for the MER baseline | **Generalizes** iff `drop < 0.15` AND `cross_rover_ci_low > baseline_on_MER_miou` (§5.7; deterministic, no p-value, single-member family). |
| **H5 (Family E)** | `{model=dinov3_sat, variant=finetuned}` and `{model=sam, variant=zeroshot}` vs `{model=baseline}`, metric=miou | On `gpu_full`: Holm over `ok` members; **support** iff ≥1 ok member Holm-p < 0.10 (Δ>0). Mixed ok/skipped → §5.8. All `status="skipped"` (windows_cpu / no weights) → **DEFERRED**; never blocks H1–H4. |
| **H0** | — | Reported **honestly**: holds iff H1 not rejected. |

---

## 11. Rubric mapping (RESEARCH.MD), licensing & compute

**Rubric mapping.** INTRO → §0.3.1–3.2 (topic, drivability goal, segmentation ML problem).
HYPOTHESES & METHOD → §3 + §5 + a block diagram (encoders/decoders, CE+Dice loss, ignore mask).
RESEARCH (related work) → AI4Mars, MarsSeg, U-Net, DeepLabV3+, SegFormer, SAM, **DINOv3 (SAT-493M)**.
APPLICATION → mIoU leaderboard, per-class + cross-rover, transfer ablation (H2), paired-significance
tables (bootstrap + descriptive McNemar), augmentation/HP notes. WHAT IS LEARNED → which architecture
wins where, the value of transfer, the cross-rover gap, and drivability implications.

**Licensing, data use & attribution (MS5 paper REQUIRES a "Data & Model Licenses" subsection).**
AI4Mars is NASA open data — cite the AI4Mars paper (Swan et al.) and NASA/JPL; **do not redistribute
raw images** (`data/` gitignored; download via `scripts/download_data.py`, Zenodo 15995036). DINOv3
weights are gated under Meta's DINOv3 license (research use; accept on HF) — **never commit the
weights**, cite the DINOv3 paper, record the model id + license acceptance in `manifest.extra`. SAM is
Apache-2.0 but its ViT-B checkpoint is downloaded separately and must not be committed. ImageNet
encoder weights arrive via `smp`/`transformers` under their respective licenses.

**Compute topology.** Development + smoke run on **Windows 11 CPU** (`profile=windows_cpu`, versions
float — §2). The local **RTX 5070 Ti (Blackwell) is INTENTIONALLY UNUSED** — do not target it. Full
training runs on the incoming **V100 (16 GB, Ubuntu)** via `scripts/run_gpu.sh` — size models for 16 GB
(hence distilled **DINOv3 ViT-L/16 SAT-493M**, not ViT-7B). `run_gpu.sh` makes it a one-command job
whose outputs merge back into `experiments/results_store.parquet` via `merge_results.py` and re-decide
the gated H5 (and finalize H1–H4). The CPU profile runs subset smoke only.
