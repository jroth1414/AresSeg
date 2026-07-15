# DEVPLAN — Terrain-Aware Semantic Segmentation for Mars Rover Drivability (AI4Mars)

**Master development plan for the `marsseg` project (JHU EN.705.742, Advanced Applied ML). Author: John Roth.**

This project builds and rigorously evaluates deep semantic-segmentation models that label Martian
terrain — *soil, bedrock, sand, big rock* — from rover camera images, the core perception task for
**autonomous drivability** assessment. We compare CNN (U-Net, DeepLabV3+) vs transformer (SegFormer)
architectures, measure the value of **ImageNet transfer**, test **cross-rover generalization**
(Curiosity → Opportunity/Spirit), and benchmark a **foundation-model reference** (DINOv3 pretrained on
Earth satellite imagery + a SAM region-oracle upper bound), all under a pre-registered,
significance-tested protocol.

---

## 0. COLD-START: READ THIS FIRST

**What this is.** A single, self-contained, mechanically executable plan. Any agent with **no prior
conversation context**, given only this file + the repo, must reach the **same results and the same
H0–H5 verdicts** as any other agent. Every load-bearing constant is pinned below. Do not invent
versions, paths, thresholds, or file names.

**STATUS (2026-07-10).** The data, model, training, metrics, and analysis implementation exists,
including a historical CPU smoke artifact. The protocol now uses a constant-majority H1/H4
reference, an explicitly named learned Tiny U-Net H5 reference, three learned-model training seeds,
a two-level seed/image bootstrap, and one Holm family for H3 overall plus per-class tests. The
original `experiments/PREREG.md` remains intact; the dated Protocol V2 amendment and executable
snapshot verifier close the prose/code binding gap.

The final Protocol V2 snapshot must be sealed after runtime/model configs stabilize and before
confirmatory analysis. The confirmatory GPU sweep and paper are not complete; H0–H5 therefore
remain unresolved/deferred. `scripts/run_gpu.sh` is an implementation under active validation, not
evidence that the sweep has run successfully. The authoritative current state is the repository
plus the newest entry in `docs/DEVLOG.md`, not historical branch or machine notes.

**Build trail:** append every phase's decisions/verification to **`docs/DEVLOG.md`** (newest at
bottom). Update §0 here and README §Status at the end of each phase.

---

## 1. Operating rules (NON-NEGOTIABLE — read before your first commit)

1. **Authorship policy.** Do not add automated co-author trailers. The tracked
   `.pre-commit-config.yaml` installs a `commit-msg` check that rejects prohibited trailers; cloned
   repositories do not inherit `.git/hooks`. Install it with `pre-commit install --install-hooks`.
2. **RESEARCH.MD is the immutable course rubric — never edit or reformat it.**
   `RESEARCH.MD.sha256` is the sole checksum source. Run `python scripts/check_integrity.py`; do not
   copy the digest into other documents.
3. **`.env` is gitignored and NEVER committed.** Secrets (only `HF_TOKEN`, for gated DINOv3) live
   **only** in `.env`. Only `.env.example` is committed. Never document or infer the contents of a
   developer's live environment file.
4. **`experiments/` and `data/` are gitignored.** The ONLY experiment files that may be committed:
   `experiments/results_store.parquet`, `experiments/results_store.csv`,
   `experiments/**/manifest.json`, `experiments/PREREG*.md`, `experiments/manifests/**`, and
   `.gitkeep`. `data/` keeps only its skeleton via `.gitkeep`. **Never `git add -f`** a checkpoint
   (`*.pt` / `*.pth` / `*.ckpt` / `*.safetensors`), raw data, or predicted-mask PNGs.
5. **Seed policy.** Split seed is 1414. Learned confirmatory arms require training seeds
   `[1414, 1415, 1416]`; deterministic majority/SAM artifacts are computed once and paired against
   each learned seed. Bootstrap purpose-seed is 0.
6. **Work task-by-task.** Preserve user changes, record verification evidence, and pause when a
   change would require new authority.
7. **Compute is capability-driven.** CPU is for smoke/offline tests; full training requires a CUDA
   device that passes `scripts/check_env.py`. Record the detected GPU and memory in each manifest;
   do not encode an assumed machine size into the protocol.

---

## 2. Environment & setup

Use Python **3.11** (`pyproject.toml` requires `>=3.11,<3.12`) and a local `.venv`. Installation is
profile-first, third-party lock second, checked-out source last.

**CPU:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements-cpu.lock.txt
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/check_integrity.py
.venv/bin/python scripts/check_env.py
```

**CUDA 12.1:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cuda121.lock.txt
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install -r requirements-extras.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/check_integrity.py
.venv/bin/python scripts/check_env.py
```

CUDA profile installation deliberately uses dependencies: the pinned PyTorch wheel metadata pins
its Linux `nvidia-*` and Triton requirements. CPU profile installation may use `--no-deps` because
the portable main lock supplies dependencies next. On Windows, replace `.venv/bin` with
`.venv\\Scripts`. `requirements.lock.txt` is UTF-8 and third-party-only; never install the project
from a Git URL inside that lock.

Pretrained arms require their upstream weights/cache. DINOv3 additionally requires accepted gated
access and `HF_TOKEN`; SAM requires its separately downloaded checkpoint. Record resolved model
revisions, checkpoint hashes, package versions, GPU identity, and training metrics in each run.
See `docs/REPRODUCIBILITY.md` for the concise profile guide.

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

Hypotheses and executable selectors live in **`configs/hypotheses.yaml`**. The historical
`experiments/PREREG.md` is preserved and the dated V2 amendment documents pre-confirmatory
corrections. **Significance threshold: α = 0.10; Holm correction is applied within each configured
family.** The executable config and sealed Protocol V2 snapshot take precedence over duplicated
historical prose.

| ID | Statement | Family | Test | Exact decision rule |
|---|---|---|---|---|
| **H0** | No tested deep system significantly beats constant-majority at mIoU. | — | — | Reject iff H1 is supported; otherwise report `fail_to_reject`, never support for H0. |
| **H1** | At least one tested deep system beats `majority`. | A | paired seed/image bootstrap, one-sided | Reject H0 iff at least one Family-A adjusted p < 0.10 with positive delta. |
| **H2** | Pretrained initialization helps at least one tested architecture. | B | paired seed/image bootstrap, one-sided | Support only this bounded claim iff at least one Family-B adjusted p < 0.10 with positive delta; do not generalize to all transfer learning. |
| **H3** | Configured SegFormer MiT-B2 systems differ from U-Net/ResNet-34 or DeepLabV3+/ResNet-50 overall or by class. | C | paired seed/image bootstrap, two-sided | One Holm family contains both overall and every emitted fixed-set per-class test. Report signed configured-system effects and parameter counts; do not claim an architecture-only causal effect. |
| **H4** | The validation-selected conventional system has bounded Curiosity→MER degradation. | D | threshold plus hierarchical MER CI | Support iff mean drop < 0.15 and MER CI low exceeds `majority` on MER; no p-value. |
| **H5** | DINOv3-SAT and the SAM region-oracle upper bound are useful learned-reference comparisons. | E | paired seed/image bootstrap vs `tiny_unet`; gated | Holm over completed members; incomplete learned seed sets defer. SAM is not deployable zero-shot semantic segmentation. |

**H5 gated-weights steps (do only when running H5):**
1. Visit `https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m` while logged in and
   **accept the license**. The configured arm is ViT-L/16 SAT-493M; memory feasibility must be
   established by the GPU preflight rather than assumed from a machine label.
2. Create an HF **read** token.
3. Put it in `.env` as `HF_TOKEN=…` (gitignored, never committed).
4. Code reads it via `config.require_secret("HF_TOKEN")` **only on the load path** — but
   `foundation.py` must FIRST check `os.environ.get("HF_TOKEN")` and, if absent (or cuda absent, or
   SAM checkpoint absent), **skip-and-log** (append a `status="skipped"` results row, record in
   `manifest.gpu_stages_skipped`, return) rather than raise. Never hard-crash the pipeline on CPU.
5. Never inspect, publish, or make claims about the contents of a developer's live `.env`.

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
| `majority` | parameter-free constant class 0 (soil) | — (`none`) | never (H0/H1 and H4 reference) | 0 |
| `tiny_unet` (`baseline` legacy alias) | `TinyUNet` (from scratch, `base=16`) | — (`none`) | never (learned H5 reference) | ~117 k |
| `unet` | smp `Unet` | `resnet34` | `encoder_weights="imagenet"` if pretrained (network on first use) | 24.4 M |
| `deeplabv3plus` | smp `DeepLabV3Plus` | `resnet50` | `encoder_weights="imagenet"` if pretrained (network on first use) | 26.7 M |
| `segformer` | `transformers` MiT (`_SegFormer`) | `"b0"` \| `"b2"` | pretrained loads only `nvidia/mit-{b}` ImageNet encoder; task head is new | report from manifest |

`pretrained=False` builds the scratch variant (random init) for H2 and needs **no network**. See §2
"Network precondition" for the pretrained arms. Foundation models (`dinov3_sat`, `sam`) live in
`models/foundation.py`.

**Backbone naming contract (results-store label).** `build_model` takes SegFormer backbone as `"b0"`/`"b2"`,
but the results-store `backbone` column uses **`mit-b0`/`mit-b2`**. Each `configs/models/*.yaml` carries
both `model.backbone` (the zoo build arg, e.g. `b0`) and `model.results_backbone` (the store label,
e.g. `mit-b0`); `run_experiment.py` writes `results_backbone` into the store. For smp models
`backbone == results_backbone` (`resnet34`, `resnet50`, `efficientnet-b0`); for `majority` and
`tiny_unet`, `backbone=none`.

### 5.2 Lightning module (`train/lit.py`, BUILT) — frozen hyperparameters

`SegLitModule` hparams (all pinned in code): `model_name`, `num_classes=4`, `backbone`,
`pretrained`, `class_weights`, **`lr=3e-4`**, **`weight_decay=1e-4`**, `dice_weight=1.0`,
`ignore_index=255`, `max_epochs=50`.
- Optimizer: **`AdamW(lr=3e-4, weight_decay=1e-4)`**; scheduler **`CosineAnnealingLR(T_max=max_epochs)`**.
- Metrics: `torchmetrics.MulticlassJaccardIndex(ignore_index=255)`, per-class + macro; logs
  `train_loss`, `val_loss`, `val_miou`, `val_iou_{class}`.

`SegDataModule` wraps `SegDataset`; batch size, workers, augmentation, and seed come from the merged
run config. DataLoader generators and workers are seeded from the run seed.

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
| Gold-dir integrity | `build_index` honors the explicit min3 paths; `scripts/check_data.py` verifies counts, pairing, split disjointness, masks, and fingerprints before a run. | `build_index` / `configs/data.yaml` / `check_data.py` |
| Epochs | `max_epochs=50` (= cosine `T_max`) | `SegLitModule` / `configs/models/*.yaml: train.max_epochs` |
| Early stop | `EarlyStopping(monitor="val_miou", mode="max", patience=10, min_delta=0.001)` | `run_experiment.py` callback |
| Checkpoint | `ModelCheckpoint(monitor="val_miou", mode="max", save_top_k=1)`; **evaluate the BEST ckpt** (not last) | `run_experiment.py` callback |
| Grad clip | `gradient_clip_val=1.0` | `L.Trainer` |
| Precision | `"16-mixed"` on `gpu_full`, `"32-true"` on `windows_cpu` (affects speed only, not the verdict) | `run_experiment.py` |
| LR / opt / sched | `AdamW(3e-4, wd=1e-4)` + `CosineAnnealingLR(T_max=50)` | `train/lit.py` (frozen) |
| Class weights | **`w_c = median(counts)/counts_c`, clipped to `[0.5, 10.0]`**, computed on the **train split only** (post-`make_splits`, never val/test), `max_images=null` (full scan); record the 4-vector in `manifest.extra.class_weights` | `configs/data.yaml: class_weights` |
| Augmentation | Values are read from `configs/data.yaml: aug`; train/eval both resize and normalize consistently. | `data/transforms.py` / `configs/data.yaml` |
| Batch size | Per-model config; gradient accumulation may preserve effective batch size after GPU preflight. | `configs/models/*.yaml` |
| CPU-smoke subset | `data.max_train_images` (int or null) — **consumed by `run_experiment.py`**, which truncates `train_records` to the first N **after** `make_splits` and **before** building the DataModule (see §8), and ALSO caps `val_records` to the same N (recorded as `manifest.val_truncated_to`) so the smoke validation pass is tractable on CPU. `null` = full data, train AND val untouched (GPU runs unaffected). WIRED (MS3). | `configs/data.yaml` / `run_experiment.py` |
| Bootstrap | `n_resamples=10000`, purpose-seed 0; sample training seeds then paired images within sampled seed; recompute metrics from sufficient counts; 90% percentile CI. | `eval/stats.py` / `configs/hypotheses.yaml` |
| McNemar | unit = pixel, valid pixels only (`mask!=255`), Edwards continuity `(|b−c|−1)²/(b+c)`; **secondary/descriptive only — never overrides the image-level bootstrap verdict** | `eval/stats.py` |
| H4 mechanics | **§5.7** (single procedure, no p-value) | `configs/hypotheses.yaml: H4` |
| Foundation gating on CPU | Missing CUDA/token/checkpoint/dependency produces explicit skipped/deferred artifacts. SAM output variant is `region_oracle_upper_bound`. | `models/foundation.py` / `run_experiment.py` |
| Descriptive-only metrics | `boundary_f1` and `pixel_acc` are **leaderboard/reporting only**; **NO hypothesis is tested on them.** All bootstraps operate solely on `iou`-derived macro-mIoU (H1/H2/H4) and per-class `iou` (H3). | §5.6 / §6 / §7.4 |
| Seed policy | split seed 1414; learned primary seeds `[1414,1415,1416]`; deterministic majority/SAM artifact seed 1414; bootstrap seed 0 | `configs/hypotheses.yaml` / `utils/seed.py` |

### 5.5 `configs/` schema (implemented)

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
  Enumerated model configs: `majority`, `tiny_unet`; `unet` (resnet34) {pretrained, scratch};
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
    resampling_unit: seed_then_image
    ci_method: percentile
    bootstrap_seed: 0
    rng: numpy_default_rng          # np.random.default_rng(0); one draw advanced per replicate (§5.6)
    seed_reset_per_comparison: true # each comparison re-seeds default_rng(0); deterministic + independent
    p_estimator: plus_one           # p = (1 + #{...}) / (n_resamples + 1)  (§5.6)
    fixed_class_set: full_split_present   # classes with union>0 over the FULL split, fixed before bootstrap
    empty_class_in_resample: iou_zero     # a fixed-set class with union==0 in a resample contributes IoU=0
    mcnemar: {unit: pixel, scope: valid_pixels_only, correction: continuity, report: secondary_only}
  canonical_run_selection:
    filter: {seeds: [1414, 1415, 1416], profile: gpu_full, status: ok}
    require_complete_seed_set: true
    deterministic_models: [majority, sam]
    deterministic_artifact_seed: 1414
  families:
    A: {members: [majority_vs_unet, majority_vs_deeplabv3plus, majority_vs_segformer]}
    B: {members: [unet_pretrained_vs_scratch, deeplabv3plus_pretrained_vs_scratch, segformer_pretrained_vs_scratch]}
    C: {members: [segformer_vs_unet, segformer_vs_deeplabv3plus]}   # delta orientation: segformer - cnn
    D: {members: [best_in_rover_vs_cross_rover]}
    E: {members: [dinov3_sat_vs_tiny_unet, sam_region_oracle_vs_tiny_unet]}
  hypotheses:
    H1: {family: A, test: paired_bootstrap, statistic: delta_miou, tail: greater,
         metric: miou, decision_rule: "reject_H0 if holm_p < 0.10 for >=1 member (delta>0)"}
    H2: {family: B, test: paired_bootstrap, statistic: delta_miou, tail: greater, metric: miou,
         decision_rule: "support only a benefit for >=1 tested architecture"}
    H3: {family: C, test: paired_bootstrap, tail: two_sided, strata: [all, per_class], metric: miou,
         delta_orientation: "segformer_minus_cnn", per_class_metric: iou,
         holm_scope: all_members_x_overall_and_fixed_set_classes}
    H4: {family: D, test: threshold_plus_ci, statistic: miou_drop, drop_threshold: 0.15,
         emits_p_value: false, decision_rule: "support iff drop < 0.15 AND cross_rover_ci_low > majority_on_MER_miou"}
    H5: {family: E, gated: true, decided_on_profile: gpu_full, on_missing_gpu: deferred,
         partial_gpu_rule: "holm over ok members only; support iff >=1 ok member holm_p<0.10 (delta>0); reject if all ok fail; deferred if zero ok members"}
  ```
  Executable `comparisons` selectors are intentionally not duplicated here; inspect
  `configs/hypotheses.yaml`. Each learned side resolves to a complete three-seed run set; one
  deterministic majority/SAM artifact is reused across paired seed strata. H0 has no comparison
  entry and uses `fail_to_reject` when H1 is not supported.
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
3. **Per replicate (10,000 total):** sample the three learned-model seeds with replacement, then
   sample images with replacement independently within every sampled seed. Each image draw is
   shared by both comparison sides. Deterministic majority/SAM predictions are reused for the
   corresponding sampled learned seed.
4. **Recompute each model's macro-mIoU on that resample from SUMMED counts:** for each class `c ∈ S`,
   `sum_inter_c = Σ_over_sampled_images inter_c`, `sum_union_c = Σ union_c` (with multiplicity). Per-class
   IoU `= sum_inter_c / sum_union_c`; if `sum_union_c == 0` in this resample, that class contributes
   **IoU = 0** (`empty_class_in_resample: iou_zero`) — the macro denominator stays `|S|`, fixed.
   `macro_mIoU = mean over S`.
5. **Delta per replicate:** equal-weight mean of sampled seed-level deltas. Orientation is
   candidate−majority for A, pretrained−scratch for B, segformer−cnn for C, and
   foundation−tiny_unet for E.
6. **p-value (plus-one estimator).**
   - one-sided `greater` (A/B/E): `p = (1 + #{delta_b <= 0}) / (n_resamples + 1)`.
   - two-sided (C): `p = 2 * min( (1+#{delta_b<=0}), (1+#{delta_b>=0}) ) / (n_resamples + 1)`, clipped to 1.
7. **CI:** percentile CI of the `delta_b` distribution at `ci_level=0.90` → `[5th, 95th]` percentiles.
   (The reported point `delta` is the observed split-level delta, not the bootstrap mean.)
8. **Per-class strata (H3):** reuse the same seed/image draws as the overall statistic.
9. **Holm within family:** Family C contains both overall tests and all emitted fixed-set per-class
   tests; adjusted p-values and decisions are exposed at every scope.

### 5.7 H4 mechanics (single procedure — no p-value)

- **Subject** = the single in-rover model with the highest `val_miou` among the trained MSL models;
  **reuse its H1 checkpoint (no retraining).**
- Evaluate the subject on: (i) the **MSL ncam gold test** (322) → `mIoU_in_rover` point estimate; and
  (ii) the **MER gold test** (204) → `mIoU_cross_rover` point estimate.
- `drop = mIoU_in_rover − mIoU_cross_rover` (**point estimate**, no bootstrap on the drop).
- Bootstrap the **MER-test images only** (`default_rng(0)`, n=10000, resample the 204 images with
  replacement, recompute macro-mIoU from summed counts over the fixed class set of the MER split) →
  90% percentile CI on `mIoU_cross_rover` → `cross_rover_ci_low` (5th pct).
- `majority_on_MER_miou` = the constant-majority model's seed-reused MER-test point estimate.
- **SUPPORT H4 iff `drop < 0.15` AND `cross_rover_ci_low > majority_on_MER_miou`.** No p-value; Family D
  has one member so Holm is a no-op. This rule is stated identically in §3, §5.5, §5.7, and §10.

### 5.8 H5 partial-GPU (Family E mixed ok/skipped) rule

Family E has two members (`dinov3_sat_vs_tiny_unet`, `sam_region_oracle_vs_tiny_unet`). On CPU both
are `status="skipped"` ⇒ **H5 = DEFERRED** (never blocks H1–H4). On a run where only some members
completed (e.g. SAM ran but DINOv3 skipped):
- **Holm runs over ONLY the members with `status=="ok"`.** Skipped members are reported
  `verdict=deferred` individually.
- **H5 overall = support** iff ≥1 ok member has `holm_p < 0.10` (Δ>0); **reject** if all ok members fail;
  **deferred** if there are zero ok members.

### 5.9 Canonical-run selection (comparison id → primary run set)

The results store may contain multiple runs for a `(model, backbone, variant, stratum)` tuple
(re-runs, multiple seeds, CPU-smoke rows). `verdict.py` resolves learned selectors to seeds
`[1414,1415,1416]` with `profile=gpu_full,status=ok`; incomplete learned sets defer. Majority and SAM
reuse one deterministic seed-1414 artifact across the paired strata.
1. Require the configured seed policy and explicit YAML selector.
2. If >1 row remains for one seed and `(model, backbone, variant, stratum)` tuple, take the run whose
   `manifest.timestamp_utc` is newest; **if still tied, raise** (do not guess).
3. Load and align every resolved `per_image.parquet`, then run §5.6.

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
reference it via config key `model.sam_checkpoint`. Its SHA-256 is pinned by
`model.sam_checkpoint_sha256`; `scripts/run_gpu.sh` promotes a download only after verification.
Checkpoints are gitignored (`*.pth`). If the SAM import or a verified checkpoint is absent,
`foundation.py` skip-and-logs (`status="skipped"`).

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
| `variant` | `constant, pretrained, scratch, finetuned, region_oracle_upper_bound` |
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
           foundation.py (DINOv3 ViT-L/16 SAT frozen backbone + head; SAM ViT-B zero-shot; skip-and-log) [BUILT MS2]
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
  and `experiments/manifests/leaderboard.csv`; render H0–H5. The leaderboard contains explicit
  `gpu_full` per-seed rows for `[1414,1415,1416]`; it is not the multi-seed hypothesis summary.
- `scripts/merge_results.py --incoming <gpu_store.parquet> --into experiments/results_store.parquet`
  → dedup on `DEDUP_KEYS`.
- `scripts/run_gpu.sh` → V100 turnkey (§2).

**`verdicts.json` shape:** `{alpha:0.10, correction:"holm", generated_git_sha, families:{A:{members:
[{comparison, delta, ci_low, ci_high, raw_p, holm_p, decision}]}, …}, hypotheses:{H0..H5:{decision:
support|reject|deferred, evidence}}}`. **`leaderboard.csv`:** `model, backbone, variant, stratum,
profile, seed, miou, ci_low, ci_high`; CI endpoints remain per-seed and are never averaged.
`prereg.py` freezes `experiments/PREREG.md` (hypotheses + families + thresholds +
seed) **before** any test-set numbers, and it is committed (gitignore-whitelisted).

---

## 9. Phases & gates (each gate = an exact runnable command + pass signal)

Interpreter is `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (POSIX). Pytest auto-excludes
`integration`/`network` tests via `pyproject addopts = "-q -m 'not integration and not network'"`.
**PAUSE for user review at each gate before starting the next phase.**

> **Executability note (updated 2026-07-10).** The CPU smoke is historical pipeline evidence only.
> Protocol V2 verification intentionally refuses confirmatory analysis until the final snapshot is
> sealed. GPU preflight, the multi-seed sweep, and result merge remain pending.

| Phase | Status | Tasks | Gate command → pass signal |
|---|---|---|---|
| **MS0 — setup** | Implemented | scaffold, utils, locked profiles, tracked pre-commit, CI, integrity check | `python scripts/check_integrity.py`; `python scripts/check_env.py`; offline lint/tests. |
| **MS1 — data** | **DONE (fixed + re-verified 2026-07-01)** — §4.1 camera/pairing + §4.3 `name` fixes applied; real-layout count asserts green on this box | apply §4.1 camera/pairing fix + §4.3 `name` fix; add real-layout count asserts | `.venv/Scripts/python.exe -m pytest tests/test_data.py -q` → all pass, **including** new asserts: `build_index(DATA_ROOT,"msl")` → `len(train)==16064` and `len(test)==322` (against **min3**); `build_index(DATA_ROOT,"mer")` → `len(test)==204`, `train==[]`; every record has a non-empty unique camera-qualified `name`; `make_splits(val_frac=0.2, seed=1414)` → `set(train_names) & set(val_names) == set()`. Item contract = `{image FloatTensor (3,H,W), mask LongTensor (H,W) in {0,1,2,3,255}, name, rover}`. |
| **MS2 — models** | **DONE (2026-07-02)** — gates A+B green (33 tests: gating skip/happy/load-failure/layout contracts + zoo/loss/lit); gate C runs with MS3 by design | build `models/foundation.py`; add contract test | (A) `.venv/Scripts/python.exe -m pytest tests/test_models.py -q` → pass (forward shapes `(B,4,H,W)`; unknown-model `ValueError`; combined-loss-ignore backward finite; Lightning `fast_dev_run`). (B) `build_model("dinov3_sat"|"sam", …)` returns a module OR skip-and-logs (`status="skipped"`, no crash) when weights/GPU absent. (C) **BLOCKED until MS3** authors `configs/models/baseline.yaml` + `scripts/run_experiment.py`: then `run_experiment.py --config configs/models/baseline.yaml --override data.max_train_images=64 train.max_epochs=1` writes a **contract-valid** run (§7.4) + ≥1 results row with `status ∈ {ok,skipped}`. |
| **MS3 — eval** | Implemented; V2 seal pending final stable tree | metrics, hierarchical bootstrap, YAML selectors, amendment, fail-closed snapshot verification | `python -m pytest tests/test_eval.py -q`; seal Protocol V2 before analysis. |
| **MS4 — runs** | Pending | GPU memory preflight; three genuine seeds per learned arm; deterministic majority/SAM reuse; H4/H5; merge | Validate `scripts/run_gpu.sh`, then require complete manifests/checkpoints/per-image tables and zero duplicate result keys. |
| **MS5 — paper** | Not started | rubric-aligned paper, results binding, limitations/licenses | `python scripts/check_integrity.py`; paper decisions must match sealed verdicts. |

---

## 10. Hypothesis → evidence decision table (results_store row patterns → verdicts)

`verdict.py` resolves each comparison member to a primary run set (§5.9), selects rows by
`(model, backbone, variant, scope, stratum, metric)`, joins the two runs' `per_image.parquet` on the
camera-qualified `name` for the paired bootstrap (§5.6, seed 0, n=10000), then applies Holm within the
family. **All bootstraps are on `iou`/macro-mIoU only; `pixel_acc`/`boundary_f1` are never tested (§6).**

| Verdict | Rows compared (candidate vs baseline) | Decision |
|---|---|---|
| **H1 (Family A)** | pretrained deep systems vs `majority` | Reject H0 iff at least one positive comparison is Holm-significant. |
| **H2 (Family B)** | pretrained vs scratch within each tested architecture/backbone | Support only “helps at least one tested architecture” when significant. |
| **H3 (Family C)** | MiT-B2 vs the two configured CNN systems, overall and per class | One Holm family over every overall/per-class p-value; report direction and parameter counts. |
| **H4 (Family D)** | validation-selected subject in-rover vs cross-rover; `majority` MER reference | Support iff drop < 0.15 and cross-rover CI low > majority-on-MER. |
| **H5 (Family E)** | DINOv3-SAT and SAM region oracle vs `tiny_unet` | Holm over completed members; missing learned seed sets defer. |
| **H0** | — | `reject` with supported H1; otherwise `fail_to_reject` or `deferred`. |

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

**Compute topology.** Capability detection, not a hard-coded workstation description, selects CPU
smoke versus CUDA execution. Record exact GPU model, memory, driver, wheel profile, and effective
batch settings in each run manifest. A script being present is not proof that a full sweep completed.
