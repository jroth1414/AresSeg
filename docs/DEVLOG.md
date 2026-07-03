# Development Log — marsseg (Mars terrain segmentation)

Running, dated trail of decisions, problems + fixes, and verification evidence — the raw material for
the paper's methodology / limitations / reproducibility sections. Newest entry at the bottom.

---

## MS0 — Project scaffold · branch `mars-terrain-segmentation`

Stood up the `marsseg` package and the reproducibility backbone: config-driven runs, run manifests,
a long/tidy results store, deterministic seeding, capability detection, logging, CI, and a
`check_env.py` gate. CV stack installs clean on Windows/CPU (torch 2.4.1+cpu, torchvision 0.19.1+cpu,
`segmentation-models-pytorch` 0.5.0, transformers, albumentations, opencv, scikit-image);
`check_env.py` → `profile=windows_cpu`, `core OK`.

**Compute split.** Segmentation training is GPU-bound. The full pipeline + smoke tests run CPU-only;
full model training targets the V100 (16 GB) via `scripts/run_gpu.sh`. The local RTX 5070 Ti
(Blackwell) is left unused for training — its bleeding-edge toolchain isn't worth fighting when a
well-supported V100 is available.

## MS1 — AI4Mars data pipeline

Built the AI4Mars acquisition + dataset layer: a resumable, MD5-verified Zenodo downloader
(`scripts/download_data.py`); an index builder over the merged-0.6 layout
(`msl/images/edr` + `msl/labels/{train,test}`, MER for the cross-rover test); a `SegDataset` that
pairs each image with its label map, replicates grayscale to 3 channels, and preserves the **255 =
ignore** label (unlabeled / rover-self / >30 m range) through albumentations transforms; **by-image**
train/val splits (a frame never crosses splits); and per-class pixel counts for the class-weighted
loss. Offline fixture tests cover the index, the dataset item + ignore handling, split disjointness,
and class counts.

**Verified facts (AI4Mars merged-0.6, Zenodo).** Labels are pixel values **0–3**
(soil/bedrock/sand/big_rock), **255 = NULL/ignore**; MSL (Curiosity) ≈ 16,064 train + 322
expert-labeled test images; MER (Opportunity/Spirit) supports the cross-rover test. The 16 GB
dataset is downloaded to `data/raw/ai4mars` (gitignored).

**Next:** MS2 — model zoo (baseline U-Net; smp U-Net / DeepLabV3+ with ImageNet encoders; SegFormer;
DINOv3-SAT + SAM foundation references) + the segmentation Trainer (class-weighted CE + Dice,
`ignore_index`, AMP on GPU).

## DEVPLAN hardening — adversarial cold-start pass

Rewrote `DEVPLAN.md` to be **adversarially hardened**: self-contained and mechanically executable so any
agent with no prior context can reach the same results and the same H0–H5 verdicts. Method was an
11-agent red-team fan-out — six adversarial lenses (structure/contract drift, data/artifact contracts,
environment & reproducibility, protocol ambiguity, guardrails, gate executability) → synthesis → three
**cold-start verifier** agents that tried to execute the rewrite blind → reconcile. It surfaced 71
findings (35 blocker) and 22 residual cold-start blockers, all folded in. Every load-bearing claim was
re-verified by hand against the repo (RESEARCH.MD SHA-256, image/label counts, `seed.py` API, the
results-store/manifest schemas, pyproject markers).

The rewrite pins every load-bearing constant and adds sections the plan lacked: a cold-start status +
resume block, non-negotiable operating rules (no-AI-author + commit-msg hook, RESEARCH.MD SHA, `.env`,
gitignore policy, seed=1414, per-task-commit/per-phase-branch/pause-at-gate), exact env/setup + V100
handoff, the verified on-disk data layout, frozen protocol constants, the single-source-of-truth
paired-bootstrap spec (§5.6) + deterministic H4 rule (§5.7) + H5 partial-GPU rule (§5.8) +
canonical-run selection (§5.9), the artifact contracts (results store, manifest, and a **newly-defined
segmentation predictions contract**), a corrected repo tree, per-phase gates as runnable commands, and
a hypothesis→evidence decision table.

**Defect discovered (MS1 REOPENED).** The adversarial pass found a real, silent bug the synthetic-fixture
unit tests never exercised: `data/ai4mars.py::build_index` returns an **EMPTY MSL training index** and an
**empty MER index** against the actual nested `merged-0.6` layout. Verified empirically:
`build_index('data/raw/ai4mars')` → MSL `train=0, test=322`; `rover='mer'` → `train=0, test=0`. Root
causes: (A) MSL images are under `msl/ncam/images/edr` and labels under `msl/ncam/labels/train`, but
`find_dir`'s `rglob` fallback binds "train" to the wrong camera (`msl/mcam/labels/train`, since `mcam`
sorts before `ncam`) → ncam images paired against mcam label stems → 0 matches; (B) MER JPGs live under
`mer/images/{eff,test}` (not directly in `mer/images/`) and gold labels carry `_<digits>_T<digits>_merged`
suffixes the stem normalizer doesn't strip. Also: `build_index` records carry no `name`, so the
paired-bootstrap cross-model join key is unusable, and the gold-dir picker hard-selects `min1` while the
protocol pins `min3` (counts coincide at 322/204, so a green count doesn't prove the right set). Exact
fixes are specified in DEVPLAN §4.1/§4.3/§5.4 as hard prerequisites of the first training run.

**Next:** MS2 — build `models/foundation.py`, then MS3 (`configs/*`, `eval/*`, `run_experiment.py`,
`analyze_results.py`, `hypotheses.yaml`, `PREREG.md`) — but first apply the §4.1/§4.3 `build_index`
fixes and re-verify the count assertions (16064 / 322 / 204, MER train empty).

## MS1 (reopened) — build_index fixed against the real layout · branch `phase-ms1-fix`

Applied the two blocking DEVPLAN fixes and re-closed MS1. `build_index` is now **camera-aware and
ncam-scoped**: MSL pairs images/labels strictly inside one `msl/<camera>/` subtree (default `ncam`,
the protocol camera; the old `find_dir` rglob had bound ncam images to mcam labels → 0 train pairs);
MER indexes images from the union of `mer/images/{eff,test}` with the gold `test` pool winning stem
collisions, and a new `label_key()` normalizer strips `_merged`/`_label` then trailing
`_<digits>`/`_T<digits>` tokens so MER's `<stem>_16165_T0_merged.png` labels pair. The gold test dir
is now **pinned explicitly** (default `masked-gold-min3-100agree`; explicit `test_gold_dir` accepts a
base-relative path or bare name, must resolve inside the current rover/camera's `labels/test`, and
raises if absent) — never `sorted()[0]`, which had silently picked min1. Every record carries the
canonical camera-qualified join key `name = {rover}_{camera|pool}_{label_key(image_stem).lower()}`
(`msl_ncam_…`, `mer_test_…`), records are sorted by `name` (index order no longer depends on
filesystem enumeration → identical splits on Windows and the V100), `build_index` hard-fails on
duplicate names, and `SegDataset` returns `rec["name"]` (missing name = `KeyError`, the old `str(i)`
fallback that could silently misalign cross-model joins is gone).

**Verification (this box, real data):** `msl ncam train=16064, test=322` with every test label under
`masked-gold-min3-100agree` by path; `mer train=[], test=204` (min3 by path); names unique/non-empty;
`make_splits(0.2, 1414)` disjoint with `val=3213`. Real-layout asserts live in `tests/test_data.py`
and **skip when the dataset is absent** so CI stays offline-green. Synthetic fixtures now mirror the
real nested layout and include an **mcam decoy** (regression for the camera-crossing bug), **min1
decoys with counts different from min3** on both rovers (a green count proves min3 selection — the
real 322/204 counts coincide across min1/2/3), MER pool/suffix fixtures, literal name pins
(`msl_ncam_ntrain0`, `mer_test_1n0eff0338p1931l0m1`), a non-trivial 32→16 mask resize in the item
test, and a split-determinism assert.

**Adversarial review pass (multi-agent).** A 4-lens review (spec compliance / correctness /
cross-platform determinism / test adequacy) with independent refuter votes per finding confirmed 3
gaps — MER gold-dir pinning untested (the one *major*: a `sorted()[0]` regression in the MER branch
would have passed the entire suite), no literal name pins, no-op resize in the offline item test —
plus three cheap correctness hardenings (gold-dir containment, raise on pinned-but-missing test
root, in-index name-uniqueness guard). All folded in; refuted findings (e.g. Windows/Linux glob-case
concerns — covered by the sort + both-case extensions) were dropped.

**Also closed:** the pending MS0 `.env.example` fix (HF_TOKEN = required-for-H5 with the DINOv3
license URL; stale NTL tokens purged from the live `.env`) and the DINOv2→DINOv3 naming sweep
(README, requirements, capabilities, proposal tex+pdf).

**Gate (MS1, green):** 27 pytest tests pass (16 in `test_data.py`, incl. 5 real-layout asserts on
this box), `ruff` clean, `black` clean, `check_env.py` → `core OK`.

**Next:** MS2 tail — `models/foundation.py` (DINOv3-SAT frozen backbone + head; SAM zero-shot;
skip-and-log gating), then MS3.

## MS2 (tail) — foundation models (H5, gated) · branch `phase-ms2-foundation`

Built `models/foundation.py`, closing MS2. **DINOv3-SAT arm:** `DinoV3SatSegmenter` wraps the gated
`facebook/dinov3-vitl16-pretrain-sat493m` ViT-L/16 backbone (SAT-493M, the largest variant that fits
the 16 GB V100) **frozen** — pinned to eval mode even under `.train()` — with a trainable
Conv3×3–GN–GELU–Conv1×1 head over the patch-token grid, bilinear-upsampled to input resolution; a
token-grid/prefix mismatch raises at first forward rather than silently mis-gridding. **SAM arm:**
`SamZeroShotSegmenter` builds the ViT-B automatic mask generator from `data/weights/sam/` (eval-only;
`forward()` raises by design), moved to CUDA at build since it never passes through the Lightning
Trainer. **Gating (DEVPLAN §3/§5.4):** `gating_reasons()` checks cuda, `HF_TOKEN` (environment first,
`.env` via `load_env`), the `segment-anything` import, and checkpoint presence; `build_foundation()`
skip-and-logs (returns `None`, reason logged) — and the **load path is inside the same contract**, so
a pending HF license approval, wrong-scope token, network error, or corrupt checkpoint logs a skip
instead of crashing the V100 run. Routed through `zoo.build_model` (single registry entry point),
which gained an optional `sam_checkpoint` kwarg so config key `model.sam_checkpoint` (§6) reaches the
arm — previously a non-default path was silently ignored.

**Protocol flag for MS3 PREREG (user sign-off needed):** SAM emits class-AGNOSTIC regions and AI4Mars
has no prompt channel, so the planned H5 scoring maps each SAM region to the **majority ground-truth
class among its valid pixels (region-oracle)** — an explicit UPPER BOUND on any zero-shot region
labeler, documented in the module docstring. This must be frozen in `experiments/PREREG.md` before
any test-set number is computed.

**Adversarial review pass (multi-agent, 3 lenses + refuter votes).** Confirmed and fixed: gate B's
"returns a module" arm had zero coverage (a mutation returning `None` unconditionally passed the
suite); the stub-ViT forward test was value-blind (dropped `transpose`, gh/gw swap, and wrong-end
prefix slicing all survived — now caught by a positional-token stub with exact bilinear-corner
asserts on a non-square 2×4 grid); the "log" half of skip-and-log was unasserted (marsseg loggers
set `propagate=False`, so caplog can't see them — an injected recorder logger now can); the skip
test was environment-inverted on a provisioned V100 (could even trigger a gated download inside
pytest — now the cuda gate is forced off, deterministic on any box); gate isolation gaps
(cuda-alone, cuda-gates-both-arms, sam all-pass). Plus the three correctness items above
(load-failure skip, SAM device move, `sam_checkpoint` plumb-through), found by the review's
correctness lens.

**Gate (MS2, green):** gates A+B — 33 pytest tests pass, `ruff` clean, `black` clean. Gate C (the
`run_experiment.py` contract-valid smoke) is **MS3-gated by design** and rolls into the MS3 gate.

**Next:** MS3 — `configs/{data,models/*,hypotheses}.yaml`, `eval/{metrics,stats,prereg,aggregate,
verdict,plots}.py`, `scripts/{run_experiment,analyze_results}.py`, `experiments/PREREG.md` (freeze
the SAM region-oracle decision there), then the MS2-C/MS4 CPU smoke.

## MS3 — evaluation stack, configs, pre-registration · branch `phase-ms3-eval`

Built the entire hypothesis-testing machinery, closing MS3 (and MS2's gate C). **Configs:**
`data.yaml` (min3 gold pins, 322/204 expected counts, the CPU-smoke `max_train_images` cap),
`hypotheses.yaml` (families A–E, the frozen §5.6 stats block, H4 `drop_threshold: 0.15`, §5.9
canonical-run selection), and 11 model configs including the gated H5 arms. **Eval:** `metrics.py`
(per-image integer count tables; split-level IoU from summed counts; fixed class set S; boundary-F1
descriptive-only), `stats.py` (the §5.6 paired image-bootstrap — `default_rng(0)` reset per
comparison, ONE index draw per replicate shared across both models and all per-class strata,
iou_zero rule with |S| denominator, plus-one p, percentile CI; the deterministic §5.7 H4 rule; Holm;
descriptive McNemar), `aggregate.py` (§7.4 per-image schema + store rows), `verdict.py` (§5.9
resolution → family comparisons → Holm → H0–H5 decisions; §5.8 H5 partial rule; honest H0),
`prereg.py` (PREREG seal: write-once + SHA-256 verify, enforced by `analyze_results.py`), and
numpy/cv2 overlay `plots.py`. **Scripts:** `run_experiment.py` (train or `--eval-only --h4`; §7.4
predictions contract; manifest extras incl. `resolved_test_gold_dir`, `class_weights`,
`weights_source`, `best_val_miou`; committed mirror under `experiments/manifests/`),
`analyze_results.py`, `merge_results.py`. **PREREG.md is sealed**, freezing α=0.10/Holm, seeds
(1414 / bootstrap 0), the pinned min3 test sets, and the **SAM region-oracle scoring rule** (each
SAM proposal takes its majority valid-GT class; uncovered pixels → soil; an explicit upper bound).

**Gate (MS3 + MS2-C, green):** 53 pytest tests pass (20 in `test_eval.py`), ruff + black clean;
`analyze_results.py` exits 0 and writes `verdicts.json` deciding **every** H0–H5 (all `deferred`
pending GPU runs — H5 deferral never blocks H1–H4) + `leaderboard.csv`; the CPU smoke
(`run_experiment.py --config configs/models/baseline.yaml --override data.max_train_images=64
train.max_epochs=1`) exits 0 and is contract-valid: 322 pred PNGs, 1,932 per-image rows, 8 store
rows, manifest + mirror committed. Smoke sanity: 1-epoch scratch TinyUNet on 64 images → test mIoU
0.056 (garbage, as expected — the pipeline, not the model, was under test).

**Adversarial review (capped at 10 agents per user directive: 3 lenses + 7 refuters).** Confirmed
and fixed 4 majors: (1) H4 runs' per-class store rows collided on DEDUP_KEYS across the MSL/MER
splits — merge would have silently deleted in-rover per-class values and mislabeled MER numbers
(per-class rows now inherit `in_rover`/`cross_rover`); (2) `--eval-only` manifests recorded
`best_val_miou: null`, crashing H4 subject selection with >1 candidate (val score now recovered
from the checkpoint's ModelCheckpoint state; verdict side is None-safe with timestamp fallback);
(3) H4's baseline-on-MER was picked by store row order, not canonical resolution (now
newest-by-manifest); (4) the `COMPARISONS` table was completely unpinned by tests (now pinned
field-by-field to §10). Also triaged the unverified overflow: H5 launch configs added,
`leaderboard()` now seed-filtered and gpu-preferring, multi-backbone scoring reads the newest run,
iou_zero/Holm-order test gaps closed, and the smoke val-cap deviation documented in §5.4.

**Next:** MS4 — `scripts/run_gpu.sh`, the V100 sweep (7 training arms + H4 evals + H5), merge-back,
`analyze_results.py` for real verdicts; then MS5 (paper).

## MS4 (prep) — V100 turnkey handoff script · branch `phase-ms4-runs`

Built `scripts/run_gpu.sh` — **bash-only** (LF endings enforced by `.gitattributes`; never run on
Windows) — implementing DEVPLAN §2 steps 1–9 as one idempotent command on the Ubuntu V100 node:
checkout `main` → venv (python3.11) → core deps → **cu121 torch 2.4.1 override** → extras
(segment-anything, timm) → editable install → **pretrained-weight pre-cache** while the box has
network (smp resnet34/resnet50 ImageNet + SegFormer b0/b2 ADE) → SAM ViT-B checkpoint fetch →
hard `profile=gpu_full` + `core OK` gate → dataset presence check (downloads the 16 GB merged
archive only if absent) → the **9 training arms** (resume markers under `experiments/.gpu_markers/`
so an interrupted sweep restarts where it stopped; a failed arm is recorded and the sweep
continues) → **H4 cross-rover evals** driven by the manifests (subject = highest `best_val_miou`
non-baseline run with a `best.ckpt`, plus the baseline MER reference; both via
`--eval-only <ckpt> --h4`) → the gated **H5 arms** (skip-and-log without HF_TOKEN/ckpt) →
`analyze_results.py` → a timestamped store export plus printed merge-back/commit instructions
(committing on the V100 must keep author = John Roth, no AI co-author). Verified on this box:
`bash -n` clean, pure-LF bytes, and the H4 selection helper tested against both the real
CPU-only tree (correctly selects nothing) and a synthetic 3-manifest gpu_full tree (correctly
picks the best subject + baseline). Also wired an optional `train.num_workers` config knob
through the Lightning DataModule and both eval loaders (default 0, verdict-neutral;
`run_gpu.sh` passes 8) so the 50-epoch sweep is not bottlenecked on single-threaded JPEG decode.

**Gate:** full suite still green after the wiring (53 tests, ruff, black). MS4 remains OPEN —
the V100 execution + merge-back are the remaining work; MS5 (paper) after that.
