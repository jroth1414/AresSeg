#!/usr/bin/env bash
# =============================================================================
# run_gpu.sh — V100 (Ubuntu/Linux) turnkey handoff (DEVPLAN section 2, steps 1-9).
#
#   bash scripts/run_gpu.sh
#
# Exact-lock environment + sealed protocol/data gates + verified/pinned weights + three-seed sweep
# + mean-validation H4 subject + deterministic references + analysis/export.
#
# Re-runs skip only config/git/code-matching manifests with every required artifact/result row.
# Secrets: the DINOv3 H5 member needs HF_TOKEN in .env; without it that member is deferred
# and the configured H5 partial-GPU rule applies. H1-H4 remain unaffected (DEVPLAN 5.4).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYBIN="${PYBIN:-python3.11}"       # base interpreter for the venv (pyproject: >=3.11,<3.12)
PY=".venv/bin/python"
NUM_WORKERS="${NUM_WORKERS:-8}"    # DataLoader workers (verdict-neutral; speeds JPEG decode)

if [ -n "${GPU_DEVICE:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
elif [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-$PWD/data/cache}"
export TMPDIR="${TMPDIR:-$RUNTIME_CACHE_ROOT/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RUNTIME_CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$RUNTIME_CACHE_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_ROOT/xdg}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUNTIME_CACHE_ROOT/cuda}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT="data/weights/sam/sam_vit_b_01ec64.pth"
SAM_EXPECTED_SHA256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
DATA_DIR="data/raw/ai4mars/ai4mars-dataset-merged-0.6"
DATA_ARCHIVE="data/raw/ai4mars/ai4mars-dataset-merged-0.6.zip"
SEEDS=(1414 1415 1416)
TRAIN_ARMS=(tiny_unet unet unet_scratch deeplabv3plus deeplabv3plus_scratch
            segformer_b0 segformer_b0_scratch segformer_b2 segformer_b2_scratch dinov3_sat)
FAILED=()

file_sha256() {
  "$PY" - "$1" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
print(digest)
PY
}
log() { printf '\n=== [run_gpu] %s ===\n' "$*"; }

# --- step 1: prepared-checkout provenance + venv (never mutates branch/worktree) -------------
log "step 1: inspect prepared checkout + venv"
GIT=(git -c "safe.directory=$PWD")
"${GIT[@]}" rev-parse --is-inside-work-tree >/dev/null
GIT_SHA="$("${GIT[@]}" rev-parse HEAD)"
GIT_REF="$("${GIT[@]}" rev-parse --abbrev-ref HEAD)"
log "prepared checkout ref=$GIT_REF commit=$GIT_SHA"
if [ -n "$("${GIT[@]}" status --porcelain)" ]; then
  log "NOTE: prepared checkout is dirty; protocol and runtime fingerprints must still match"
fi
if [ ! -x "$PY" ]; then
  "$PYBIN" -m venv .venv
fi

# --- steps 2-4: exact CUDA, main, and extras locks; local package without dependency drift ----
log "steps 2-4: exact lock installs (cu121, main, extras, local --no-deps)"
"$PY" -m pip install --disable-pip-version-check -r requirements-cuda121.lock.txt
"$PY" -m pip install --disable-pip-version-check -r requirements.lock.txt
"$PY" -m pip install --disable-pip-version-check -r requirements-extras.lock.txt
"$PY" -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e .
"$PY" -m pip check

# --- step 5: pre-cache pretrained encoder/backbone weights while the box has network ---------
log "step 5: pre-cache encoder weights (locked SMP package + pinned MiT revisions)"
"$PY" - <<'PY'
import segmentation_models_pytorch as smp
from aresseg.models.zoo import SEGFORMER_SAFETENSORS, _verified_segformer_snapshot
from aresseg.utils.config import load_yaml
from transformers import SegformerModel

smp.Unet("resnet34", encoder_weights="imagenet")
smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet")
for variant in ("b0", "b2"):
    model = load_yaml(f"configs/models/segformer_{variant}.yaml")["model"]
    pin = SEGFORMER_SAFETENSORS[variant]
    if model.get("revision") != pin["revision"]:
        raise SystemExit(f"MiT-{variant} config/code revision pin mismatch")
    if model.get("weights_filename") != "model.safetensors":
        raise SystemExit(f"MiT-{variant} must configure model.safetensors")
    if model.get("weights_sha256") != pin["sha256"]:
        raise SystemExit(f"MiT-{variant} config/code sha256 pin mismatch")
    snapshot = _verified_segformer_snapshot(variant, model["revision"])
    SegformerModel.from_pretrained(
        snapshot, use_safetensors=True, local_files_only=True
    )
print("pre-cache OK")
PY
# --- step 6: SAM ViT-B checkpoint (H5) -------------------------------------------------------
log "step 6: SAM checkpoint"
SAM_CONFIG_PIN="$("$PY" -c 'from aresseg.utils.config import load_yaml; m=load_yaml("configs/models/sam.yaml")["model"]; print(m["sam_checkpoint"] + "|" + m["sam_checkpoint_sha256"])')"
[ "$SAM_CONFIG_PIN" = "$SAM_CKPT|$SAM_EXPECTED_SHA256" ] || { echo "FATAL: SAM config path/hash pin differs from run_gpu.sh"; exit 1; }
SAM_TMP="${SAM_CKPT}.part"
if [ -s "$SAM_CKPT" ]; then
  SAM_ACTUAL_SHA256="$(file_sha256 "$SAM_CKPT")"
  if [ "$SAM_ACTUAL_SHA256" != "$SAM_EXPECTED_SHA256" ]; then
    SAM_INVALID="${SAM_CKPT}.invalid-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$SAM_CKPT" "$SAM_INVALID"
    log "WARN: quarantined SAM checkpoint with sha256=$SAM_ACTUAL_SHA256"
  fi
fi
if [ ! -s "$SAM_CKPT" ]; then
  mkdir -p "$(dirname "$SAM_CKPT")"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-all-errors -o "$SAM_TMP" "$SAM_URL" || log "WARN: SAM download failed — sam arm will skip-and-log"
  else
    wget -O "$SAM_TMP" "$SAM_URL" || log "WARN: SAM download failed — sam arm will skip-and-log"
  fi
  if [ -s "$SAM_TMP" ]; then
    SAM_ACTUAL_SHA256="$(file_sha256 "$SAM_TMP")"
    if [ "$SAM_ACTUAL_SHA256" = "$SAM_EXPECTED_SHA256" ]; then
      mv "$SAM_TMP" "$SAM_CKPT"
      log "SAM checkpoint verified (sha256=$SAM_ACTUAL_SHA256)"
    else
      SAM_INVALID="${SAM_TMP}.invalid-$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$SAM_TMP" "$SAM_INVALID"
      log "WARN: SAM download sha256=$SAM_ACTUAL_SHA256, expected=$SAM_EXPECTED_SHA256; arm will skip"
    fi
  fi
else
  log "SAM checkpoint verified (sha256=$SAM_EXPECTED_SHA256; $(du -h "$SAM_CKPT" | cut -f1))"
fi

# --- step 7: environment gate (must be the GPU profile) --------------------------------------
log "step 7: check_env gate (expect profile=gpu_full + core OK)"
"$PY" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is not available to PyTorch")
props = torch.cuda.get_device_properties(0)
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"selected_gpu={props.name} compute={props.major}.{props.minor} memory_gib={props.total_memory / 2**30:.1f}")
PY
ENV_OUT="$("$PY" scripts/check_env.py)"
echo "$ENV_OUT"
echo "$ENV_OUT" | grep -q '^profile=gpu_full' || { echo "FATAL: not on the gpu_full profile (cuda absent?)"; exit 1; }
echo "$ENV_OUT" | tail -n 1 | grep -q '^core OK$' || { echo "FATAL: check_env did not end with 'core OK'"; exit 1; }

log "Protocol V3 gate: historical chain + executable snapshot must match live inputs"
"$PY" - <<'PY'
from pathlib import Path

from aresseg.eval.prereg import verify_protocol
from aresseg.utils.config import load_yaml

root = Path.cwd()
verify_protocol(
    load_yaml(root / "configs" / "hypotheses.yaml"),
    load_yaml(root / "configs" / "data.yaml"),
    repo_root=root,
)
print("Protocol V3 verification OK")
PY

# --- dataset precondition: source archive + extraction + strict protocol preflight -----------
if [ ! -s "$DATA_ARCHIVE" ]; then
  log "source archive absent — downloading AI4Mars merged (~16 GB, resumable)"
  "$PY" scripts/download_data.py --out data/raw/ai4mars --which merged --no-extract
fi
if [ ! -d "$DATA_DIR/msl/ncam" ]; then
  log "dataset extraction absent — extracting verified source archive"
  "$PY" -m zipfile -e "$DATA_ARCHIVE" data/raw/ai4mars
fi
log "dataset preflight: counts, pairing, masks, splits, archive MD5, index fingerprints"
"$PY" scripts/check_data.py --archive "$DATA_ARCHIVE" --json-out data/preflight.json
[ -z "${HF_TOKEN:-}" ] && ! grep -qE '^HF_TOKEN=.+' .env 2>/dev/null \
  && log "NOTE: HF_TOKEN not set — the dinov3_sat arm will skip-and-log (H5 deferred)"

# --- step 8a: learned arms; resume only from matching manifests + complete artifacts ----------
log "step 8a: learned arms (${#TRAIN_ARMS[*]} configs x ${#SEEDS[*]} seeds)"
for seed in "${SEEDS[@]}"; do
  for arm in "${TRAIN_ARMS[@]}"; do
    overrides=("train.num_workers=${NUM_WORKERS}" "data.seed=${seed}")
    if completed="$("$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
          --override "${overrides[@]}" --check-complete)"; then
      log "arm $arm seed=$seed already complete: $completed"
      continue
    fi
    log "training arm: $arm seed=$seed"
    if ! "$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
          --override "${overrides[@]}"; then
      log "ARM FAILED: $arm seed=$seed (continuing)"
      FAILED+=("${arm}:seed${seed}")
    fi
  done
done

# --- step 8b: H4 subject = highest mean val-mIoU over complete pretrained 3-seed candidates ---
log "step 8b: H4 cross-rover evals (--eval-only --h4)"
H4_PICKS="$("$PY" - "$NUM_WORKERS" <<'PY'
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "scripts").resolve()))
from aresseg.utils.capabilities import detect
from aresseg.utils.config import load_config
from run_experiment import _matching_complete_run, _resolved_git_sha, _runtime_code_fingerprint

CONFIGS = {
    ("unet", "resnet34", "pretrained"): "unet",
    ("deeplabv3plus", "resnet50", "pretrained"): "deeplabv3plus",
    ("segformer", "mit-b0", "pretrained"): "segformer_b0",
    ("segformer", "mit-b2", "pretrained"): "segformer_b2",
}
SEEDS = (1414, 1415, 1416)
num_workers = int(sys.argv[1])
profile = detect().profile
git_sha = _resolved_git_sha()
code_fingerprint = _runtime_code_fingerprint()
resolved = {}
missing = []

for key, cfg_name in CONFIGS.items():
    for seed in SEEDS:
        cfg = load_config(
            f"configs/models/{cfg_name}.yaml",
            overrides=[
                f"train.num_workers={num_workers}",
                f"data.seed={seed}",
            ],
            base_paths=[str(Path.cwd() / "configs" / "data.yaml")],
        )
        run = _matching_complete_run(
            cfg,
            profile=profile,
            model_name=key[0],
            git_sha=git_sha,
            code_fingerprint=code_fingerprint,
        )
        if run is None:
            missing.append(f"{cfg_name}:seed{seed}")
            continue
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        actual_key = (manifest.get("model"), manifest.get("backbone"), manifest.get("variant"))
        if actual_key != key or manifest.get("seed") != seed:
            raise RuntimeError(
                f"H4 resolver returned mismatched run {run}: {actual_key}, seed={manifest.get('seed')}"
            )
        score = manifest.get("best_val_miou")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise RuntimeError(f"H4 candidate {run} has invalid best_val_miou={score!r}")
        resolved[(key, seed)] = (float(score), str(run / "best.ckpt"))

if missing:
    raise SystemExit(f"H4 requires every configured three-seed candidate; missing: {missing}")

means = []
for key, cfg_name in CONFIGS.items():
    mean_score = sum(resolved[(key, seed)][0] for seed in SEEDS) / len(SEEDS)
    means.append((mean_score, cfg_name, key))
best_mean = max(item[0] for item in means)
winners = [item for item in means if item[0] == best_mean]
if len(winners) != 1:
    raise SystemExit(f"exact H4 mean-val tie requires a preregistered resolution: {winners}")
_, cfg_name, winner = winners[0]
for seed in SEEDS:
    print(f"{cfg_name}|{resolved[(winner, seed)][1]}|{seed}")
PY
)"
if [ -z "$H4_PICKS" ]; then
  echo "FATAL: no complete three-seed pretrained H4 candidate set"; exit 1
else
  while IFS='|' read -r cfg ckpt seed; do
    [ -z "$cfg" ] && continue
    log "H4 eval: config=$cfg seed=$seed ckpt=$ckpt"
    overrides=("train.num_workers=${NUM_WORKERS}" "data.seed=${seed}")
    if completed="$("$PY" scripts/run_experiment.py --config "configs/models/${cfg}.yaml" \
          --override "${overrides[@]}" --eval-only "$ckpt" --h4 --check-complete)"; then
      log "H4 eval already complete: $completed"
    elif ! "$PY" scripts/run_experiment.py --config "configs/models/${cfg}.yaml" \
          --override "${overrides[@]}" --eval-only "$ckpt" --h4; then
      log "H4 EVAL FAILED: $cfg seed=$seed"
      FAILED+=("h4:${cfg}:seed${seed}")
    fi
  done <<< "$H4_PICKS"
fi

# --- step 8c: deterministic references (one seed-1414 artifact each) -------------------------
log "step 8c: deterministic majority and SAM references (seed 1414 only)"
for arm in majority sam; do
  overrides=("train.num_workers=${NUM_WORKERS}" "data.seed=1414")
  if completed="$("$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
        --override "${overrides[@]}" --check-complete)"; then
    log "reference $arm already complete: $completed"
  elif ! "$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
        --override "${overrides[@]}"; then
    log "REFERENCE FAILED: $arm"
    FAILED+=("reference:$arm")
  fi
done

log "H4 majority cross-rover reference (seed 1414 only)"
if completed="$("$PY" scripts/run_experiment.py --config configs/models/majority.yaml \
      --override "train.num_workers=${NUM_WORKERS}" "data.seed=1414" --h4 --check-complete)"; then
  log "H4 majority already complete: $completed"
elif ! "$PY" scripts/run_experiment.py --config configs/models/majority.yaml \
      --override "train.num_workers=${NUM_WORKERS}" "data.seed=1414" --h4; then
  FAILED+=("h4:majority")
fi

# --- step 9: verdicts on this box + export the store for merge-back --------------------------
log "step 9: analyze + export"
"$PY" scripts/analyze_results.py --store experiments/results_store.parquet \
  --hypotheses configs/hypotheses.yaml --out experiments/analysis/
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT="experiments/gpu_results_store_${STAMP}.parquet"
cp experiments/results_store.parquet "$EXPORT"
log "canonical results store: experiments/results_store.parquet"
log "timestamped export: $EXPORT"


if [ "${#FAILED[@]}" -gt 0 ]; then
  log "COMPLETED WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
log "all arms completed; no branch, commit, or remote state was changed"
