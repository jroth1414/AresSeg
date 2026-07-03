#!/usr/bin/env bash
# =============================================================================
# run_gpu.sh — V100 (Ubuntu/Linux) turnkey handoff (DEVPLAN section 2, steps 1-9).
#
#   bash scripts/run_gpu.sh
#
# One command: env build -> weight pre-cache -> SAM checkpoint -> gpu_full gate ->
# 9 training arms -> H4 cross-rover evals (subject + baseline) -> H5 foundation arms ->
# analyze -> export a timestamped store for merge-back on the dev box.
#
# Bash only (never run on Windows/PowerShell). Idempotent: re-running skips the venv,
# downloads, and any arm whose gpu_full manifest + best.ckpt already exist.
# Secrets: H5 DINOv3 needs HF_TOKEN in .env (license accepted on Hugging Face); if absent
# the arm skip-and-logs and H5 stays deferred — H1-H4 are unaffected (DEVPLAN 5.4).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYBIN="${PYBIN:-python3.11}"       # base interpreter for the venv (pyproject: >=3.11,<3.12)
PY=".venv/bin/python"
NUM_WORKERS="${NUM_WORKERS:-8}"    # DataLoader workers (verdict-neutral; speeds JPEG decode)
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT="data/weights/sam/sam_vit_b_01ec64.pth"
DATA_DIR="data/raw/ai4mars/ai4mars-dataset-merged-0.6"
TRAIN_ARMS=(baseline unet unet_scratch deeplabv3plus deeplabv3plus_scratch
            segformer_b0 segformer_b0_scratch segformer_b2 segformer_b2_scratch)
FAILED=()

log() { printf '\n=== [run_gpu] %s ===\n' "$*"; }

# --- step 1: canonical branch + venv --------------------------------------------------------
log "step 1: git checkout main + venv"
git checkout main
if [ ! -x "$PY" ]; then
  "$PYBIN" -m venv .venv
fi
"$PY" -m pip install -U pip

# --- steps 2-4: deps (CUDA torch overrides the CPU wheel), extras, editable install ----------
log "steps 2-4: pip installs (core, cu121 torch, extras, -e .)"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install -r requirements-extras.txt
"$PY" -m pip install -e .

# --- step 5: pre-cache pretrained encoder/backbone weights while the box has network ---------
log "step 5: pre-cache ImageNet/ADE weights (smp resnet34/resnet50, SegFormer b0/b2)"
if ! "$PY" - <<'PY'
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

smp.Unet("resnet34", encoder_weights="imagenet")
smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet")
for b in ("b0", "b2"):
    SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/segformer-{b}-finetuned-ade-512-512", num_labels=4, ignore_mismatched_sizes=True
    )
print("pre-cache OK")
PY
then
  log "WARN: weight pre-cache failed (offline?) — pretrained arms will fail unless cached"
fi

# --- step 6: SAM ViT-B checkpoint (H5) -------------------------------------------------------
log "step 6: SAM checkpoint"
if [ ! -s "$SAM_CKPT" ]; then
  mkdir -p "$(dirname "$SAM_CKPT")"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$SAM_CKPT" "$SAM_URL" || log "WARN: SAM download failed — sam arm will skip-and-log"
  else
    wget -O "$SAM_CKPT" "$SAM_URL" || log "WARN: SAM download failed — sam arm will skip-and-log"
  fi
else
  log "SAM checkpoint present ($(du -h "$SAM_CKPT" | cut -f1))"
fi

# --- step 7: environment gate (must be the GPU profile) --------------------------------------
log "step 7: check_env gate (expect profile=gpu_full + core OK)"
ENV_OUT="$("$PY" scripts/check_env.py)"
echo "$ENV_OUT"
echo "$ENV_OUT" | grep -q '^profile=gpu_full' || { echo "FATAL: not on the gpu_full profile (cuda absent?)"; exit 1; }
echo "$ENV_OUT" | tail -n 1 | grep -q '^core OK$' || { echo "FATAL: check_env did not end with 'core OK'"; exit 1; }

# --- dataset precondition (16 GB; download only if the extracted tree is absent) --------------
if [ ! -d "$DATA_DIR/msl/ncam" ]; then
  log "dataset absent — downloading AI4Mars merged (~16 GB, one-time)"
  "$PY" scripts/download_data.py --out data/raw/ai4mars --which merged
fi
[ -z "${HF_TOKEN:-}" ] && ! grep -qE '^HF_TOKEN=.+' .env 2>/dev/null \
  && log "NOTE: HF_TOKEN not set — the dinov3_sat arm will skip-and-log (H5 deferred)"

# --- step 8a: training arms (marker files make re-runs resume where they stopped) ------------
log "step 8a: training arms (${#TRAIN_ARMS[*]} configs)"
mkdir -p experiments/.gpu_markers
for arm in "${TRAIN_ARMS[@]}"; do
  marker="experiments/.gpu_markers/${arm}.done"
  if [ -f "$marker" ]; then
    log "arm $arm already done (marker) — skipping"
    continue
  fi
  log "training arm: $arm"
  if "$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
        --override "train.num_workers=${NUM_WORKERS}"; then
    touch "$marker"
  else
    log "ARM FAILED: $arm (continuing so the rest of the sweep still runs)"
    FAILED+=("$arm")
  fi
done

# --- step 8b: H4 cross-rover evals — subject (highest best_val_miou) + baseline ---------------
log "step 8b: H4 cross-rover evals (--eval-only --h4)"
H4_PICKS="$("$PY" - <<'PY'
import json
from pathlib import Path

CONFIGS = {
    ("baseline", "none", "scratch"): "baseline",
    ("unet", "resnet34", "pretrained"): "unet",
    ("unet", "resnet34", "scratch"): "unet_scratch",
    ("deeplabv3plus", "resnet50", "pretrained"): "deeplabv3plus",
    ("deeplabv3plus", "resnet50", "scratch"): "deeplabv3plus_scratch",
    ("segformer", "mit-b0", "pretrained"): "segformer_b0",
    ("segformer", "mit-b0", "scratch"): "segformer_b0_scratch",
    ("segformer", "mit-b2", "pretrained"): "segformer_b2",
    ("segformer", "mit-b2", "scratch"): "segformer_b2_scratch",
}
runs = []
for mf in sorted(Path("experiments").glob("*/manifest.json")):
    m = json.loads(mf.read_text(encoding="utf-8"))
    ckpt = mf.parent / "best.ckpt"
    if (
        m.get("profile") == "gpu_full"
        and "train" in (m.get("stages_completed") or [])
        and isinstance(m.get("best_val_miou"), (int, float))
        and ckpt.is_file()
    ):
        key = (m.get("model"), m.get("backbone"), m.get("variant"))
        if key in CONFIGS:
            runs.append((m["best_val_miou"], m.get("timestamp_utc", ""), CONFIGS[key], str(ckpt), m["model"]))
subjects = [r for r in runs if r[4] != "baseline"]
baselines = [r for r in runs if r[4] == "baseline"]
picks = []
if subjects:
    picks.append(max(subjects))          # highest best_val_miou (ties -> newest timestamp)
if baselines:
    picks.append(max(baselines))         # baseline_on_MER reference (5.7)
for _, _, cfg, ckpt, _ in picks:
    print(f"{cfg}|{ckpt}")
PY
)"
if [ -z "$H4_PICKS" ]; then
  log "WARN: no eligible gpu_full training runs found — skipping H4 evals"
else
  while IFS='|' read -r cfg ckpt; do
    [ -z "$cfg" ] && continue
    log "H4 eval: config=$cfg ckpt=$ckpt"
    if ! "$PY" scripts/run_experiment.py --config "configs/models/${cfg}.yaml" \
          --override "train.num_workers=${NUM_WORKERS}" --eval-only "$ckpt" --h4; then
      log "H4 EVAL FAILED: $cfg"
      FAILED+=("h4:$cfg")
    fi
  done <<< "$H4_PICKS"
fi

# --- step 8c: H5 foundation arms (gated; they skip-and-log rather than fail) -----------------
log "step 8c: H5 foundation arms (dinov3_sat, sam)"
for arm in dinov3_sat sam; do
  if ! "$PY" scripts/run_experiment.py --config "configs/models/${arm}.yaml" \
        --override "train.num_workers=${NUM_WORKERS}"; then
    log "H5 ARM CRASHED (gating should have skipped instead): $arm"
    FAILED+=("h5:$arm")
  fi
done

# --- step 9: verdicts on this box + export the store for merge-back --------------------------
log "step 9: analyze + export"
"$PY" scripts/analyze_results.py --store experiments/results_store.parquet \
  --hypotheses configs/hypotheses.yaml --out experiments/analysis/
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT="experiments/gpu_results_store_${STAMP}.parquet"
cp experiments/results_store.parquet "$EXPORT"

log "DONE. Merge-back + commit checklist:"
cat <<EOF
  1) On this box, commit the committed-by-contract artifacts (author = John Roth, NO AI co-author):
       git add experiments/manifests experiments/results_store.parquet experiments/results_store.csv \\
               experiments/*/manifest.json
       git commit -m "MS4: V100 sweep results (manifests + store)" && git push
  2) On the dev box (after pulling), dedup-merge this export into the canonical store:
       .venv\\Scripts\\python.exe scripts/merge_results.py --incoming ${EXPORT} --into experiments/results_store.parquet
  3) Re-run the verdicts there:
       .venv\\Scripts\\python.exe scripts/analyze_results.py
EOF

if [ "${#FAILED[@]}" -gt 0 ]; then
  log "COMPLETED WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
log "all arms completed"
