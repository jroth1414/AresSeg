# marsseg — Terrain-Aware Semantic Segmentation for Mars Rover Drivability

**Course:** JHU EN.705.742 — Advanced Applied Machine Learning · **Author:** John Roth

Deep semantic segmentation of Martian terrain (*soil, bedrock, sand, big rock*) from rover camera
images — the core perception task for autonomous **drivability** — on NASA's **AI4Mars** dataset. We
compare CNN (U-Net, DeepLabV3+) vs transformer (SegFormer) architectures, quantify the value of
**ImageNet transfer**, test **cross-rover generalization** (Curiosity → Opportunity/Spirit), and
benchmark a **foundation-model reference** (SAM / DINOv3), under a leakage-safe, pre-registered,
significance-tested protocol.

See **[`DEVPLAN.md`](DEVPLAN.md)** for the full plan (research question, hypotheses H0–H5, method,
data protocol, phases, rubric mapping) and **[`docs/DEVLOG.md`](docs/DEVLOG.md)** for the build trail.

## Hypotheses (summary)

| | Hypothesis |
|---|---|
| **H0** | No deep model beats a simple baseline at mIoU. |
| **H1** | A deep model (U-Net/DeepLab/SegFormer) beats the baseline (paired-bootstrap significant). |
| **H2** | ImageNet-pretrained encoder beats identical from-scratch. |
| **H3** | SegFormer vs U-Net/DeepLab — which wins overall and per class. |
| **H4** | A model generalizes across rovers (Curiosity → Opportunity/Spirit) with a bounded mIoU drop. |
| **H5** | A foundation model is a useful reference: DINOv3 (Earth-satellite SAT-493M weights) backbone + head, and SAM zero-shot. |

## Setup (Windows / CPU)

```powershell
C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/check_env.py    # -> profile=windows_cpu, core OK
```

The full pipeline + smoke tests run CPU-only. Full model training runs on a CUDA GPU (the V100) via
`scripts/run_gpu.sh`; the SAM/DINOv3 foundation models live in `requirements-extras.txt`.

## Status

MS0 (scaffold), MS1 (AI4Mars data pipeline), MS2 (models), and MS3 (configs + evaluation stack)
are **done**. The data layer is verified against the on-disk merged-0.6 layout (camera-aware
`build_index`, pinned min3 gold test sets: 16,064 train / 322 MSL test / 204 MER test,
camera-qualified `name` join keys, by-image splits). The model zoo covers
baseline/U-Net/DeepLabV3+/SegFormer plus the gated H5 foundation arms (DINOv3-SAT frozen backbone
+ head, SAM zero-shot; skip-and-log on CPU/missing weights). The evaluation stack implements the
pre-registered protocol end-to-end: per-image count tables, the paired image-bootstrap
(n=10,000, seed 0) with Holm correction, the deterministic H4 cross-rover rule, canonical-run
selection, and `analyze_results.py` → `verdicts.json` (H0–H5, currently all *deferred* pending
GPU runs). `experiments/PREREG.md` is sealed (SHA-256 recorded) and the CPU smoke run is
committed. Next: MS4 — `run_gpu.sh` + the V100 training sweep, then merge-back and verdicts.
