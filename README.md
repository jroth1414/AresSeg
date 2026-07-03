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

MS0 (scaffold), MS1 (AI4Mars data pipeline), and MS2 (models) are **done**. MS1 is verified against
the on-disk merged-0.6 layout: camera-aware `build_index` (MSL ncam-scoped; MER images from
`images/{eff,test}`), pinned `masked-gold-min3-100agree` test sets (16,064 train / 322 MSL test /
204 MER test), canonical camera-qualified `name` join keys, by-image splits. MS2 covers the model
zoo (baseline/U-Net/DeepLabV3+/SegFormer), CE+Dice loss, Lightning training module, and the gated
H5 foundation arms (`foundation.py`: DINOv3-SAT frozen backbone + head, SAM zero-shot,
skip-and-log on CPU/missing weights). Next: MS3 (configs, eval/stats, `run_experiment.py`,
pre-registration).
