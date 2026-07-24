# Protocol V4 Amendment — AresSeg Rename and Administrative Checkpoint Reuse

**Amendment date:** 2026-07-24  
**Timing:** after all preregistered MSL training runs completed, before H4 MER evaluation and
final confirmatory analysis.

## Purpose

The public project and Python package were renamed from `marsseg` to `AresSeg` / `aresseg`
because the original name conflicts with an existing project. The rename is administrative:
the package directory and imports changed, but the trained architectures, tensor operations,
data, splits, preprocessing, loss, optimization, metrics, seeds, and hypothesis rules did not.
Protocols V1--V3 and their `marsseg.protocol_snapshot.*` schema identifiers remain immutable.

## Completed training inventory

The confirmatory inventory contains all 30 learned runs: ten preregistered model/initialization
arms at seeds 1414, 1415, and 1416. The runs were produced at commits
`c188b320d700e01c8ffb37330e30f188862ad995`,
`c2c2860f40626413eb95dd9bfec3d492fdde9035`, and
`3e52372ce7ea6f923dddec95338384a6dd3693bd`. Every run records the same result-driving runtime
fingerprint:

`bebd8b2dae8fb62a087ded6f5334dbf19bfde1a157bad743b17471ba200325d3`.

All 30 runs have complete manifests, best checkpoints, training metrics, MSL per-image outputs,
and canonical result rows. No test-set outcome was used to choose this amendment.

## Narrow reuse rule for H4

H4 requires reusing the validation-selected pretrained checkpoint rather than retraining.
Protocol V4 permits the H4 resolver to accept a completed Protocol V3 training run only when
all of the following match:

1. the resolved configuration hash, model, backbone, variant, seed, profile, and gold split;
2. one of the three exact training commits listed above;
3. the exact runtime fingerprint listed above;
4. complete training, best-checkpoint, MSL per-image, training-metric, manifest, and canonical
   result-store artifacts; and
5. a finite recorded best-validation mIoU.

This exception applies only while resolving the already-completed training checkpoints for H4.
New H4, deterministic reference, and final-analysis artifacts remain bound to the live Protocol
V4 code and configuration. It does not authorize reuse of a run with an unknown commit,
fingerprint, configuration, or incomplete artifact set.

## Exploratory MPBA screen

The Mars Perspective-Biased Adapter (MPBA) seed-1414 validation screen is exploratory and
outside H0--H5. Its promotion rule rejected both perspective variants, so no variant was
promoted to gold-test evaluation and no MPBA result contributes to confirmatory claims. Compact
screen evidence is archived under `experiments/mpba/` for transparency.

## Unchanged confirmatory protocol

This amendment does not change the research hypotheses, comparison selectors, decision rules,
class set, ignore-label behavior, AI4Mars data, leakage controls, preprocessing, model
architectures, pretrained weights, training hyperparameters, seeds, test sets, metrics,
bootstrap procedure, confidence level, significance level, or Holm correction. Protocol V4
chains the immutable V3 snapshot and sidecar byte-for-byte and seals the live AresSeg paths.
