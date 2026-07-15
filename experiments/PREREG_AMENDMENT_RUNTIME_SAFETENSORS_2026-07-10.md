# Runtime compatibility amendment — 2026-07-10 UTC (Protocol V3)

## Timing and absence of confirmatory information

This amendment was made after the first full-sweep startup attempt but before any experiment,
training, validation selection, or test-set evaluation began. That startup passed the exact-lock
dependency installation and Protocol V2 verification. Its pretrained-MiT cache step then exposed
the runtime incompatibility described below; the old script warned and continued. The operator
interrupted the process while the strict AI4Mars data preflight was still running, before the
script reached its first `training arm` stage. The log contains no `run_id` or `training arm`
event, and the attempt produced no confirmatory run, training metric, validation metric,
per-image prediction, leaderboard row, hypothesis verdict, or test-set result. Therefore no
outcome was available to motivate or tune this amendment.

## Observed runtime incompatibility

The exact environment locks Transformers 5.12.1 with PyTorch 2.4.1. Loading the original
PyTorch `.bin` MiT checkpoints raised a Transformers `ValueError` beginning:

> Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now
> require users to upgrade torch to at least v2.6 …

Transformers identifies this restriction with CVE-2025-32434 and exempts weights loaded from
the non-pickle safetensors format. Raising the PyTorch version would change the sealed runtime
stack. The narrow compatibility correction instead pins each official Hugging Face
safetensors-conversion commit whose exact parent is the originally sealed model revision and
requires the named `model.safetensors` blob by SHA-256.

## Exact weight-representation substitutions

| Arm | Protocol V2 revision (parent) | Protocol V3 safetensors conversion revision | Required file | SHA-256 |
|---|---|---|---|---|
| MiT-B0 | `80983a413c30d36a39c20203974ae7807835e2b4` | `25ce79d97e6d9d509ed12e17cb2eb89b0a83a2dc` | `model.safetensors` | `3e5ad9cd1dd8ecf8305c23fcdf01ef241f08c7b2dddacb6ec7de5a887188798a` |
| MiT-B2 | `3bb39e8739149c3777d0325349b2a6c32c6413db` | `d15ed1f9ae92346f6a6067dbb490a62494ae0d28` | `model.safetensors` | `b3ad4dd552f9e1b871f46666f39187414133b861e3d07eda016600230f8a1ad6` |

At both conversion revisions, the model configuration, preprocessor configuration, and original
`.bin` blobs are unchanged from the exact parent revision; the child commit adds only safe
serialization metadata and the safetensors file. Thus the model architecture and learned
parameters are unchanged. Runtime loading is fail-closed: the configured filename, conversion
revision, and downloaded bytes must match these pins.

## Unchanged confirmatory protocol

This amendment changes only the serialization format and fail-closed loading path for the two
pretrained MiT arms. It does not change the hypotheses, comparison selectors, AI4Mars data or
splits, class definitions, model architectures, optimization settings, seeds, number of seeds,
metrics, resampling unit or count, significance thresholds, multiplicity correction, selection
rules, or decision rules. The legacy `PREREG.md`, the 2026-07-10 Protocol V2 amendment, and both
Protocol V2 seal files remain immutable historical records and are incorporated byte-for-byte
into the Protocol V3 chain.
