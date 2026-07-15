# Preregistration amendment — Protocol V2 (2026-07-10)

This amendment preserves `experiments/PREREG.md` byte-for-byte. Its recorded SHA-256 remains
`79158132c24c583c3852630371a0602f52990b7d800c35aefa749e026f3fed7d`. A 64-image,
one-epoch Windows CPU smoke result existed before this amendment; it was a pipeline check, was
not used to decide H0–H5, and is excluded by the `gpu_full` canonical-run filter. No
confirmatory GPU result existed when this amendment was made.

## Why this amendment exists

The prose seal did not bind the executable comparison selectors, model configurations, or
result-driving implementation. Protocol V2 adds a canonical machine-readable snapshot at
`experiments/manifests/PROTOCOL_V2.json`, with its own SHA-256 sidecar. Analysis now fails closed
unless the legacy seal, this amendment, the complete data and hypothesis configurations, every
model YAML, and the exact data/preflight/model/training/evaluation/configuration/capability/
manifest/result-store/orchestration code all match that snapshot. Exact environment lockfiles stay
under the separate repository-integrity/reproducibility contract; each run manifest records the
environment actually used.

## Clarifications and corrections made before confirmatory evaluation

1. Family A/H1 and H0 use a genuine non-deep reference: the constant training-set majority
   predictor (class 0, soil). The old learned Tiny U-Net is named `tiny_unet`; it is not the
   simple baseline for H1.
2. Family E compares DINOv3-SAT and SAM with `tiny_unet`, an explicitly named learned reference.
   SAM is the preregistered ground-truth region-assignment oracle and is labeled
   `region_oracle_upper_bound`, never deployable “zero-shot” semantic segmentation.
3. H4 compares the validation-selected best conventional subject with the majority predictor on
   MER. Subject candidates and the majority selector are executable YAML, not hidden code.
4. SegFormer “pretrained” means an ImageNet-pretrained MiT backbone (`nvidia/mit-b0` or
   `nvidia/mit-b2`) with a newly initialized task head; it does not transfer an ADE20K semantic
   segmentation head.
5. Primary inference includes training variability. Every learned confirmatory arm must have the
   complete seed set `[1414, 1415, 1416]`. Deterministic majority and SAM-oracle predictions use
   one sealed seed-1414 artifact, reused as the paired side for each learned seed; duplicating
   their computation would add no training uncertainty. The observed effect is the equal-weight
   mean of seed-level effects. Each bootstrap replicate samples training seeds with replacement,
   then samples images with replacement independently inside every sampled seed; model A and B
   share each image draw. Missing learned-arm seeds defer the comparison rather than silently
   falling back to one seed.
   The executable `resampling_unit` is `seed_then_image`. Descriptive leaderboard output contains
   explicit per-seed `gpu_full` rows and per-seed CIs; it does not average CI endpoints or replace
   the hierarchical hypothesis analysis.
6. Family C is one confirmatory Holm family containing both overall comparisons and every
   fixed-set per-class test emitted for both comparisons. Adjusted p-values and decisions are
   reported for overall and per-class scopes. H3 is supported when at least one test in this
   family is significant, with direction taken from its observed signed effect. Both selectors
   pin SegFormer MiT-B2; the comparators are U-Net/ResNet-34 and DeepLabV3+/ResNet-50. H3 is a
   comparison of these configured systems, not an architecture-only causal effect or a claim of
   exact capacity matching. Total and trainable parameter counts are reported from run manifests.
7. Failure to reject Family A is reported for H0 as `fail_to_reject`, not as evidence supporting
   or establishing H0.
8. H2 can establish only that pretrained initialization helps at least one tested architecture
   under the sealed rule. It does not establish a universal benefit of transfer learning.

These changes close implementation and terminology gaps before the preregistered full GPU sweep;
they do not reinterpret any confirmatory result after seeing it.
