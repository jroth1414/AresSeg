# Pre-registration — marsseg (frozen BEFORE any test-set number is computed)

- Significance: alpha = 0.1, correction = holm (within family), CI level = 0.9 (percentile).
- Primary metric: miou (macro over the fixed class set); per-class metric: iou.
- Descriptive-only (NEVER tested): pixel_acc, boundary_f1.
- Bootstrap: n_resamples = 10000, unit = image, seed = 0 (reset per comparison), p = plus_one, empty fixed-set class in a resample contributes IoU = 0.
- Training/split seed = 1414; by-image splits, val_frac = 0.2.
- MSL test set: msl/ncam/labels/test/masked-gold-min3-100agree (n = 322).
- MER test set (H4): mer/labels/test/masked-gold-min3-100agree (n = 204); MER is never trained on.

## Families

- **A**: baseline_vs_unet, baseline_vs_deeplabv3plus, baseline_vs_segformer
- **B**: unet_pretrained_vs_scratch, deeplabv3plus_pretrained_vs_scratch, segformer_pretrained_vs_scratch
- **C**: segformer_vs_unet, segformer_vs_deeplabv3plus
- **D**: best_in_rover_vs_cross_rover
- **E**: dinov3_sat_vs_baseline, sam_zeroshot_vs_baseline

## Hypotheses & decision rules

- **H1**: reject_H0 if holm_p < 0.10 for >=1 member (delta>0)
- **H2**: support if holm_p < 0.10 for >=1 member (delta>0)
- **H3**: support a direction (sign(observed delta)) iff holm_p < 0.10 for that member
- **H4**: support iff drop < 0.15 AND cross_rover_ci_low > baseline_on_MER_miou
- **H5**: {'family': 'E', 'gated': True, 'decided_on_profile': 'gpu_full', 'on_missing_gpu': 'deferred', 'partial_gpu_rule': 'holm over ok members only; support iff >=1 ok member holm_p<0.10 (delta>0); reject if all ok fail; deferred if zero ok members'}

## H5 SAM scoring rule (frozen)

SAM emits class-AGNOSTIC region proposals and AI4Mars has no prompt channel. The SAM
zero-shot arm is scored with the **region-oracle assignment**: each SAM proposal takes
the majority ground-truth class among its valid (non-ignore) pixels (later proposals
overwrite earlier ones where they overlap); pixels outside every proposal are assigned
class 0 (soil, the majority terrain class). This is an EXPLICIT UPPER BOUND on any
zero-shot region labeler built on SAM's masks, and is reported as such.

H0 is reported honestly: it holds iff H1 is not rejected.
