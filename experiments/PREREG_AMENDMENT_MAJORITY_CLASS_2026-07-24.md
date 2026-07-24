# Protocol V5 Amendment — Training-Derived Constant-Majority Class

**Amendment date:** 2026-07-24  
**Timing:** after the 30 learned MSL runs and three H4 U-Net MER evaluations completed, while the
independent SAM upper-bound evaluation was running, and before any confirmatory majority-baseline
artifact or final H0--H5 analysis was produced.

## Fail-closed discovery

The first attempted deterministic baseline run stopped before prediction because its consistency
assertion found that the frozen text's assumed class 0 (soil) is not the training-split pixel
majority. The ignore-aware class counts over the fixed seed-1414 training split were:

| Class ID | Terrain | Valid training pixels |
|---:|---|---:|
| 0 | soil | 3,485,135,841 |
| 1 | bedrock | 4,660,855,168 |
| 2 | sand | 1,126,246,813 |
| 3 | big rock | 86,225,600 |

Thus bedrock (class 1) is the empirical majority. The official merged-0.6 `label_keys.json`
confirms the class-ID mapping. The failed attempts wrote no manifest, prediction, per-image table,
result-store row, leaderboard entry, hypothesis verdict, or test metric.

## Narrow correction

The parameter-free baseline remains a constant-majority predictor, but its predicted class is now
derived deterministically as `argmax` of valid training-split pixel counts. The derived class and
counts are recorded in the run manifest. For the frozen data and split this selects class 1.
H1 and H4 continue to compare against the same deterministic training-derived constant predictor;
only the erroneous hard-coded class assumption is removed.

SAM's separate region-oracle scoring rule is unchanged: uncovered proposal pixels continue to
default to class 0 exactly as frozen in V1. SAM is an explicit proposal-quality upper bound and is
not the H1/H4 constant-majority baseline.

## Unchanged evidence and inference

This correction does not change the dataset, split, learned models, weights, training runs, H4
subject selection, H4 U-Net predictions, hypotheses, test sets, metrics, bootstrap, confidence
level, significance level, Holm correction, or decision rules. Protocol V5 chains Protocol V4 and
all earlier artifacts byte-for-byte, and binds the corrected implementation before any majority
test prediction or final hypothesis decision is computed.
