"""Paired bootstrap + H4 rule + Holm + McNemar (MS3; DEVPLAN 5.6 / 5.7 — the single source of
truth). Two independent implementations of this spec must produce identical numbers.

Bootstrap mechanics (5.6): unit = image, WITH replacement, ``numpy.random.default_rng(0)``
re-seeded at the start of every comparison; ONE index draw per replicate shared by both models
AND all per-class strata of that comparison; split-level metrics recomputed from SUMMED counts;
fixed class set S = {classes with full-split union > 0, over EITHER model}; a fixed-set class
with union == 0 in a resample contributes IoU = 0 (macro denominator stays |S|); p = plus-one
estimator; CI = percentile at ci_level. Protocol V2 generalizes the count arrays to
``(seed, class, image)`` and uses a two-level paired bootstrap: sample training seeds, then
sample images independently within every sampled seed. The a/b image draw remains paired.

H4 (5.7) is a deterministic threshold+CI rule, NO p-value. McNemar is descriptive-only.
"""

from __future__ import annotations

import numpy as np

from ..data.ai4mars import CLASSES
from ..utils.seed import BOOTSTRAP_SEED

N_RESAMPLES = 10_000
CI_LEVEL = 0.90


def counts_from_per_image(df, split: str, classes: list[str] = CLASSES):
    """Pivot a per_image table (7.4) to aligned arrays: (names, inter (C,N), union (C,N)).

    Keeps ``metric == 'iou'`` rows of ``split`` with a class scope; names are sorted so two
    tables of the same split align positionally after a set-equality check by the caller.
    """
    sub = df[(df["metric"] == "iou") & (df["split"] == split) & (df["scope"].isin(classes))]
    names = sorted(sub["name"].unique().tolist())
    n_index = {n: i for i, n in enumerate(names)}
    c_index = {c: i for i, c in enumerate(classes)}
    inter = np.zeros((len(classes), len(names)), dtype=np.int64)
    union = np.zeros((len(classes), len(names)), dtype=np.int64)
    for row in sub.itertuples(index=False):
        inter[c_index[row.scope], n_index[row.name]] = row.inter
        union[c_index[row.scope], n_index[row.name]] = row.union
    return names, inter, union


def _macro(inter_sums: np.ndarray, union_sums: np.ndarray, class_set: list[int]) -> float:
    """Macro mIoU over the FIXED set with the iou_zero rule (denominator stays |S|)."""
    vals = [(inter_sums[c] / union_sums[c] if union_sums[c] > 0 else 0.0) for c in class_set]
    return float(np.mean(vals))


def _p_value(deltas: np.ndarray, tail: str) -> float:
    b = len(deltas)
    le = int(np.count_nonzero(deltas <= 0))
    ge = int(np.count_nonzero(deltas >= 0))
    if tail == "greater":
        return (1 + le) / (b + 1)
    if tail == "two_sided":
        return min(1.0, 2 * min(1 + le, 1 + ge) / (b + 1))
    raise ValueError(f"unknown tail {tail!r}; expected 'greater' or 'two_sided'")


def _seeded_counts(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Normalize legacy ``(C,N)`` and V2 ``(S,C,N)`` count arrays to ``(S,C,N)``."""
    out = []
    for array in arrays:
        value = np.asarray(array)
        if value.ndim == 2:
            value = value[None, ...]
        if value.ndim != 3:
            raise ValueError("count arrays must have shape (C,N) or (S,C,N)")
        out.append(value)
    if len({value.shape for value in out}) != 1:
        raise ValueError("count arrays must share one (C,N) or (S,C,N) shape")
    return tuple(out)


def paired_bootstrap(
    inter_a: np.ndarray,
    union_a: np.ndarray,
    inter_b: np.ndarray,
    union_b: np.ndarray,
    *,
    tail: str = "greater",
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
    class_names: list[str] = CLASSES,
    per_class: bool = False,
) -> dict:
    """The V2 5.6 procedure on aligned ``(C,N)`` or ``(S,C,N)`` count arrays.

    Delta orientation is a − b (caller passes candidate as a / baseline as b; segformer as a
    for Family C). Returns {'ALL': {...}} plus one entry per fixed-set class when ``per_class``,
    each {'delta', 'ci_low', 'ci_high', 'p', 'n_images', 'n_seeds'}. For multiple training
    seeds, the observed statistic is the equal-weight mean of seed-level deltas. Each replicate
    samples seeds with replacement and then images with replacement within each sampled seed;
    all strata and both models share those draws. A one-seed input retains the legacy draw
    sequence exactly.
    """
    inter_a, union_a, inter_b, union_b = _seeded_counts(inter_a, union_a, inter_b, union_b)
    n_seed, n_cls, n_img = inter_a.shape
    if n_img == 0:
        raise ValueError("no images to bootstrap")
    # fixed set S over the FULL split, symmetric across the two models (fixed before resampling)
    full_union = union_a.sum(axis=(0, 2)) + union_b.sum(axis=(0, 2))
    class_set = [c for c in range(n_cls) if full_union[c] > 0]
    if not class_set:
        raise ValueError("empty fixed class set (no class has any union)")
    strata = ["ALL"] + ([class_names[c] for c in class_set] if per_class else [])

    def observed(stat_scope) -> float:
        seed_deltas = []
        for s in range(n_seed):
            ia, ua = inter_a[s].sum(axis=1), union_a[s].sum(axis=1)
            ib, ub = inter_b[s].sum(axis=1), union_b[s].sum(axis=1)
            if stat_scope == "ALL":
                seed_deltas.append(_macro(ia, ua, class_set) - _macro(ib, ub, class_set))
            else:
                c = class_names.index(stat_scope)
                va = ia[c] / ua[c] if ua[c] > 0 else 0.0
                vb = ib[c] / ub[c] if ub[c] > 0 else 0.0
                seed_deltas.append(float(va - vb))
        return float(np.mean(seed_deltas))

    rng = np.random.default_rng(seed)  # seed_reset_per_comparison
    deltas = {s: np.empty(n_resamples) for s in strata}
    for b in range(n_resamples):
        sampled_seeds = (
            np.array([0], dtype=np.int64) if n_seed == 1 else rng.integers(0, n_seed, size=n_seed)
        )
        replicate = {scope: [] for scope in strata}
        for sampled_seed in sampled_seeds:
            idx = rng.integers(0, n_img, size=n_img)
            ia = inter_a[sampled_seed, :, idx].sum(axis=0)
            ua = union_a[sampled_seed, :, idx].sum(axis=0)
            ib = inter_b[sampled_seed, :, idx].sum(axis=0)
            ub = union_b[sampled_seed, :, idx].sum(axis=0)
            replicate["ALL"].append(_macro(ia, ua, class_set) - _macro(ib, ub, class_set))
            if per_class:
                for c in class_set:
                    va = ia[c] / ua[c] if ua[c] > 0 else 0.0
                    vb = ib[c] / ub[c] if ub[c] > 0 else 0.0
                    replicate[class_names[c]].append(float(va - vb))
        for scope in strata:
            deltas[scope][b] = float(np.mean(replicate[scope]))
    lo_q, hi_q = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
    return {
        s: {
            "delta": observed(s),
            "ci_low": float(np.percentile(deltas[s], lo_q)),
            "ci_high": float(np.percentile(deltas[s], hi_q)),
            "p": _p_value(deltas[s], tail),
            "n_images": n_img,
            "n_seeds": n_seed,
        }
        for s in strata
    }


def bootstrap_miou_ci(
    inter: np.ndarray,
    union: np.ndarray,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> dict:
    """Single-model macro-mIoU CI, with the V2 seed/image hierarchy when ``S > 1``."""
    inter, union = _seeded_counts(inter, union)
    n_seed, n_cls, n_img = inter.shape
    full_union = union.sum(axis=(0, 2))
    class_set = [c for c in range(n_cls) if full_union[c] > 0]
    if not class_set or n_img == 0:
        raise ValueError("cannot bootstrap an empty split/class set")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        sampled_seeds = (
            np.array([0], dtype=np.int64) if n_seed == 1 else rng.integers(0, n_seed, size=n_seed)
        )
        seed_vals = []
        for sampled_seed in sampled_seeds:
            idx = rng.integers(0, n_img, size=n_img)
            seed_vals.append(
                _macro(
                    inter[sampled_seed, :, idx].sum(axis=0),
                    union[sampled_seed, :, idx].sum(axis=0),
                    class_set,
                )
            )
        vals[b] = float(np.mean(seed_vals))
    lo_q, hi_q = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
    observed = [
        _macro(inter[s].sum(axis=1), union[s].sum(axis=1), class_set) for s in range(n_seed)
    ]
    return {
        "miou": float(np.mean(observed)),
        "ci_low": float(np.percentile(vals, lo_q)),
        "ci_high": float(np.percentile(vals, hi_q)),
        "n_images": n_img,
        "n_seeds": n_seed,
    }


def h4_rule(
    miou_in_rover: float,
    mer_inter: np.ndarray,
    mer_union: np.ndarray,
    baseline_on_mer_miou: float,
    *,
    drop_threshold: float = 0.15,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> dict:
    """DEVPLAN 5.7 — deterministic, no p-value.

    SUPPORT iff drop = mIoU_in_rover − mIoU_cross_rover < drop_threshold (point estimates)
    AND cross_rover_ci_low > baseline_on_MER_miou (5th pct of the MER-image bootstrap).
    """
    boot = bootstrap_miou_ci(
        mer_inter, mer_union, n_resamples=n_resamples, seed=seed, ci_level=ci_level
    )
    in_rover = float(np.mean(np.asarray(miou_in_rover, dtype=float)))
    baseline_mer = float(np.mean(np.asarray(baseline_on_mer_miou, dtype=float)))
    drop = in_rover - boot["miou"]
    support = (drop < drop_threshold) and (boot["ci_low"] > baseline_mer)
    return {
        "miou_in_rover": in_rover,
        "miou_cross_rover": boot["miou"],
        "drop": drop,
        "drop_threshold": drop_threshold,
        "cross_rover_ci_low": boot["ci_low"],
        "cross_rover_ci_high": boot["ci_high"],
        "baseline_on_mer_miou": baseline_mer,
        "n_seeds": boot["n_seeds"],
        "decision": "support" if support else "reject",
    }


def holm(p_values: list[float], alpha: float = 0.10) -> list[float]:
    """Holm step-down adjusted p-values (same order as input); compare adjusted < alpha."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def mcnemar(b: int, c: int) -> dict:
    """Pixel-level McNemar with Edwards continuity correction — DESCRIPTIVE ONLY (5.4).

    ``b`` = #pixels A correct & B wrong; ``c`` = #pixels A wrong & B correct (valid pixels only).
    """
    from scipy.stats import chi2 as chi2_dist

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": int(b), "c": int(c), "chi2": float(chi2), "p": float(chi2_dist.sf(chi2, 1))}


def discordant_counts(pred_a: np.ndarray, pred_b: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """Per-image (b, c) discordant pixel counts over valid pixels (gt != 255)."""
    valid = gt != 255
    a_ok = (pred_a == gt) & valid
    b_ok = (pred_b == gt) & valid
    return int(np.count_nonzero(a_ok & ~b_ok)), int(np.count_nonzero(~a_ok & b_ok))
