"""Canonical-run selection (5.9) + family comparisons + Holm -> H0..H5 verdicts (MS3).

``decide(store, hyp_cfg, manifests_dir)`` resolves each YAML comparison selector to a complete
set of primary-seed runs (profile/status filter; backbone chosen by mean validation mIoU;
within-seed ties by newest ``timestamp_utc``), aligns per-image sufficient statistics, runs the
V2 paired seed/image bootstrap (threshold+CI for H4), and applies Holm within each family.
Family C's correction family contains every overall and fixed-set per-class test. H0 uses the
standard ``fail_to_reject`` wording when H1 is not significant. Missing seed sets defer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.ai4mars import CLASSES
from .stats import counts_from_per_image, h4_rule, holm, paired_bootstrap

MANIFESTS_DIR = Path("experiments/manifests")


class Unresolvable(Exception):
    """A comparison side cannot be resolved to a canonical run (missing/skipped) -> deferred."""


def _manifest(manifests_dir: Path, run_id: str) -> dict:
    p = Path(manifests_dir) / run_id / "manifest.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _best_val(manifests_dir: Path, run_id: str) -> float | None:
    """Numeric best_val_miou or None (eval-only manifests may record null — never compare None)."""
    v = _manifest(manifests_dir, run_id).get("best_val_miou")
    return float(v) if isinstance(v, (int, float)) else None


def _parameter_counts(manifests_dir: Path, run_ids: list[str]) -> list[dict]:
    """Parameter counts aligned to a comparison's primary seeds (null fails visibly in output)."""
    return [
        {
            "run_id": run_id,
            "total_parameters": _manifest(manifests_dir, run_id).get("total_parameters"),
            "trainable_parameters": _manifest(manifests_dir, run_id).get("trainable_parameters"),
        }
        for run_id in run_ids
    ]


def _newest(manifests_dir: Path, run_ids) -> str:
    """Newest run by manifest timestamp; exact tie raises (5.9 — do not guess)."""
    run_ids = sorted(run_ids)
    if len(run_ids) == 1:
        return run_ids[0]
    stamps = {r: _manifest(manifests_dir, r).get("timestamp_utc", "") for r in run_ids}
    newest = max(stamps.values())
    winners = [r for r, t in stamps.items() if t == newest]
    if len(winners) > 1:
        raise ValueError(f"canonical-run tie on timestamp_utc: {winners}")
    return winners[0]


def _primary_seeds(canon: dict) -> list[int]:
    flt = canon.get("filter", {})
    seeds = flt.get("seeds")
    if seeds is None:
        seeds = [flt.get("seed", 1414)]
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("canonical_run_selection.filter.seeds must be non-empty and unique")
    return seeds


def resolve_runs(
    store: pd.DataFrame,
    selector: dict,
    canon: dict,
    manifests_dir: Path = MANIFESTS_DIR,
    backbone: str | None = None,
) -> dict[int, str]:
    """Resolve one selector to one canonical run for every required primary seed."""
    df = store[(store["metric"] == "miou") & (store["scope"] == "ALL")]
    for col, val in selector.items():
        df = df[df[col] == val]
    flt = canon.get("filter", {})
    df = df[
        (df["profile"] == flt.get("profile", "gpu_full"))
        & (df["status"] == flt.get("status", "ok"))
    ]
    seeds = _primary_seeds(canon)
    deterministic = selector.get("model") in set(canon.get("deterministic_models", []))
    source_seeds = (
        [int(canon.get("deterministic_artifact_seed", seeds[0]))] if deterministic else seeds
    )
    df = df[df["seed"].isin(source_seeds)]
    if backbone is not None:
        df = df[df["backbone"] == backbone]
    if df.empty:
        raise Unresolvable(f"no primary runs for {selector} (backbone={backbone})")

    candidates: dict[str, dict[int, str]] = {}
    for bb, by_backbone in df.groupby("backbone"):
        per_seed = {}
        for seed in source_seeds:
            rows = by_backbone[by_backbone["seed"] == seed]
            if rows.empty:
                break
            per_seed[seed] = _newest(manifests_dir, rows["run_id"].unique())
        if len(per_seed) == len(source_seeds):
            candidates[str(bb)] = (
                {seed: per_seed[source_seeds[0]] for seed in seeds} if deterministic else per_seed
            )
    if not candidates:
        raise Unresolvable(
            f"incomplete {'deterministic artifact' if deterministic else 'primary seed set'} "
            f"{source_seeds} for {selector} (backbone={backbone})"
        )
    if len(candidates) > 1:
        scores = {}
        for bb, per_seed in candidates.items():
            vals = [_best_val(manifests_dir, run_id) for run_id in per_seed.values()]
            if any(value is None for value in vals):
                raise Unresolvable(
                    f"best_val_miou missing while selecting backbone {bb} for {selector}"
                )
            scores[bb] = sum(vals) / len(vals)
        best = max(scores.values())
        winners = [bb for bb, score in scores.items() if score == best]
        if len(winners) != 1:
            raise ValueError(f"ambiguous backbone for {selector}: {winners} tie on mean best_val")
        return candidates[winners[0]]
    return next(iter(candidates.values()))


def resolve_run(
    store: pd.DataFrame,
    selector: dict,
    canon: dict,
    manifests_dir: Path = MANIFESTS_DIR,
    backbone: str | None = None,
) -> str:
    """Backward-compatible single-seed resolver; V2 analysis uses :func:`resolve_runs`."""
    runs = resolve_runs(store, selector, canon, manifests_dir, backbone)
    if len(runs) != 1:
        raise ValueError("resolve_run requires a one-seed canonical filter; use resolve_runs")
    return next(iter(runs.values()))


def _per_image(manifests_dir: Path, run_id: str) -> pd.DataFrame:
    p = Path(manifests_dir) / run_id / "per_image.parquet"
    if not p.is_file():
        raise Unresolvable(f"per_image.parquet missing for {run_id}")
    return pd.read_parquet(p)


def _compare(
    store: pd.DataFrame, spec: dict, canon: dict, manifests_dir: Path, stats_cfg: dict
) -> dict:
    runs_a = resolve_runs(store, spec["a"], canon, manifests_dir)
    backbone = None
    if spec.get("match_backbone"):  # H2: scratch twin must share the pretrained side's backbone
        backbone = store[store["run_id"] == next(iter(runs_a.values()))].iloc[0]["backbone"]
    runs_b = resolve_runs(store, spec["b"], canon, manifests_dir, backbone=backbone)
    if list(runs_a) != list(runs_b):
        raise ValueError("comparison sides resolved different primary seed sets")
    arrays = []
    expected_names = None
    for seed in runs_a:
        run_a, run_b = runs_a[seed], runs_b[seed]
        names_a, ia, ua = counts_from_per_image(_per_image(manifests_dir, run_a), "test_msl")
        names_b, ib, ub = counts_from_per_image(_per_image(manifests_dir, run_b), "test_msl")
        if names_a != names_b or (expected_names is not None and names_a != expected_names):
            raise ValueError(f"per-image names are not aligned for seed {seed}: {run_a} vs {run_b}")
        expected_names = names_a
        arrays.append((ia, ua, ib, ub))
    ia, ua, ib, ub = (np.stack(values) for values in zip(*arrays, strict=True))
    res = paired_bootstrap(
        ia,
        ua,
        ib,
        ub,
        tail=spec["tail"],
        n_resamples=int(stats_cfg.get("n_resamples", 10000)),
        seed=int(stats_cfg.get("bootstrap_seed", 0)),
        ci_level=float(stats_cfg.get("ci_level", 0.90)),
        per_class=bool(spec.get("per_class")),
    )
    return {
        "run_a": [runs_a[seed] for seed in runs_a],
        "run_b": [runs_b[seed] for seed in runs_b],
        "seeds": list(runs_a),
        **res,
    }


def _validate_comparison_config(hyp_cfg: dict) -> dict:
    """Fail closed when YAML family membership and executable selectors disagree."""
    if hyp_cfg.get("correction") != "holm":
        raise ValueError("only the sealed Holm correction is implemented")
    comparisons = hyp_cfg.get("comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        raise ValueError("hypotheses config must contain non-empty comparisons selectors")
    expected = {member for family, cfg in hyp_cfg["families"].items() for member in cfg["members"]}
    if set(comparisons) != expected:
        raise ValueError(
            f"comparison selector keys differ from family members: "
            f"missing={sorted(expected - set(comparisons))}, "
            f"extra={sorted(set(comparisons) - expected)}"
        )
    for comp_id, spec in comparisons.items():
        family = spec.get("family")
        if comp_id not in hyp_cfg["families"].get(family, {}).get("members", []):
            raise ValueError(f"comparison {comp_id} declares inconsistent family {family!r}")
        if family == "D":
            candidates = spec.get("subject_candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("Family D must define non-empty subject_candidates")
            if not isinstance(spec.get("reference"), dict):
                raise ValueError("Family D must define a reference selector")
            continue
        if spec.get("tail") not in {"greater", "two_sided"}:
            raise ValueError(f"comparison {comp_id} has invalid tail")
        if not all(isinstance(spec.get(side), dict) for side in ("a", "b")):
            raise ValueError(f"comparison {comp_id} must define selector mappings a and b")
    return comparisons


def decide(
    store: pd.DataFrame,
    hyp_cfg: dict,
    manifests_dir: Path = MANIFESTS_DIR,
    git_sha: str = "UNKNOWN",
) -> dict:
    """Full verdict pass -> the verdicts.json dict (section 8 shape)."""
    alpha = float(hyp_cfg["alpha"])
    canon = hyp_cfg.get("canonical_run_selection", {})
    stats_cfg = {**hyp_cfg.get("stats", {}), "ci_level": float(hyp_cfg["ci_level"])}
    families_cfg = hyp_cfg["families"]
    comparisons = _validate_comparison_config(hyp_cfg)

    members: dict[str, dict] = {}
    for comp_id, spec in comparisons.items():
        if spec["family"] == "D":
            continue
        try:
            members[comp_id] = {
                "status": "ok",
                "result": _compare(store, spec, canon, manifests_dir, stats_cfg),
            }
        except Unresolvable as e:
            members[comp_id] = {"status": "deferred", "reason": str(e)}

    families: dict[str, dict] = {}
    for fam, fcfg in families_cfg.items():
        if fam == "D":
            continue  # H4 is deterministic (5.7), no Holm — handled below
        fam_members = fcfg["members"]
        ok = [m for m in fam_members if members.get(m, {}).get("status") == "ok"]
        test_keys = []
        for member in ok:
            result = members[member]["result"]
            scopes = ["ALL"]
            if fam == "C":
                scopes.extend(class_name for class_name in CLASSES if class_name in result)
            test_keys.extend((member, scope) for scope in scopes)
        raw_ps = [members[m]["result"][scope]["p"] for m, scope in test_keys]
        holm_ps = holm(raw_ps, alpha) if raw_ps else []
        adjusted = {key: holm_ps[i] for i, key in enumerate(test_keys)}
        out = []
        for m in fam_members:
            if m in ok:
                r = members[m]["result"]
                hp = adjusted[(m, "ALL")]
                overall_sig = hp < alpha and (r["ALL"]["delta"] > 0 if fam != "C" else True)
                per_class = {}
                if fam == "C":
                    for class_name in CLASSES:
                        if class_name not in r:
                            continue
                        class_result = r[class_name]
                        class_hp = adjusted[(m, class_name)]
                        class_sig = class_hp < alpha
                        delta = class_result["delta"]
                        per_class[class_name] = {
                            **class_result,
                            "raw_p": class_result["p"],
                            "holm_p": class_hp,
                            "decision": "significant" if class_sig else "not_significant",
                            "direction": (
                                "segformer" if delta > 0 else "cnn" if delta < 0 else "tie"
                            ),
                        }
                    any_sig = overall_sig or any(
                        result["decision"] == "significant" for result in per_class.values()
                    )
                else:
                    any_sig = overall_sig
                out.append(
                    {
                        "comparison": m,
                        "runs": [r["run_a"], r["run_b"]],
                        "seeds": r["seeds"],
                        "parameter_counts": {
                            "a": _parameter_counts(manifests_dir, r["run_a"]),
                            "b": _parameter_counts(manifests_dir, r["run_b"]),
                        },
                        "delta": r["ALL"]["delta"],
                        "ci_low": r["ALL"]["ci_low"],
                        "ci_high": r["ALL"]["ci_high"],
                        "raw_p": r["ALL"]["p"],
                        "holm_p": hp,
                        "decision": ("significant" if any_sig else "not_significant"),
                        **(
                            {
                                "overall_decision": (
                                    "significant" if overall_sig else "not_significant"
                                ),
                                "direction": (
                                    "segformer"
                                    if r["ALL"]["delta"] > 0
                                    else "cnn" if r["ALL"]["delta"] < 0 else "tie"
                                ),
                            }
                            if fam == "C"
                            else {}
                        ),
                        **({"per_class": per_class} if fam == "C" else {}),
                    }
                )
            else:
                out.append(
                    {"comparison": m, "decision": "deferred", "reason": members[m]["reason"]}
                )
        families[fam] = {
            "members": out,
            "holm_test_count": len(test_keys),
            "holm_scopes": [f"{member}:{scope}" for member, scope in test_keys],
        }

    def fam_decision(fam: str) -> str:
        mem = families[fam]["members"]
        if any(m["decision"] == "significant" for m in mem):
            return "support"
        if all(m["decision"] == "not_significant" for m in mem):
            return "reject"
        return "deferred"  # some members unresolved and none significant

    # ---- H4 (5.7): best validation-selected subject vs majority-on-MER ----
    h4: dict = {"decision": "deferred"}
    try:
        h4_spec = comparisons["best_in_rover_vs_cross_rover"]
        candidates = []
        deferred_reasons = []
        for selector in h4_spec["subject_candidates"]:
            try:
                runs = resolve_runs(store, selector, canon, manifests_dir)
                vals = [_best_val(manifests_dir, run_id) for run_id in runs.values()]
                if any(value is None for value in vals):
                    raise Unresolvable(f"best_val_miou missing for H4 candidate {selector}")
                candidates.append((float(np.mean(vals)), selector, runs))
            except Unresolvable as exc:
                deferred_reasons.append(str(exc))
        if not candidates:
            raise Unresolvable("no complete H4 subject candidate: " + "; ".join(deferred_reasons))
        best_score = max(score for score, _, _ in candidates)
        winners = [(selector, runs) for score, selector, runs in candidates if score == best_score]
        if len(winners) != 1:
            raise ValueError(f"H4 subject candidates tie on mean best_val_miou: {winners}")
        subject_selector, subject_runs = winners[0]
        reference_runs = resolve_runs(store, h4_spec["reference"], canon, manifests_dir)
        if list(subject_runs) != list(reference_runs):
            raise ValueError("H4 subject and majority reference resolved different seed sets")

        in_rover_values, reference_values, mer_counts = [], [], []
        expected_names = None
        for seed, subject_run in subject_runs.items():
            in_rover = store[
                (store["run_id"] == subject_run)
                & (store["stratum"] == "in_rover")
                & (store["metric"] == "miou")
                & (store["scope"] == "ALL")
            ]
            if len(in_rover) != 1:
                raise Unresolvable(f"subject run {subject_run} lacks one in_rover mIoU row")
            reference = store[
                (store["run_id"] == reference_runs[seed])
                & (store["stratum"] == "cross_rover")
                & (store["metric"] == "miou")
                & (store["scope"] == "ALL")
            ]
            if len(reference) != 1:
                raise Unresolvable(
                    f"majority reference run {reference_runs[seed]} lacks one MER mIoU row"
                )
            names, mi, mu = counts_from_per_image(
                _per_image(manifests_dir, subject_run), "test_mer"
            )
            if expected_names is not None and names != expected_names:
                raise ValueError("H4 MER image names differ across primary seeds")
            expected_names = names
            in_rover_values.append(float(in_rover.iloc[0]["value"]))
            reference_values.append(float(reference.iloc[0]["value"]))
            mer_counts.append((mi, mu))
        mi, mu = (np.stack(values) for values in zip(*mer_counts, strict=True))
        h4 = h4_rule(
            in_rover_values,
            mi,
            mu,
            reference_values,
            drop_threshold=float(hyp_cfg["hypotheses"]["H4"]["drop_threshold"]),
            n_resamples=int(stats_cfg.get("n_resamples", 10000)),
            seed=int(stats_cfg.get("bootstrap_seed", 0)),
            ci_level=float(stats_cfg.get("ci_level", 0.90)),
        )
        h4.update(
            {
                "subject_selector": subject_selector,
                "subject_runs": list(subject_runs.values()),
                "reference_runs": list(reference_runs.values()),
                "seeds": list(subject_runs),
                "mean_best_val_miou": best_score,
            }
        )
    except Unresolvable as e:
        h4 = {"decision": "deferred", "reason": str(e)}
    families["D"] = {"members": [{"comparison": "best_in_rover_vs_cross_rover", **h4}]}

    # ---- H5 (5.8): Holm over ok members only; deferred if zero ok members ----
    e_members = families["E"]["members"]
    e_ok = [m for m in e_members if m["decision"] in ("significant", "not_significant")]
    if not e_ok:
        h5_decision = "deferred"
    elif any(m["decision"] == "significant" for m in e_ok):
        h5_decision = "support"
    else:
        h5_decision = "reject"

    h1 = fam_decision("A")
    verdicts = {
        "alpha": alpha,
        "correction": hyp_cfg["correction"],
        "generated_git_sha": git_sha,
        "families": families,
        "hypotheses": {
            "H0": {
                "decision": (
                    "reject"
                    if h1 == "support"
                    else "fail_to_reject" if h1 == "reject" else "deferred"
                ),
                "evidence": (
                    "H0 is rejected iff Family A rejects at least one majority comparison; "
                    "otherwise non-significance is fail-to-reject, never support for H0."
                ),
            },
            "H1": {
                "decision": h1,
                "evidence": "Family A deep models vs constant-majority, Holm alpha=0.10.",
            },
            "H2": {
                "decision": fam_decision("B"),
                "evidence": (
                    "Family B pretrained vs scratch, Holm alpha=0.10; support means transfer "
                    "helped at least one tested architecture, not that transfer helps generally."
                ),
            },
            "H3": {
                "decision": fam_decision("C"),
                "evidence": (
                    "Family C is one Holm family over both overall tests and every fixed-set "
                    "per-class test for both SegFormer-CNN comparisons."
                ),
            },
            "H4": {
                "decision": h4["decision"],
                "evidence": {k: v for k, v in h4.items() if k != "decision"},
            },
            "H5": {
                "decision": h5_decision,
                "evidence": (
                    "Family E compares DINOv3-SAT and the SAM region-oracle upper bound with "
                    "the learned TinyUNet reference; incomplete primary seed sets defer."
                ),
            },
        },
    }
    return verdicts


def leaderboard(
    store: pd.DataFrame,
    seeds: tuple[int, ...] = (1414, 1415, 1416),
    profile: str = "gpu_full",
) -> pd.DataFrame:
    """Per-seed primary-profile mIoU rows; this is not a multi-seed inferential summary.

    Seed is explicit and per-seed CI endpoints are never averaged. Within one
    model/backbone/variant/stratum/seed tuple, the last-appended row wins.
    """
    df = store[
        (store["metric"] == "miou")
        & (store["scope"] == "ALL")
        & (store["status"] == "ok")
        & (store["profile"] == profile)
        & (store["seed"].isin(seeds))
    ]
    identity = ["model", "backbone", "variant", "stratum", "profile", "seed"]
    df = df.drop_duplicates(subset=identity, keep="last").sort_values(identity, kind="stable")
    return df[[*identity, "value", "ci_low", "ci_high"]].rename(columns={"value": "miou"})
