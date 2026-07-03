"""Canonical-run selection (5.9) + family comparisons + Holm -> H0..H5 verdicts (MS3).

``decide(store, hyp_cfg, manifests_dir)`` resolves every comparison member to exactly one run_id
(filter seed/profile/status; multi-backbone ties broken by manifest ``best_val_miou``; multi-run
ties by newest ``timestamp_utc``; exact ties raise), joins the two runs' per_image tables on the
camera-qualified ``name``, runs the 5.6 paired bootstrap (5.7 for H4), applies Holm within each
family at alpha=0.10, and renders a decision in {support, reject, deferred} for every hypothesis.
H0 is reported honestly from H1. A missing/skipped member is ``deferred``, never a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.ai4mars import CLASSES
from .stats import counts_from_per_image, h4_rule, holm, paired_bootstrap

MANIFESTS_DIR = Path("experiments/manifests")

# Family-member comparison table (DEVPLAN section 10). 'a' is the delta's positive side.
COMPARISONS: dict[str, dict] = {
    "baseline_vs_unet": {
        "family": "A",
        "tail": "greater",
        "a": {"model": "unet", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "baseline", "stratum": "all"},
    },
    "baseline_vs_deeplabv3plus": {
        "family": "A",
        "tail": "greater",
        "a": {"model": "deeplabv3plus", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "baseline", "stratum": "all"},
    },
    "baseline_vs_segformer": {
        "family": "A",
        "tail": "greater",
        "a": {"model": "segformer", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "baseline", "stratum": "all"},
    },
    "unet_pretrained_vs_scratch": {
        "family": "B",
        "tail": "greater",
        "match_backbone": True,
        "a": {"model": "unet", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "unet", "variant": "scratch", "stratum": "all"},
    },
    "deeplabv3plus_pretrained_vs_scratch": {
        "family": "B",
        "tail": "greater",
        "match_backbone": True,
        "a": {"model": "deeplabv3plus", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "deeplabv3plus", "variant": "scratch", "stratum": "all"},
    },
    "segformer_pretrained_vs_scratch": {
        "family": "B",
        "tail": "greater",
        "match_backbone": True,
        "a": {"model": "segformer", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "segformer", "variant": "scratch", "stratum": "all"},
    },
    "segformer_vs_unet": {  # delta orientation: segformer - cnn (two-sided)
        "family": "C",
        "tail": "two_sided",
        "per_class": True,
        "a": {"model": "segformer", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "unet", "variant": "pretrained", "stratum": "all"},
    },
    "segformer_vs_deeplabv3plus": {
        "family": "C",
        "tail": "two_sided",
        "per_class": True,
        "a": {"model": "segformer", "variant": "pretrained", "stratum": "all"},
        "b": {"model": "deeplabv3plus", "variant": "pretrained", "stratum": "all"},
    },
    "dinov3_sat_vs_baseline": {
        "family": "E",
        "tail": "greater",
        "a": {"model": "dinov3_sat", "variant": "finetuned", "stratum": "all"},
        "b": {"model": "baseline", "stratum": "all"},
    },
    "sam_zeroshot_vs_baseline": {
        "family": "E",
        "tail": "greater",
        "a": {"model": "sam", "variant": "zeroshot", "stratum": "all"},
        "b": {"model": "baseline", "stratum": "all"},
    },
}


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


def resolve_run(
    store: pd.DataFrame,
    selector: dict,
    canon: dict,
    manifests_dir: Path = MANIFESTS_DIR,
    backbone: str | None = None,
) -> str:
    """One selector -> exactly one run_id per 5.9 (+ best_val_miou across backbones)."""
    df = store[(store["metric"] == "miou") & (store["scope"] == "ALL")]
    for col, val in selector.items():
        df = df[df[col] == val]
    if backbone is not None:
        df = df[df["backbone"] == backbone]
    flt = canon.get("filter", {})
    df = df[
        (df["seed"] == flt.get("seed", 1414))
        & (df["profile"] == flt.get("profile", "gpu_full"))
        & (df["status"] == flt.get("status", "ok"))
    ]
    if df.empty:
        raise Unresolvable(f"no canonical run for {selector} (backbone={backbone})")
    # multiple BACKBONES for one (model, variant, stratum): best manifest best_val_miou wins;
    # each backbone is scored from its NEWEST run, and null scores fall back to timestamps
    if df["backbone"].nunique() > 1:
        scores = {
            bb: _best_val(manifests_dir, _newest(manifests_dir, sub["run_id"].unique()))
            for bb, sub in df.groupby("backbone")
        }
        numeric = {bb: s for bb, s in scores.items() if s is not None}
        if numeric:
            best = max(numeric.values())
            winners = [bb for bb, s in numeric.items() if s == best]
        else:  # no val scores recorded anywhere: newest run's backbone wins
            stamps = {
                bb: _manifest(manifests_dir, _newest(manifests_dir, sub["run_id"].unique())).get(
                    "timestamp_utc", ""
                )
                for bb, sub in df.groupby("backbone")
            }
            newest = max(stamps.values())
            winners = [bb for bb, t in stamps.items() if t == newest]
        if len(winners) > 1:
            raise ValueError(f"ambiguous backbone for {selector}: {winners} tie on best_val_miou")
        df = df[df["backbone"] == winners[0]]
    # multiple runs for the SAME tuple: newest manifest timestamp; exact tie raises
    return _newest(manifests_dir, df["run_id"].unique())


def _per_image(manifests_dir: Path, run_id: str) -> pd.DataFrame:
    p = Path(manifests_dir) / run_id / "per_image.parquet"
    if not p.is_file():
        raise Unresolvable(f"per_image.parquet missing for {run_id}")
    return pd.read_parquet(p)


def _compare(
    store: pd.DataFrame, spec: dict, canon: dict, manifests_dir: Path, stats_cfg: dict
) -> dict:
    run_a = resolve_run(store, spec["a"], canon, manifests_dir)
    backbone = None
    if spec.get("match_backbone"):  # H2: scratch twin must share the pretrained side's backbone
        backbone = store[store["run_id"] == run_a].iloc[0]["backbone"]
    run_b = resolve_run(store, spec["b"], canon, manifests_dir, backbone=backbone)
    names_a, ia, ua = counts_from_per_image(_per_image(manifests_dir, run_a), "test_msl")
    names_b, ib, ub = counts_from_per_image(_per_image(manifests_dir, run_b), "test_msl")
    if names_a != names_b:
        raise ValueError(
            f"per-image name sets differ between {run_a} and {run_b} "
            f"({len(names_a)} vs {len(names_b)} names) — cannot pair"
        )
    res = paired_bootstrap(
        ia,
        ua,
        ib,
        ub,
        tail=spec["tail"],
        n_resamples=int(stats_cfg.get("n_resamples", 10000)),
        seed=int(stats_cfg.get("bootstrap_seed", 0)),
        per_class=bool(spec.get("per_class")),
    )
    return {"run_a": run_a, "run_b": run_b, **res}


def decide(
    store: pd.DataFrame,
    hyp_cfg: dict,
    manifests_dir: Path = MANIFESTS_DIR,
    git_sha: str = "UNKNOWN",
) -> dict:
    """Full verdict pass -> the verdicts.json dict (section 8 shape)."""
    alpha = float(hyp_cfg["alpha"])
    canon = hyp_cfg.get("canonical_run_selection", {})
    stats_cfg = hyp_cfg.get("stats", {})
    families_cfg = hyp_cfg["families"]

    members: dict[str, dict] = {}
    for comp_id, spec in COMPARISONS.items():
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
        raw_ps = [members[m]["result"]["ALL"]["p"] for m in ok]
        holm_ps = holm(raw_ps, alpha) if raw_ps else []
        out = []
        for m in fam_members:
            if m in ok:
                r = members[m]["result"]
                hp = holm_ps[ok.index(m)]
                sig = hp < alpha and (r["ALL"]["delta"] > 0 if fam != "C" else True)
                out.append(
                    {
                        "comparison": m,
                        "runs": [r["run_a"], r["run_b"]],
                        "delta": r["ALL"]["delta"],
                        "ci_low": r["ALL"]["ci_low"],
                        "ci_high": r["ALL"]["ci_high"],
                        "raw_p": r["ALL"]["p"],
                        "holm_p": hp,
                        "decision": ("significant" if sig else "not_significant"),
                        **(
                            {"direction": ("segformer" if r["ALL"]["delta"] > 0 else "cnn")}
                            if fam == "C"
                            else {}
                        ),
                        **(
                            {"per_class": {k: v for k, v in r.items() if k in CLASSES}}
                            if fam == "C"
                            else {}
                        ),
                    }
                )
            else:
                out.append(
                    {"comparison": m, "decision": "deferred", "reason": members[m]["reason"]}
                )
        families[fam] = {"members": out}

    def fam_decision(fam: str) -> str:
        mem = families[fam]["members"]
        if any(m["decision"] == "significant" for m in mem):
            return "support"
        if all(m["decision"] == "not_significant" for m in mem):
            return "reject"
        return "deferred"  # some members unresolved and none significant

    # ---- H4 (5.7): deterministic rule on the resolved subject + baseline-on-MER ----
    h4: dict = {"decision": "deferred"}
    try:
        cross = store[
            (store["metric"] == "miou")
            & (store["scope"] == "ALL")
            & (store["stratum"] == "cross_rover")
            & (store["status"] == "ok")
        ]
        flt = canon.get("filter", {})
        cross = cross[
            (cross["seed"] == flt.get("seed", 1414))
            & (cross["profile"] == flt.get("profile", "gpu_full"))
        ]
        subjects = cross[cross["model"] != "baseline"]
        base_mer = cross[cross["model"] == "baseline"]
        if subjects.empty or base_mer.empty:
            raise Unresolvable("missing cross_rover rows for subject and/or baseline")
        if subjects["run_id"].nunique() > 1:  # >1 candidate: highest manifest best_val_miou
            scores = {r: _best_val(manifests_dir, r) for r in subjects["run_id"].unique()}
            numeric = {r: s for r, s in scores.items() if s is not None}
            if numeric:
                best = max(numeric.values())
                winners = [r for r, s in numeric.items() if s == best]
                if len(winners) > 1:
                    raise ValueError(f"H4 subject tie on best_val_miou: {winners}")
                subj_run = winners[0]
            else:  # eval-only manifests may all lack val scores: newest wins, tie raises
                subj_run = _newest(manifests_dir, subjects["run_id"].unique())
            subjects = subjects[subjects["run_id"] == subj_run]
        subj_run = subjects.iloc[0]["run_id"]
        if base_mer["run_id"].nunique() > 1:  # baseline-on-MER must be canonical too, not iloc[0]
            base_run = _newest(manifests_dir, base_mer["run_id"].unique())
            base_mer = base_mer[base_mer["run_id"] == base_run]
        in_rover = store[
            (store["run_id"] == subj_run)
            & (store["stratum"] == "in_rover")
            & (store["metric"] == "miou")
            & (store["scope"] == "ALL")
        ]
        if in_rover.empty:
            raise Unresolvable(f"subject run {subj_run} lacks an in_rover miou row")
        _, mi, mu = counts_from_per_image(_per_image(manifests_dir, subj_run), "test_mer")
        h4 = h4_rule(
            float(in_rover.iloc[0]["value"]),
            mi,
            mu,
            float(base_mer.iloc[0]["value"]),
            drop_threshold=float(hyp_cfg["hypotheses"]["H4"]["drop_threshold"]),
            n_resamples=int(stats_cfg.get("n_resamples", 10000)),
            seed=int(stats_cfg.get("bootstrap_seed", 0)),
        )
        h4["subject_run"] = subj_run
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
                    "reject" if h1 == "support" else "support" if h1 == "reject" else "deferred"
                ),
                "evidence": "H0 holds iff H1 is not rejected (reported honestly).",
            },
            "H1": {"decision": h1, "evidence": "Family A vs baseline, Holm alpha=0.10."},
            "H2": {
                "decision": fam_decision("B"),
                "evidence": "Family B pretrained vs scratch, Holm alpha=0.10.",
            },
            "H3": {
                "decision": fam_decision("C"),
                "evidence": "Family C segformer-cnn two-sided + per-class strata.",
            },
            "H4": {
                "decision": h4["decision"],
                "evidence": {k: v for k, v in h4.items() if k != "decision"},
            },
            "H5": {
                "decision": h5_decision,
                "evidence": "Family E on gpu_full; deferred members lack ok runs (5.8).",
            },
        },
    }
    return verdicts


def leaderboard(store: pd.DataFrame, seed: int = 1414) -> pd.DataFrame:
    """model, backbone, variant, stratum, miou, ci_low, ci_high (status ok, primary seed).

    gpu_full rows outrank windows_cpu smoke rows for the same tuple, so a post-merge CPU smoke
    can never overwrite a GPU number; within a profile the last-appended (newest) row wins.
    """
    df = store[
        (store["metric"] == "miou")
        & (store["scope"] == "ALL")
        & (store["status"] == "ok")
        & (store["seed"] == seed)
    ]
    df = df.sort_values("profile", key=lambda s: (s == "gpu_full").astype(int), kind="stable")
    df = df.drop_duplicates(subset=["model", "backbone", "variant", "stratum"], keep="last")
    return df[["model", "backbone", "variant", "stratum", "value", "ci_low", "ci_high"]].rename(
        columns={"value": "miou"}
    )
