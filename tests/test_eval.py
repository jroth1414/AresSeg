"""Eval-stack unit tests (MS3): metrics counts, 5.6 paired bootstrap, Holm, H4 rule, verdict
resolution, config concreteness, prereg seal. Offline, no network, small n_resamples for speed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from marsseg.data.ai4mars import CLASSES
from marsseg.eval import aggregate, metrics, prereg, stats, verdict
from marsseg.utils.results import RESULT_COLUMNS

REPO = Path(__file__).resolve().parents[1]
B = 200  # small bootstrap for tests; production default stays 10000 (asserted below)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_per_image_counts_hand_case():
    gt = np.array([[0, 0, 1], [1, 255, 2]])
    pred = np.array([[0, 1, 1], [1, 0, 3]])
    c = metrics.per_image_counts(pred, gt, num_classes=4)
    assert c["n_valid"] == 5  # the 255 pixel is excluded everywhere
    assert c["correct"] == 3
    # class 0: inter {(0,0)}; union {(0,0),(0,1)} — the ignore-pixel pred 0 must NOT count
    assert c["inter"][0] == 1 and c["union"][0] == 2
    assert c["inter"][1] == 2 and c["union"][1] == 3
    assert c["inter"][2] == 0 and c["union"][2] == 1
    assert c["inter"][3] == 0 and c["union"][3] == 1


def test_macro_fixed_set_and_iou_zero():
    inter = np.array([5, 0, 0, 0])
    union = np.array([10, 20, 0, 0])
    s = metrics.fixed_class_set(union)
    assert s == [0, 1]  # union-0 classes are excluded from S, not counted as 0
    assert metrics.macro_miou_from_counts(inter, union, s) == pytest.approx((0.5 + 0.0) / 2)
    assert np.isnan(metrics.iou_from_counts(0, 0))


def test_boundary_f1_perfect_and_disjoint():
    gt = np.zeros((16, 16), dtype=int)
    gt[4:12, 4:12] = 1
    assert metrics.boundary_f1(gt, gt) == pytest.approx(1.0)
    pred = np.zeros_like(gt)  # misses class 1 entirely
    assert metrics.boundary_f1(pred, gt) < 1.0


# ---------------------------------------------------------------------------
# stats: paired bootstrap (5.6), Holm, McNemar, H4 (5.7)
# ---------------------------------------------------------------------------


def _counts(n_img, iou_per_class):
    """(C,N) constant-count arrays with the given per-class IoU (union fixed at 100)."""
    c = len(iou_per_class)
    union = np.full((c, n_img), 100, dtype=np.int64)
    inter = np.array([[int(100 * v)] * n_img for v in iou_per_class], dtype=np.int64)
    return inter, union


def test_paired_bootstrap_direction_and_p_bounds():
    ia, ua = _counts(6, [1.0, 1.0])
    ib, ub = _counts(6, [0.5, 0.5])
    r = stats.paired_bootstrap(
        ia, ua, ib, ub, tail="greater", n_resamples=B, class_names=["a", "b"]
    )
    assert r["ALL"]["delta"] == pytest.approx(0.5)
    assert r["ALL"]["p"] == pytest.approx(1 / (B + 1))  # every replicate delta > 0
    assert r["ALL"]["ci_low"] == pytest.approx(0.5) and r["ALL"]["ci_high"] == pytest.approx(0.5)
    # identical models: delta 0 everywhere -> one-sided p = 1, two-sided clipped to 1
    r0 = stats.paired_bootstrap(
        ia, ua, ia, ua, tail="greater", n_resamples=B, class_names=["a", "b"]
    )
    assert r0["ALL"]["delta"] == 0 and r0["ALL"]["p"] == pytest.approx(1.0)
    r2 = stats.paired_bootstrap(
        ia, ua, ia, ua, tail="two_sided", n_resamples=B, class_names=["a", "b"]
    )
    assert r2["ALL"]["p"] == pytest.approx(1.0)


def test_paired_bootstrap_deterministic_and_per_class():
    rng = np.random.default_rng(7)
    n = 12
    ua = rng.integers(50, 150, size=(4, n))
    ia = (ua * rng.uniform(0.3, 0.9, size=(4, n))).astype(np.int64)
    ub = rng.integers(50, 150, size=(4, n))
    ib = (ub * rng.uniform(0.3, 0.9, size=(4, n))).astype(np.int64)
    r1 = stats.paired_bootstrap(ia, ua, ib, ub, tail="two_sided", n_resamples=B, per_class=True)
    r2 = stats.paired_bootstrap(ia, ua, ib, ub, tail="two_sided", n_resamples=B, per_class=True)
    assert r1 == r2  # default_rng(seed) reset per comparison => bit-identical
    assert set(r1) == {"ALL", *CLASSES}
    for v in r1.values():
        assert v["ci_low"] <= v["delta"] <= v["ci_high"] or v["ci_low"] <= v["ci_high"]


def test_paired_bootstrap_default_constants():
    assert stats.N_RESAMPLES == 10_000 and stats.CI_LEVEL == 0.90
    from marsseg.utils.seed import BOOTSTRAP_SEED

    assert BOOTSTRAP_SEED == 0


def test_holm_hand_case():
    #  m=3: sorted p = [.01,.03,.04] -> adj [.03,.06,.06]; order preserved on return
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # distinct adjusted values pin the ORDER MAPPING back to input positions
    assert stats.holm([0.01, 0.4, 0.03]) == pytest.approx([0.03, 0.4, 0.06])
    assert stats.holm([0.5]) == [0.5]
    assert stats.holm([]) == []


def test_macro_iou_zero_in_resample_rule():
    # a fixed-set class with union==0 in a resample contributes IoU=0; denominator stays |S|
    assert stats._macro(np.array([5, 0]), np.array([10, 0]), [0, 1]) == pytest.approx(0.25)
    assert stats._macro(np.array([5, 7]), np.array([10, 0]), [0, 1]) == pytest.approx(0.25)


def test_mcnemar_hand_case():
    r = stats.mcnemar(30, 10)
    assert r["chi2"] == pytest.approx((abs(30 - 10) - 1) ** 2 / 40)
    assert 0 < r["p"] < 0.01
    assert stats.mcnemar(0, 0)["p"] == 1.0


def test_h4_rule_branches():
    mi, mu = _counts(10, [0.5, 0.5, 0.5, 0.5])  # constant 0.5 -> degenerate CI at 0.5
    kw = dict(n_resamples=B)
    r = stats.h4_rule(0.60, mi, mu, baseline_on_mer_miou=0.40, **kw)
    assert r["decision"] == "support" and r["drop"] == pytest.approx(0.10)
    assert stats.h4_rule(0.70, mi, mu, 0.40, **kw)["decision"] == "reject"  # drop 0.20 >= 0.15
    assert stats.h4_rule(0.60, mi, mu, 0.55, **kw)["decision"] == "reject"  # ci_low !> baseline


# ---------------------------------------------------------------------------
# aggregate: 7.4 table + store rows
# ---------------------------------------------------------------------------


def test_aggregate_roundtrip():
    counts = {
        "inter": np.array([50, 0, 25, 0]),
        "union": np.array([100, 0, 50, 0]),
        "correct": 75,
        "n_valid": 150,
    }
    rows = aggregate.image_rows("r1", "msl_ncam_x", "test_msl", counts, bf1=0.8)
    df = aggregate.per_image_frame(rows)
    assert set(df["metric"]) == {"iou", "pixel_acc", "boundary_f1"}
    assert len(df) == 6 and df["n_valid"].eq(150).all()
    store = aggregate.store_rows(
        df,
        split="test_msl",
        stratum="all",
        run_id="r1",
        model="baseline",
        backbone="none",
        variant="scratch",
        profile="windows_cpu",
        seed=1414,
        git_sha="g",
        config_hash="c",
    )
    by = {(r["metric"], r["scope"], r["stratum"]): r for r in store}
    assert by[("miou", "ALL", "all")]["value"] == pytest.approx(0.5)  # S={0,2}: (0.5+0.5)/2
    assert by[("n", "ALL", "all")]["value"] == 1
    assert by[("pixel_acc", "ALL", "all")]["value"] == pytest.approx(0.5)
    assert np.isnan(by[("iou", "bedrock", "per_class")]["value"])  # union-0 class -> NaN
    # H4 eval runs: per-class rows must inherit in_rover/cross_rover so the two splits of one
    # run_id never collide on DEDUP_KEYS (7.1) and merge_results cannot silently drop rows
    cross = aggregate.store_rows(
        df,
        split="test_msl",
        stratum="cross_rover",
        run_id="r1",
        model="unet",
        backbone="resnet34",
        variant="pretrained",
        profile="gpu_full",
        seed=1414,
        git_sha="g",
        config_hash="c",
    )
    assert all(r["stratum"] == "cross_rover" for r in cross if r["metric"] == "iou")


def test_h4_run_store_rows_never_collide_on_dedup_keys():
    from marsseg.utils.results import DEDUP_KEYS

    counts = {
        "inter": np.array([50, 10, 25, 5]),
        "union": np.array([100, 20, 50, 10]),
        "correct": 75,
        "n_valid": 150,
    }
    rows = aggregate.image_rows("r1", "msl_ncam_x", "test_msl", counts, bf1=0.8)
    rows += aggregate.image_rows("r1", "mer_test_y", "test_mer", counts, bf1=0.7)
    df = aggregate.per_image_frame(rows)
    kw = dict(
        run_id="r1",
        model="unet",
        backbone="resnet34",
        variant="pretrained",
        profile="gpu_full",
        seed=1414,
        git_sha="g",
        config_hash="c",
    )
    both = aggregate.store_rows(df, split="test_msl", stratum="in_rover", **kw)
    both += aggregate.store_rows(df, split="test_mer", stratum="cross_rover", **kw)
    assert int(pd.DataFrame(both).duplicated(subset=DEDUP_KEYS).sum()) == 0


# ---------------------------------------------------------------------------
# verdict: 5.9 canonical-run selection + empty-store decisions
# ---------------------------------------------------------------------------


def _store_row(
    run_id,
    model="unet",
    backbone="resnet34",
    variant="pretrained",
    stratum="all",
    value=0.5,
    status="ok",
    profile="gpu_full",
    seed=1414,
):
    return {
        "run_id": run_id,
        "model": model,
        "backbone": backbone,
        "variant": variant,
        "scope": "ALL",
        "stratum": stratum,
        "metric": "miou",
        "value": value,
        "ci_low": None,
        "ci_high": None,
        "status": status,
        "profile": profile,
        "seed": seed,
        "git_sha": "g",
        "config_hash": run_id[-2:],
    }


def _manifest(tmp, run_id, ts, best_val=None):
    d = tmp / run_id
    d.mkdir(parents=True, exist_ok=True)
    m = {"run_id": run_id, "timestamp_utc": ts}
    if best_val is not None:
        m["best_val_miou"] = best_val
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_resolve_run_filters_and_tiebreak(tmp_path):
    canon = {"filter": {"seed": 1414, "profile": "gpu_full", "status": "ok"}}
    store = pd.DataFrame(
        [
            _store_row("run_old"),
            _store_row("run_new"),
            _store_row("run_cpu", profile="windows_cpu"),
            _store_row("run_bad", status="failed"),
            _store_row("run_seed", seed=1415),
        ]
    )
    _manifest(tmp_path, "run_old", "2026-07-01T00:00:00Z")
    _manifest(tmp_path, "run_new", "2026-07-02T00:00:00Z")
    sel = {"model": "unet", "variant": "pretrained", "stratum": "all"}
    assert verdict.resolve_run(store, sel, canon, tmp_path) == "run_new"  # newest wins
    with pytest.raises(verdict.Unresolvable):
        verdict.resolve_run(store, {"model": "segformer"}, canon, tmp_path)
    _manifest(tmp_path, "run_old", "2026-07-02T00:00:00Z")  # exact timestamp tie -> raise
    with pytest.raises(ValueError):
        verdict.resolve_run(store, sel, canon, tmp_path)


def test_resolve_run_backbone_by_best_val(tmp_path):
    canon = {"filter": {"seed": 1414, "profile": "gpu_full", "status": "ok"}}
    store = pd.DataFrame(
        [
            _store_row("run_b0", model="segformer", backbone="mit-b0"),
            _store_row("run_b2", model="segformer", backbone="mit-b2"),
        ]
    )
    _manifest(tmp_path, "run_b0", "2026-07-01T00:00:00Z", best_val=0.40)
    _manifest(tmp_path, "run_b2", "2026-07-01T00:00:00Z", best_val=0.55)
    sel = {"model": "segformer", "variant": "pretrained", "stratum": "all"}
    assert verdict.resolve_run(store, sel, canon, tmp_path) == "run_b2"


def test_decide_empty_store_defers_everything(tmp_path):
    hyp_cfg = yaml.safe_load((REPO / "configs" / "hypotheses.yaml").read_text(encoding="utf-8"))
    store = pd.DataFrame(columns=RESULT_COLUMNS)
    v = verdict.decide(store, hyp_cfg, tmp_path, git_sha="test")
    assert set(v["hypotheses"]) == {"H0", "H1", "H2", "H3", "H4", "H5"}
    assert all(h["decision"] == "deferred" for h in v["hypotheses"].values())
    assert v["alpha"] == 0.10 and v["correction"] == "holm"
    assert verdict.leaderboard(store).empty


# ---------------------------------------------------------------------------
# config concreteness (section 9 MS3 gate) + prereg seal
# ---------------------------------------------------------------------------


def test_comparisons_table_pins_section_10():
    """The COMPARISONS table IS the section-10 row-pattern spec — pin every field literally."""
    c = verdict.COMPARISONS
    for member in ("baseline_vs_unet", "baseline_vs_deeplabv3plus", "baseline_vs_segformer"):
        assert c[member]["family"] == "A" and c[member]["tail"] == "greater"
        assert c[member]["a"]["variant"] == "pretrained" and c[member]["a"]["stratum"] == "all"
        assert c[member]["b"] == {"model": "baseline", "stratum": "all"}
    assert c["baseline_vs_unet"]["a"]["model"] == "unet"
    assert c["baseline_vs_deeplabv3plus"]["a"]["model"] == "deeplabv3plus"
    assert c["baseline_vs_segformer"]["a"]["model"] == "segformer"
    for member, model in (
        ("unet_pretrained_vs_scratch", "unet"),
        ("deeplabv3plus_pretrained_vs_scratch", "deeplabv3plus"),
        ("segformer_pretrained_vs_scratch", "segformer"),
    ):
        s = c[member]
        assert s["family"] == "B" and s["tail"] == "greater" and s["match_backbone"] is True
        assert s["a"] == {"model": model, "variant": "pretrained", "stratum": "all"}
        assert s["b"] == {"model": model, "variant": "scratch", "stratum": "all"}
    for member, cnn in (
        ("segformer_vs_unet", "unet"),
        ("segformer_vs_deeplabv3plus", "deeplabv3plus"),
    ):
        s = c[member]
        assert s["family"] == "C" and s["tail"] == "two_sided" and s["per_class"] is True
        # delta orientation segformer - cnn: segformer MUST be side 'a'
        assert s["a"] == {"model": "segformer", "variant": "pretrained", "stratum": "all"}
        assert s["b"] == {"model": cnn, "variant": "pretrained", "stratum": "all"}
    assert c["dinov3_sat_vs_baseline"]["a"] == {
        "model": "dinov3_sat",
        "variant": "finetuned",
        "stratum": "all",
    }
    assert c["sam_zeroshot_vs_baseline"]["a"] == {
        "model": "sam",
        "variant": "zeroshot",
        "stratum": "all",
    }
    for member in ("dinov3_sat_vs_baseline", "sam_zeroshot_vs_baseline"):
        assert c[member]["family"] == "E" and c[member]["tail"] == "greater"
        assert c[member]["b"] == {"model": "baseline", "stratum": "all"}


def test_leaderboard_prefers_gpu_rows():
    store = pd.DataFrame(
        [
            _store_row("run_gpu", value=0.61, profile="gpu_full"),
            _store_row("run_cpu_smoke", value=0.05, profile="windows_cpu"),
            _store_row("run_other_seed", value=0.99, seed=1415),
        ]
    )
    lb = verdict.leaderboard(store)
    assert len(lb) == 1 and lb.iloc[0]["miou"] == pytest.approx(0.61)


def test_hypotheses_yaml_is_concrete():
    text = (REPO / "configs" / "hypotheses.yaml").read_text(encoding="utf-8")
    assert "<pin" not in text and "TODO" not in text and "TBD" not in text
    cfg = yaml.safe_load(text)
    assert cfg["alpha"] == 0.10 and cfg["correction"] == "holm" and cfg["ci_level"] == 0.90
    st = cfg["stats"]
    assert st["n_resamples"] == 10000 and st["resampling_unit"] == "image"
    assert st["ci_method"] == "percentile" and st["bootstrap_seed"] == 0
    assert st["p_estimator"] == "plus_one" and st["seed_reset_per_comparison"] is True
    assert cfg["hypotheses"]["H4"]["drop_threshold"] == 0.15
    assert cfg["hypotheses"]["H4"]["emits_p_value"] is False
    fams = cfg["families"]
    assert fams["A"]["members"] == [
        "baseline_vs_unet",
        "baseline_vs_deeplabv3plus",
        "baseline_vs_segformer",
    ]
    assert len(fams["B"]["members"]) == 3 and len(fams["C"]["members"]) == 2
    assert fams["D"]["members"] == ["best_in_rover_vs_cross_rover"]
    assert len(fams["E"]["members"]) == 2
    # every family member has a comparison spec (D is the deterministic H4, handled separately)
    for fam, fcfg in fams.items():
        if fam == "D":
            continue
        for m in fcfg["members"]:
            assert m in verdict.COMPARISONS, m


def test_data_yaml_is_concrete():
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["expected_test_n"] == 322 and cfg["mer"]["expected_test_n"] == 204
    assert cfg["data"]["test_gold_dir"] == "msl/ncam/labels/test/masked-gold-min3-100agree"
    assert cfg["mer"]["test_gold_dir"] == "mer/labels/test/masked-gold-min3-100agree"
    assert cfg["data"]["camera"] == "ncam" and cfg["data"]["split_seed"] == 1414
    assert cfg["data"]["val_frac"] == 0.2 and cfg["data"]["size"] == 512
    assert cfg["class_weights"]["clip"] == [0.5, 10.0]


def test_prereg_freeze_verify_tamper(tmp_path):
    hyp = yaml.safe_load((REPO / "configs" / "hypotheses.yaml").read_text(encoding="utf-8"))
    data = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    p, s = tmp_path / "PREREG.md", tmp_path / "PREREG.sha256"
    prereg.freeze(hyp, data, p, s)
    prereg.verify(p, s)
    text = p.read_text(encoding="utf-8")
    assert "region-oracle" in text and "alpha = 0.1" in text
    before = text
    prereg.freeze(hyp, data, p, s)  # idempotent: sealed content is never rewritten
    assert p.read_text(encoding="utf-8") == before
    p.write_text(text + "\ntampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.verify(p, s)
    with pytest.raises(RuntimeError, match="missing"):
        prereg.verify(tmp_path / "nope.md", s)
