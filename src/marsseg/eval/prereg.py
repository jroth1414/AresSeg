"""Pre-registration freeze/verify (MS3). PREREG.md is written ONCE, before any test-set number.

``freeze()`` renders the frozen protocol (hypotheses, families, thresholds, seeds, pinned test
sets, and the SAM region-oracle scoring rule) to ``experiments/PREREG.md`` and records its
SHA-256 in ``experiments/manifests/PREREG.sha256``. If the file already exists it is NEVER
rewritten — ``verify()`` checks the hash and raises on tamper/mismatch. ``analyze_results.py``
refuses to compute verdicts unless ``verify()`` passes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PREREG_PATH = Path("experiments/PREREG.md")
SHA_PATH = Path("experiments/manifests/PREREG.sha256")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(hyp_cfg: dict, data_cfg: dict) -> str:
    """Markdown rendering of the frozen protocol (content comes from the committed configs)."""
    stats = hyp_cfg["stats"]
    fams = hyp_cfg["families"]
    lines = [
        "# Pre-registration — marsseg (frozen BEFORE any test-set number is computed)",
        "",
        f"- Significance: alpha = {hyp_cfg['alpha']}, correction = {hyp_cfg['correction']} "
        f"(within family), CI level = {hyp_cfg['ci_level']} (percentile).",
        f"- Primary metric: {hyp_cfg['primary_metric']} (macro over the fixed class set); "
        f"per-class metric: {hyp_cfg['per_class_metric']}.",
        f"- Descriptive-only (NEVER tested): {', '.join(hyp_cfg['descriptive_only'])}.",
        f"- Bootstrap: n_resamples = {stats['n_resamples']}, unit = {stats['resampling_unit']}, "
        f"seed = {stats['bootstrap_seed']} (reset per comparison), p = {stats['p_estimator']}, "
        f"empty fixed-set class in a resample contributes IoU = 0.",
        "- Training/split seed = 1414; by-image splits, val_frac = 0.2.",
        f"- MSL test set: {data_cfg['data']['test_gold_dir']} "
        f"(n = {data_cfg['data']['expected_test_n']}).",
        f"- MER test set (H4): {data_cfg['mer']['test_gold_dir']} "
        f"(n = {data_cfg['mer']['expected_test_n']}); MER is never trained on.",
        "",
        "## Families",
        "",
    ]
    for fam, fcfg in fams.items():
        lines.append(f"- **{fam}**: {', '.join(fcfg['members'])}")
    lines += ["", "## Hypotheses & decision rules", ""]
    for hid, hcfg in hyp_cfg["hypotheses"].items():
        lines.append(f"- **{hid}**: {hcfg.get('decision_rule', hcfg)}")
    lines += [
        "",
        "## H5 SAM scoring rule (frozen)",
        "",
        "SAM emits class-AGNOSTIC region proposals and AI4Mars has no prompt channel. The SAM",
        "zero-shot arm is scored with the **region-oracle assignment**: each SAM proposal takes",
        "the majority ground-truth class among its valid (non-ignore) pixels (later proposals",
        "overwrite earlier ones where they overlap); pixels outside every proposal are assigned",
        "class 0 (soil, the majority terrain class). This is an EXPLICIT UPPER BOUND on any",
        "zero-shot region labeler built on SAM's masks, and is reported as such.",
        "",
        "H0 is reported honestly: it holds iff H1 is not rejected.",
        "",
    ]
    return "\n".join(lines)


def freeze(
    hyp_cfg: dict,
    data_cfg: dict,
    prereg_path: Path = PREREG_PATH,
    sha_path: Path = SHA_PATH,
) -> Path:
    """Write PREREG.md once + record its SHA. Existing content is never overwritten."""
    prereg_path = Path(prereg_path)
    sha_path = Path(sha_path)
    if prereg_path.is_file():
        verify(prereg_path, sha_path)  # existing prereg must be intact; never rewrite
        return prereg_path
    text = render(hyp_cfg, data_cfg)
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    prereg_path.write_text(text, encoding="utf-8")
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(_sha256(text) + "\n", encoding="utf-8")
    return prereg_path


def verify(prereg_path: Path = PREREG_PATH, sha_path: Path = SHA_PATH) -> None:
    """Raise unless PREREG.md exists and matches its recorded SHA-256."""
    prereg_path, sha_path = Path(prereg_path), Path(sha_path)
    if not prereg_path.is_file():
        raise RuntimeError(f"{prereg_path} missing — run eval.prereg.freeze() BEFORE analysis")
    if not sha_path.is_file():
        raise RuntimeError(f"{sha_path} missing — the pre-registration was never sealed")
    actual = _sha256(prereg_path.read_text(encoding="utf-8"))
    expected = sha_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError(
            f"PREREG.md hash mismatch (expected {expected[:12]}…, got {actual[:12]}…) — "
            "the pre-registration must not change after sealing"
        )
