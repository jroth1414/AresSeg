"""Legacy preregistration and latest Protocol V5 executable-snapshot verification.

``freeze()`` renders the frozen protocol (hypotheses, families, thresholds, seeds, pinned test
sets, and the SAM region-oracle scoring rule) to ``experiments/PREREG.md`` and records its
SHA-256 in ``experiments/manifests/PREREG.sha256``. If the file already exists it is NEVER
rewritten — ``verify()`` checks the hash and raises on tamper/mismatch.

No amendment rewrites an earlier historical artifact. ``seal_protocol()`` creates a canonical
Protocol V5 JSON snapshot binding the complete V1--V4 chain, the majority-class correction, the full live decision configuration, every model YAML, and hashes of every
result-driving implementation. Production analysis calls ``verify_protocol()`` only: absence,
non-canonical JSON, artifact tampering, or any live-input drift fails closed before results load.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

PREREG_PATH = Path("experiments/PREREG.md")
SHA_PATH = Path("experiments/manifests/PREREG.sha256")
V2_AMENDMENT_PATH = Path("experiments/PREREG_AMENDMENT_2026-07-10.md")
V2_PROTOCOL_PATH = Path("experiments/manifests/PROTOCOL_V2.json")
V2_PROTOCOL_SHA_PATH = Path("experiments/manifests/PROTOCOL_V2.sha256")
V3_AMENDMENT_PATH = Path("experiments/PREREG_AMENDMENT_RUNTIME_SAFETENSORS_2026-07-10.md")
V3_PROTOCOL_PATH = Path("experiments/manifests/PROTOCOL_V3.json")
V3_PROTOCOL_SHA_PATH = Path("experiments/manifests/PROTOCOL_V3.sha256")
V4_AMENDMENT_PATH = Path("experiments/PREREG_AMENDMENT_ARESSEG_FINAL_2026-07-24.md")
V4_PROTOCOL_PATH = Path("experiments/manifests/PROTOCOL_V4.json")
V4_PROTOCOL_SHA_PATH = Path("experiments/manifests/PROTOCOL_V4.sha256")
V5_AMENDMENT_PATH = Path("experiments/PREREG_AMENDMENT_MAJORITY_CLASS_2026-07-24.md")
PROTOCOL_PATH = Path("experiments/manifests/PROTOCOL_V5.json")
PROTOCOL_SHA_PATH = Path("experiments/manifests/PROTOCOL_V5.sha256")
# These schema names identify immutable historical formats.  They intentionally retain the
# original project namespace after the user-facing project and Python package became AresSeg.
PROTOCOL_V2_SCHEMA = "marsseg.protocol_snapshot.v2"
PROTOCOL_V3_SCHEMA = "marsseg.protocol_snapshot.v3"
PROTOCOL_V4_SCHEMA = "aresseg.protocol_snapshot.v4"
PROTOCOL_V5_SCHEMA = "aresseg.protocol_snapshot.v5"
# ``AMENDMENT_PATH`` remains the public name for the amendment belonging to the latest seal.
AMENDMENT_PATH = V5_AMENDMENT_PATH
RESULT_DRIVING_CODE_PATHS = (
    Path("src/aresseg/data/ai4mars.py"),
    Path("src/aresseg/data/dataset.py"),
    Path("src/aresseg/data/preflight.py"),
    Path("src/aresseg/data/transforms.py"),
    Path("src/aresseg/models/zoo.py"),
    Path("src/aresseg/models/foundation.py"),
    Path("src/aresseg/train/loss.py"),
    Path("src/aresseg/train/lit.py"),
    Path("src/aresseg/eval/aggregate.py"),
    Path("src/aresseg/eval/metrics.py"),
    Path("src/aresseg/eval/prereg.py"),
    Path("src/aresseg/eval/stats.py"),
    Path("src/aresseg/eval/verdict.py"),
    Path("src/aresseg/utils/capabilities.py"),
    Path("src/aresseg/utils/config.py"),
    Path("src/aresseg/utils/manifest.py"),
    Path("src/aresseg/utils/results.py"),
    Path("src/aresseg/utils/seed.py"),
    Path("scripts/check_data.py"),
    Path("scripts/run_experiment.py"),
    Path("scripts/run_gpu.sh"),
    Path("scripts/analyze_results.py"),
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(payload: dict) -> str:
    """Stable byte representation used for both the snapshot and its sidecar."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _existing_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def _validated_historical_protocol(
    snapshot_path: Path, sidecar_path: Path, *, version: str, schema: str
) -> None:
    """Validate an immutable earlier seal as a historical link, never against live inputs."""
    snapshot_path = _existing_file(snapshot_path, f"Protocol {version} snapshot")
    sidecar_path = _existing_file(sidecar_path, f"Protocol {version} SHA sidecar")
    text = snapshot_path.read_text(encoding="utf-8")
    expected_sha = sidecar_path.read_text(encoding="utf-8").strip()
    actual_sha = _sha256(text)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Protocol {version} snapshot hash mismatch (expected {expected_sha[:12]}…, "
            f"got {actual_sha[:12]}…)"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Protocol {version} snapshot is not valid JSON") from exc
    if text != canonical_json(payload):
        raise RuntimeError(f"Protocol {version} snapshot is not in canonical JSON form")
    if payload.get("schema") != schema:
        raise RuntimeError(f"Protocol {version} snapshot has an unexpected schema")


def build_protocol_snapshot(
    hyp_cfg: dict,
    data_cfg: dict,
    *,
    repo_root: Path,
    amendment_path: Path | None = None,
    prereg_path: Path | None = None,
    prereg_sha_path: Path | None = None,
    v2_amendment_path: Path | None = None,
    v2_snapshot_path: Path | None = None,
    v2_snapshot_sha_path: Path | None = None,
    v3_snapshot_path: Path | None = None,
    v3_snapshot_sha_path: Path | None = None,
    v4_snapshot_path: Path | None = None,
    v4_snapshot_sha_path: Path | None = None,
) -> dict:
    """Build the complete deterministic Protocol V5 payload from live and historical inputs."""
    repo_root = Path(repo_root).resolve()
    amendment_path = Path(amendment_path or repo_root / AMENDMENT_PATH)
    prereg_path = Path(prereg_path or repo_root / PREREG_PATH)
    prereg_sha_path = Path(prereg_sha_path or repo_root / SHA_PATH)
    v2_amendment_path = Path(v2_amendment_path or repo_root / V2_AMENDMENT_PATH)
    v2_snapshot_path = Path(v2_snapshot_path or repo_root / V2_PROTOCOL_PATH)
    v2_snapshot_sha_path = Path(v2_snapshot_sha_path or repo_root / V2_PROTOCOL_SHA_PATH)
    v3_snapshot_path = Path(v3_snapshot_path or repo_root / V3_PROTOCOL_PATH)
    v3_snapshot_sha_path = Path(v3_snapshot_sha_path or repo_root / V3_PROTOCOL_SHA_PATH)
    v4_snapshot_path = Path(v4_snapshot_path or repo_root / V4_PROTOCOL_PATH)
    v4_snapshot_sha_path = Path(v4_snapshot_sha_path or repo_root / V4_PROTOCOL_SHA_PATH)
    _existing_file(amendment_path, "Protocol V5 amendment")
    _existing_file(prereg_path, "legacy preregistration")
    _existing_file(prereg_sha_path, "legacy preregistration SHA sidecar")
    _existing_file(v2_amendment_path, "Protocol V2 amendment")
    _existing_file(repo_root / V3_AMENDMENT_PATH, "Protocol V3 amendment")
    _existing_file(repo_root / V4_AMENDMENT_PATH, "Protocol V4 amendment")
    _validated_historical_protocol(
        v2_snapshot_path, v2_snapshot_sha_path, version="V2", schema=PROTOCOL_V2_SCHEMA
    )
    _validated_historical_protocol(
        v3_snapshot_path, v3_snapshot_sha_path, version="V3", schema=PROTOCOL_V3_SCHEMA
    )
    _validated_historical_protocol(
        v4_snapshot_path, v4_snapshot_sha_path, version="V4", schema=PROTOCOL_V4_SCHEMA
    )
    if not isinstance(hyp_cfg.get("comparisons"), dict) or not hyp_cfg["comparisons"]:
        raise RuntimeError("hypotheses config lacks executable comparison selectors")

    model_paths = sorted((repo_root / "configs" / "models").glob("*.yaml"))
    if not model_paths:
        raise RuntimeError("no model configs found under configs/models")
    model_hashes = {
        path.relative_to(repo_root).as_posix(): _file_sha256(path) for path in model_paths
    }
    code_hashes = {}
    for relative in RESULT_DRIVING_CODE_PATHS:
        path = _existing_file(repo_root / relative, "result-driving code")
        code_hashes[relative.as_posix()] = _file_sha256(path)

    return {
        "schema": PROTOCOL_V5_SCHEMA,
        "amendment_date": "2026-07-24",
        "historical_artifact_sha256": {
            PREREG_PATH.as_posix(): _file_sha256(prereg_path),
            SHA_PATH.as_posix(): _file_sha256(prereg_sha_path),
            V2_AMENDMENT_PATH.as_posix(): _file_sha256(v2_amendment_path),
            V2_PROTOCOL_PATH.as_posix(): _file_sha256(v2_snapshot_path),
            V2_PROTOCOL_SHA_PATH.as_posix(): _file_sha256(v2_snapshot_sha_path),
            V3_AMENDMENT_PATH.as_posix(): _file_sha256(repo_root / V3_AMENDMENT_PATH),
            V3_PROTOCOL_PATH.as_posix(): _file_sha256(v3_snapshot_path),
            V3_PROTOCOL_SHA_PATH.as_posix(): _file_sha256(v3_snapshot_sha_path),
            V4_AMENDMENT_PATH.as_posix(): _file_sha256(repo_root / V4_AMENDMENT_PATH),
            V4_PROTOCOL_PATH.as_posix(): _file_sha256(v4_snapshot_path),
            V4_PROTOCOL_SHA_PATH.as_posix(): _file_sha256(v4_snapshot_sha_path),
            V5_AMENDMENT_PATH.as_posix(): _file_sha256(amendment_path),
        },
        "decision_inputs": {
            "data_config": deepcopy(data_cfg),
            "hypotheses_config": deepcopy(hyp_cfg),
            "comparison_selectors": deepcopy(hyp_cfg["comparisons"]),
        },
        "model_config_sha256": model_hashes,
        "result_driving_code_sha256": code_hashes,
    }


def _first_difference(expected, actual, path: str = "snapshot") -> str:
    """Return one concise deterministic drift location for an actionable error."""
    if type(expected) is not type(actual):
        return f"{path} type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path} keys differ"
        for key in sorted(expected):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if expected == actual else f"{path} differs"


def render(hyp_cfg: dict, data_cfg: dict) -> str:
    """Markdown rendering of the frozen protocol (content comes from the committed configs)."""
    stats = hyp_cfg["stats"]
    fams = hyp_cfg["families"]
    lines = [
        "# Pre-registration — AresSeg (frozen BEFORE any test-set number is computed)",
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
        "H0 reporting: reject iff H1 is supported; otherwise report fail_to_reject. "
        "Non-significance never supports H0.",
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


def seal_protocol(
    hyp_cfg: dict,
    data_cfg: dict,
    *,
    repo_root: Path,
    snapshot_path: Path | None = None,
    snapshot_sha_path: Path | None = None,
    amendment_path: Path | None = None,
    prereg_path: Path | None = None,
    prereg_sha_path: Path | None = None,
    v2_amendment_path: Path | None = None,
    v2_snapshot_path: Path | None = None,
    v2_snapshot_sha_path: Path | None = None,
    v3_snapshot_path: Path | None = None,
    v3_snapshot_sha_path: Path | None = None,
    v4_snapshot_path: Path | None = None,
    v4_snapshot_sha_path: Path | None = None,
) -> Path:
    """Write Protocol V5 once; an existing seal is verified and never overwritten."""
    repo_root = Path(repo_root).resolve()
    snapshot_path = Path(snapshot_path or repo_root / PROTOCOL_PATH)
    snapshot_sha_path = Path(snapshot_sha_path or repo_root / PROTOCOL_SHA_PATH)
    amendment_path = Path(amendment_path or repo_root / AMENDMENT_PATH)
    prereg_path = Path(prereg_path or repo_root / PREREG_PATH)
    prereg_sha_path = Path(prereg_sha_path or repo_root / SHA_PATH)
    verify(prereg_path, prereg_sha_path)
    if snapshot_path.exists() or snapshot_sha_path.exists():
        if not snapshot_path.is_file() or not snapshot_sha_path.is_file():
            raise RuntimeError(
                "partial Protocol V5 seal — snapshot and SHA sidecar are both required"
            )
        verify_protocol(
            hyp_cfg,
            data_cfg,
            repo_root=repo_root,
            snapshot_path=snapshot_path,
            snapshot_sha_path=snapshot_sha_path,
            amendment_path=amendment_path,
            prereg_path=prereg_path,
            prereg_sha_path=prereg_sha_path,
            v2_amendment_path=v2_amendment_path,
            v2_snapshot_path=v2_snapshot_path,
            v2_snapshot_sha_path=v2_snapshot_sha_path,
            v3_snapshot_path=v3_snapshot_path,
            v3_snapshot_sha_path=v3_snapshot_sha_path,
            v4_snapshot_path=v4_snapshot_path,
            v4_snapshot_sha_path=v4_snapshot_sha_path,
        )
        return snapshot_path
    payload = build_protocol_snapshot(
        hyp_cfg,
        data_cfg,
        repo_root=repo_root,
        amendment_path=amendment_path,
        prereg_path=prereg_path,
        prereg_sha_path=prereg_sha_path,
        v2_amendment_path=v2_amendment_path,
        v2_snapshot_path=v2_snapshot_path,
        v2_snapshot_sha_path=v2_snapshot_sha_path,
        v3_snapshot_path=v3_snapshot_path,
        v3_snapshot_sha_path=v3_snapshot_sha_path,
        v4_snapshot_path=v4_snapshot_path,
        v4_snapshot_sha_path=v4_snapshot_sha_path,
    )
    text = canonical_json(payload)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(text, encoding="utf-8")
    snapshot_sha_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_sha_path.write_text(_sha256(text) + "\n", encoding="utf-8")
    return snapshot_path


def verify_protocol(
    hyp_cfg: dict,
    data_cfg: dict,
    *,
    repo_root: Path,
    snapshot_path: Path | None = None,
    snapshot_sha_path: Path | None = None,
    amendment_path: Path | None = None,
    prereg_path: Path | None = None,
    prereg_sha_path: Path | None = None,
    v2_amendment_path: Path | None = None,
    v2_snapshot_path: Path | None = None,
    v2_snapshot_sha_path: Path | None = None,
    v3_snapshot_path: Path | None = None,
    v3_snapshot_sha_path: Path | None = None,
    v4_snapshot_path: Path | None = None,
    v4_snapshot_sha_path: Path | None = None,
) -> None:
    """Fail closed unless the complete historical chain and Protocol V5 match live inputs."""
    repo_root = Path(repo_root).resolve()
    snapshot_path = Path(snapshot_path or repo_root / PROTOCOL_PATH)
    snapshot_sha_path = Path(snapshot_sha_path or repo_root / PROTOCOL_SHA_PATH)
    amendment_path = Path(amendment_path or repo_root / AMENDMENT_PATH)
    prereg_path = Path(prereg_path or repo_root / PREREG_PATH)
    prereg_sha_path = Path(prereg_sha_path or repo_root / SHA_PATH)
    verify(prereg_path, prereg_sha_path)
    _existing_file(snapshot_path, "Protocol V5 snapshot")
    _existing_file(snapshot_sha_path, "Protocol V5 SHA sidecar")

    text = snapshot_path.read_text(encoding="utf-8")
    expected_sha = snapshot_sha_path.read_text(encoding="utf-8").strip()
    actual_sha = _sha256(text)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Protocol V5 snapshot hash mismatch (expected {expected_sha[:12]}…, "
            f"got {actual_sha[:12]}…)"
        )
    try:
        sealed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Protocol V5 snapshot is not valid JSON") from exc
    if text != canonical_json(sealed):
        raise RuntimeError("Protocol V5 snapshot is not in canonical JSON form")

    live = build_protocol_snapshot(
        hyp_cfg,
        data_cfg,
        repo_root=repo_root,
        amendment_path=amendment_path,
        prereg_path=prereg_path,
        prereg_sha_path=prereg_sha_path,
        v2_amendment_path=v2_amendment_path,
        v2_snapshot_path=v2_snapshot_path,
        v2_snapshot_sha_path=v2_snapshot_sha_path,
        v3_snapshot_path=v3_snapshot_path,
        v3_snapshot_sha_path=v3_snapshot_sha_path,
        v4_snapshot_path=v4_snapshot_path,
        v4_snapshot_sha_path=v4_snapshot_sha_path,
    )
    difference = _first_difference(sealed, live)
    if difference:
        raise RuntimeError(f"Protocol V5 live-input drift: {difference}; analysis refused")
