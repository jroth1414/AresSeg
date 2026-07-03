"""Aggregate + hypothesis verdicts (MS3; DEVPLAN section 8 CLI).

    python scripts/analyze_results.py --store experiments/results_store.parquet \
        --hypotheses configs/hypotheses.yaml --out experiments/analysis/

Verifies the pre-registration seal, resolves each comparison to canonical runs (5.9), runs the
5.6 paired bootstrap / 5.7 H4 rule, applies Holm per family, and writes
``experiments/manifests/verdicts.json`` + ``experiments/manifests/leaderboard.csv``. Every
hypothesis H0..H5 gets a decision in {support, reject, deferred}; on windows_cpu (or an empty
store) members defer rather than crash — H5 deferred never blocks H1-H4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from marsseg.eval import prereg, verdict  # noqa: E402
from marsseg.utils.config import load_yaml  # noqa: E402
from marsseg.utils.logging import get_logger  # noqa: E402
from marsseg.utils.manifest import _git_sha  # noqa: E402
from marsseg.utils.results import RESULT_COLUMNS  # noqa: E402
from marsseg.utils.seed import set_seed  # noqa: E402

log = get_logger("marsseg.analyze")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="experiments/results_store.parquet")
    ap.add_argument("--hypotheses", default="configs/hypotheses.yaml")
    ap.add_argument("--out", default="experiments/analysis/")
    args = ap.parse_args(argv)
    set_seed(1414)

    hyp_cfg = load_yaml(args.hypotheses)
    data_cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    # PREREG must be sealed BEFORE any verdict pass; freeze() is a no-op if already sealed.
    prereg.freeze(
        hyp_cfg,
        data_cfg,
        prereg_path=REPO_ROOT / "experiments" / "PREREG.md",
        sha_path=REPO_ROOT / "experiments" / "manifests" / "PREREG.sha256",
    )
    prereg.verify(
        REPO_ROOT / "experiments" / "PREREG.md",
        REPO_ROOT / "experiments" / "manifests" / "PREREG.sha256",
    )

    store_path = Path(args.store)
    if store_path.is_file():
        store = pd.read_parquet(store_path)
    else:
        log.warning("results store %s missing — deciding on an empty store", store_path)
        store = pd.DataFrame(columns=RESULT_COLUMNS)

    manifests_dir = REPO_ROOT / "experiments" / "manifests"
    verdicts = verdict.decide(store, hyp_cfg, manifests_dir, git_sha=_git_sha())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = manifests_dir / "verdicts.json"
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts_path.write_text(json.dumps(verdicts, indent=2, default=str), encoding="utf-8")
    lb = verdict.leaderboard(store)
    lb.to_csv(manifests_dir / "leaderboard.csv", index=False)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)

    for hid, h in verdicts["hypotheses"].items():
        log.info("%s: %s", hid, h["decision"])
    log.info("wrote %s and leaderboard.csv (%d rows)", verdicts_path, len(lb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
