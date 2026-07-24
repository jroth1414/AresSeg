"""Aggregate + hypothesis verdicts (MS3; DEVPLAN section 8 CLI).

    python scripts/analyze_results.py --store experiments/results_store.parquet \
        --hypotheses configs/hypotheses.yaml --out experiments/analysis/

Verifies the complete historical chain and executable Protocol V4 seal, resolves each
comparison to complete primary-seed run sets, runs the paired seed/image bootstrap / H4 rule,
applies Holm per family, and writes
``experiments/manifests/verdicts.json`` + ``experiments/manifests/leaderboard.csv``. Every
hypothesis H0..H5 gets an explicit decision; on an empty/incomplete store members defer rather
than falling back to smoke results. H0 non-significance is ``fail_to_reject``, never support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from aresseg.eval import prereg, verdict  # noqa: E402
from aresseg.utils.config import load_yaml  # noqa: E402
from aresseg.utils.logging import get_logger  # noqa: E402
from aresseg.utils.manifest import _git_sha  # noqa: E402
from aresseg.utils.results import RESULT_COLUMNS  # noqa: E402
from aresseg.utils.seed import set_seed  # noqa: E402

log = get_logger("aresseg.analyze")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="experiments/results_store.parquet")
    ap.add_argument("--hypotheses", default="configs/hypotheses.yaml")
    ap.add_argument("--out", default="experiments/analysis/")
    args = ap.parse_args(argv)
    set_seed(1414)

    hyp_cfg = load_yaml(args.hypotheses)
    data_cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    # Verification is read-only and occurs before even loading the result store. Missing or
    # drifted protocol inputs fail closed; analysis must never create a seal after seeing data.
    prereg.verify_protocol(
        hyp_cfg,
        data_cfg,
        repo_root=REPO_ROOT,
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
