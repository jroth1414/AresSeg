"""Merge GPU-produced result rows into the canonical store, dedup on DEDUP_KEYS (MS3/MS4).

    python scripts/merge_results.py --incoming <gpu_store.parquet> \
        --into experiments/results_store.parquet

Incoming rows win on key collisions (a re-run supersedes). After the merge the store contains
zero duplicates on DEDUP_KEYS; both the parquet and the csv mirror are rewritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from aresseg.utils.logging import get_logger  # noqa: E402
from aresseg.utils.results import DEDUP_KEYS, RESULT_COLUMNS  # noqa: E402

log = get_logger("aresseg.merge")


def merge(incoming: pd.DataFrame, into: pd.DataFrame) -> pd.DataFrame:
    for df in (incoming, into):
        for c in RESULT_COLUMNS:
            if c not in df.columns:
                df[c] = None
    merged = pd.concat([into[RESULT_COLUMNS], incoming[RESULT_COLUMNS]], ignore_index=True)
    merged = merged.drop_duplicates(subset=DEDUP_KEYS, keep="last").reset_index(drop=True)
    assert int(merged.duplicated(subset=DEDUP_KEYS).sum()) == 0
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--into", default="experiments/results_store.parquet")
    args = ap.parse_args(argv)
    incoming = pd.read_parquet(args.incoming)
    into_path = Path(args.into)
    into = (
        pd.read_parquet(into_path) if into_path.is_file() else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    merged = merge(incoming, into)
    into_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(into_path, index=False)
    merged.to_csv(into_path.with_suffix(".csv"), index=False)
    log.info(
        "merged %d incoming rows into %s (%d total, 0 dups on DEDUP_KEYS)",
        len(incoming),
        into_path,
        len(merged),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
