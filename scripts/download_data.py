"""Download + extract the AI4Mars dataset from Zenodo (Phase MS1).

AI4Mars (NASA, open): ~50K rover images + ~425K crowdsourced segmentation labels (Curiosity,
Perseverance, Opportunity, Spirit). Labels are single-channel PNGs with terrain pixel values 0-3
(soil, bedrock, sand, big rock) and 255 = NULL/ignore. Resumable chunked download + MD5 verify.

Usage:
  python scripts/download_data.py --out data/raw/ai4mars          # full (16.2 GB) + extract
  python scripts/download_data.py --no-extract                    # just fetch the zip
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

ZENODO = "https://zenodo.org/records/15995036/files"
FILES = {
    "merged": {
        "name": "ai4mars-dataset-merged-0.6.zip",
        "md5": "daf80a86021253292e6c425f97baa5c6",
        "size_gb": 16.2,
    },
    "unmerged": {
        "name": "ai4mars-labels-unmerged.zip",
        "md5": "49fc7a969dfddc0c06d0020edda432c2",
        "size_gb": 1.6,
    },
}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Resumable chunked download (HTTP Range) with a progress print and MD5 verify."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_md5:
        got = _md5(dest)
        if got == expected_md5:
            print(f"  MD5 OK (cached): {dest.name}")
            return dest
        print(f"  cached file MD5 {got} != {expected_md5}; attempting resume/re-download")
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code in (200, 206):
            total = pos + int(r.headers.get("Content-Length", 0))
            mode = "ab" if r.status_code == 206 else "wb"
            done = pos if r.status_code == 206 else 0
            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if done % (256 << 20) < (1 << 20):
                        print(f"  {dest.name}: {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
        else:
            r.raise_for_status()
    if expected_md5:
        got = _md5(dest)
        if got != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {dest.name}: {got} != {expected_md5}")
        print(f"  MD5 OK: {dest.name}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Download AI4Mars.")
    ap.add_argument("--out", default="data/raw/ai4mars")
    ap.add_argument("--which", choices=["merged", "unmerged", "both"], default="merged")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--no-md5", action="store_true", help="skip the (slow) MD5 verify")
    args = ap.parse_args()
    out = Path(args.out)
    keys = ["merged", "unmerged"] if args.which == "both" else [args.which]
    for k in keys:
        meta = FILES[k]
        zip_path = out / meta["name"]
        print(f"downloading {meta['name']} (~{meta['size_gb']} GB) ...", flush=True)
        download(
            f"{ZENODO}/{meta['name']}?download=1", zip_path, None if args.no_md5 else meta["md5"]
        )
        if not args.no_extract:
            print(f"extracting {meta['name']} ...", flush=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out)
            print(f"extracted to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
