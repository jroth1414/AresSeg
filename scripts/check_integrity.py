"""Fail closed on immutable-rubric or reference-lock drift (CI/pre-commit entry point)."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sidecar = ROOT / "RESEARCH.MD.sha256"
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((ROOT / "RESEARCH.MD").read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"RESEARCH.MD checksum mismatch: expected {expected}, got {actual}")

    lock_bytes = (ROOT / "requirements.lock.txt").read_bytes()
    if b"\x00" in lock_bytes:
        raise SystemExit("requirements.lock.txt must be UTF-8, not UTF-16/NUL-delimited")
    lock = lock_bytes.decode("utf-8")
    forbidden = (
        "AAML-Research-Project.git",
        "AresSeg.git",
        "#egg=marsseg",
        "#egg=aresseg",
    )
    if any(token in lock for token in forbidden) or any(
        line.startswith(("-e ", "git+")) for line in lock.splitlines()
    ):
        raise SystemExit("requirements.lock.txt must contain third-party packages only")
    print("repository integrity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
