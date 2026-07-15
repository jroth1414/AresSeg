"""Reject prohibited authorship trailers without rewriting the user's commit message."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROHIBITED = re.compile(r"(?im)^Co-?Authored-?By:.*(?:Claude|Anthropic|ChatGPT|OpenAI|Codex)")


def main(path: str) -> int:
    message = Path(path).read_text(encoding="utf-8")
    if PROHIBITED.search(message):
        raise SystemExit("prohibited AI co-author trailer in commit message")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
