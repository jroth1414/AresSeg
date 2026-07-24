"""Environment/capability report for the aresseg project. Used at run start + CI smoke.

Prints the detected profile + key CV package versions. Core smoke EXCLUDES the gated foundation
packages (segment-anything) so a missing optional wheel never fails the gate.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aresseg.utils.capabilities import detect  # noqa: E402

CORE_SMOKE = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "skimage",
    "matplotlib",
    "cv2",
    "PIL",
    "albumentations",
    "torch",
    "torchvision",
    "segmentation_models_pytorch",
    "transformers",
]


def _ver(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as exc:  # pragma: no cover
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    caps = detect()
    print(f"profile={caps.profile}")
    print(f"cuda={caps.cuda} gpu={caps.gpu_name}")
    print(f"smp={caps.smp} transformers={caps.transformers} sam={caps.sam} timm={caps.timm}")
    print("--- core package versions ---")
    missing = []
    for mod in CORE_SMOKE:
        v = _ver(mod)
        if v.startswith("MISSING"):
            missing.append(mod)
        print(f"  {mod}: {v}")
    if missing:
        print(f"WARNING: missing core packages: {missing}", file=sys.stderr)
        return 1
    print("core OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
