"""Scaffold smoke tests (MS0): package imports, capability detection, seeding, manifest."""

from __future__ import annotations

import json

import numpy as np

from aresseg.utils import manifest as manifest_module
from aresseg.utils.capabilities import detect
from aresseg.utils.manifest import REPO_ROOT, write_manifest
from aresseg.utils.seed import set_seed


def test_package_imports():
    import aresseg

    assert aresseg.__version__ == "0.1.0"


def test_capabilities_profile():
    caps = detect()
    assert caps.profile in ("windows_cpu", "gpu_full")
    # boolean flags present
    for f in ("cuda", "smp", "transformers", "sam", "timm"):
        assert isinstance(getattr(caps, f), bool)


def test_seed_reproducible():
    set_seed(1414)
    a = np.random.rand(5)
    set_seed(1414)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_manifest_superset(tmp_path):
    p = write_manifest(tmp_path / "run", {"k": "v"}, seed=1414, model="unet", task="seg")
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["seed"] == 1414 and m["model"] == "unet"
    assert "git_sha" in m and "capabilities" in m and "config_hash" in m


def test_git_provenance_is_repo_scoped_and_safe_directory_aware(monkeypatch):
    calls = []

    def fake_check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "abc123\n" if command[-2:] == ["rev-parse", "HEAD"] else " M tracked.py\n"

    monkeypatch.setattr(manifest_module.subprocess, "check_output", fake_check_output)

    assert manifest_module._git_sha() == "abc123"
    assert manifest_module._git_dirty() is True
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[:3] == ["git", "-c", f"safe.directory={REPO_ROOT}"]
        assert kwargs["cwd"] == REPO_ROOT
        assert kwargs["text"] is True
        assert kwargs["stderr"] is manifest_module.subprocess.DEVNULL
