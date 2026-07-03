"""Model zoo + loss + Lightning smoke tests (MS2). CPU, no network (scratch variants only)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from marsseg.models.zoo import build_model
from marsseg.train.loss import CombinedLoss


@pytest.mark.parametrize(
    "name,kw",
    [
        ("baseline", {}),
        ("unet", {"pretrained": False}),
        ("deeplabv3plus", {"pretrained": False}),
        ("segformer", {"backbone": "b0", "pretrained": False}),
    ],
)
def test_zoo_forward_shapes(name, kw):
    m = build_model(name, num_classes=4, **kw)
    out = m(torch.randn(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 4, 64, 64)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope")


def test_combined_loss_ignore():
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    target[:, 0, :] = 255  # an ignore row
    loss = CombinedLoss(class_weights=[1, 2, 3, 4], ignore_index=255)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def _make_records(root, n=4, size=64):
    img_dir, lab_dir = root / "img", root / "lab"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        ip, lp = img_dir / f"{i}.png", lab_dir / f"{i}.png"
        cv2.imwrite(str(ip), rng.integers(0, 255, (size, size), dtype="uint8"))
        m = rng.integers(0, 4, (size, size), dtype="uint8")
        m[0, 0] = 255
        cv2.imwrite(str(lp), m)
        recs.append({"image": str(ip), "label": str(lp), "name": str(i)})
    return recs


def test_lightning_fast_dev_run(tmp_path):
    import lightning as L

    from marsseg.train.lit import SegDataModule, SegLitModule

    recs = _make_records(tmp_path, n=4, size=64)
    dm = SegDataModule(recs[:3], recs[3:], batch_size=2, size=64)
    model = SegLitModule(model_name="baseline", max_epochs=1, pretrained=False)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)  # 1 train + 1 val batch; must complete without error


# --------------------------------------------------------------------------------------
# foundation (H5, gated) — MS2 gate B: build a module OR skip-and-log, never crash
# --------------------------------------------------------------------------------------


class _RecLogger:
    """Captures log.warning calls (marsseg loggers set propagate=False, so caplog can't)."""

    def __init__(self):
        self.msgs = []

    def warning(self, msg, *args):
        self.msgs.append(msg % args if args else msg)


def test_foundation_skips_and_logs_when_gated(monkeypatch):
    """Gate B skip arm, deterministic on ANY box (cuda gate forced off): None + reason logged.

    No skipif needed and no network possible — the gate trips before any load path.
    """
    from marsseg.models import foundation

    rec = _RecLogger()
    monkeypatch.setattr(foundation, "has_cuda", lambda: False)
    monkeypatch.setattr(foundation, "log", rec)
    assert build_model("dinov3_sat") is None
    assert build_model("sam") is None
    assert len(rec.msgs) == 2  # the "log" half of skip-and-log is contract, not decoration
    assert "dinov3_sat" in rec.msgs[0] and "cuda" in rec.msgs[0]
    assert "sam" in rec.msgs[1] and "cuda" in rec.msgs[1]


def test_foundation_happy_paths_through_registry(monkeypatch, tmp_path):
    """Gate B 'returns a module' arm, offline: gates pass, loaders stubbed, kwargs plumb through."""
    from marsseg.models import foundation

    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    sentinel = torch.nn.Identity()
    seen = {}

    def fake_load(num_classes):
        seen["num_classes"] = num_classes
        return sentinel

    monkeypatch.setattr(foundation, "_load_dinov3", fake_load)
    assert build_model("dinov3_sat", num_classes=4) is sentinel
    assert seen["num_classes"] == 4

    class _StubSam(torch.nn.Module):
        def __init__(self, checkpoint, model_type="vit_b"):
            super().__init__()
            seen["ckpt"] = str(checkpoint)

    ckpt = tmp_path / "sam_vit_b.pth"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(foundation, "_sam_importable", lambda: True)
    monkeypatch.setattr(foundation, "SamZeroShotSegmenter", _StubSam)
    out = build_model("sam", sam_checkpoint=str(ckpt))
    assert isinstance(out, _StubSam)
    assert seen["ckpt"] == str(ckpt)  # model.sam_checkpoint reaches the arm via the registry


def test_foundation_load_failure_skips_not_raises(monkeypatch):
    """Gates pass but the load blows up (e.g. HF 403 pending license) => skip-and-log, no crash."""
    from marsseg.models import foundation

    rec = _RecLogger()
    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(foundation, "log", rec)

    def boom(num_classes):
        raise RuntimeError("403 Forbidden: gated repo, license approval pending")

    monkeypatch.setattr(foundation, "_load_dinov3", boom)
    assert build_model("dinov3_sat") is None
    assert any("load failed" in m and "403" in m for m in rec.msgs)


def test_foundation_gating_reasons(monkeypatch, tmp_path):
    from marsseg.models import foundation

    ckpt = tmp_path / "sam_vit_b.pth"
    ckpt.write_bytes(b"x")
    # dinov3: all gates pass
    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert foundation.gating_reasons("dinov3_sat") == []
    # token gate ALONE (cuda still True)
    monkeypatch.setenv("HF_TOKEN", "")
    reasons = foundation.gating_reasons("dinov3_sat")
    assert len(reasons) == 1 and "HF_TOKEN" in reasons[0]
    # cuda gate ALONE (token restored)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(foundation, "has_cuda", lambda: False)
    reasons = foundation.gating_reasons("dinov3_sat")
    assert len(reasons) == 1 and "cuda" in reasons[0]
    # the cuda gate applies to BOTH arms (5.4 knob row), and sam can fully pass
    monkeypatch.setattr(foundation, "_sam_importable", lambda: True)
    reasons = foundation.gating_reasons("sam", sam_checkpoint=ckpt)
    assert len(reasons) == 1 and "cuda" in reasons[0]
    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    assert foundation.gating_reasons("sam", sam_checkpoint=ckpt) == []
    # sam import + checkpoint gates, each named independently
    monkeypatch.setattr(foundation, "_sam_importable", lambda: False)
    reasons = foundation.gating_reasons("sam", sam_checkpoint=tmp_path / "missing.pth")
    assert any("segment-anything" in r for r in reasons)
    assert any("checkpoint absent" in r for r in reasons)
    with pytest.raises(ValueError):
        foundation.gating_reasons("clip")


def test_dinov3_head_forward_contract():
    """Head/reshape/upsample logic offline via a stub backbone: (B,3,H,W) -> (B,4,H,W)."""
    from types import SimpleNamespace

    from marsseg.models.foundation import DinoV3SatSegmenter

    class _StubViT(torch.nn.Module):
        def __init__(self, hidden=64, patch=16, prefix=5):
            super().__init__()
            self.proj = torch.nn.Linear(hidden, hidden)  # a real param, to assert freezing
            self.hidden, self.patch, self.prefix = hidden, patch, prefix

        def forward(self, pixel_values):
            b, _, h, w = pixel_values.shape
            n = (h // self.patch) * (w // self.patch)
            t = torch.randn(b, self.prefix + n, self.hidden)
            return SimpleNamespace(last_hidden_state=self.proj(t))

    m = DinoV3SatSegmenter(
        _StubViT(), hidden_size=64, patch_size=16, num_prefix_tokens=5, num_classes=4
    )
    out = m(torch.randn(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 4, 64, 64)
    # backbone frozen + pinned to eval; head trainable
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    m.train()
    assert m.training and not m.backbone.training
    # a wrong prefix-token count must fail loudly at first forward, not silently mis-grid
    bad = DinoV3SatSegmenter(
        _StubViT(), hidden_size=64, patch_size=16, num_prefix_tokens=3, num_classes=4
    )
    with pytest.raises(RuntimeError, match="token grid mismatch"):
        bad(torch.randn(1, 3, 64, 64))


def test_dinov3_head_spatial_layout():
    """Value-level grid contract: row-major patch layout, FIRST prefix tokens dropped, channels
    kept per-token (shape-only asserts are blind to transpose/gh-gw-swap/wrong-end-slice bugs)."""
    from types import SimpleNamespace

    from marsseg.models.foundation import DinoV3SatSegmenter

    class _PosViT(torch.nn.Module):
        def forward(self, pixel_values):
            b, _, h, w = pixel_values.shape
            n = (h // 16) * (w // 16)
            pos = torch.arange(n, dtype=torch.float32)
            patches = torch.stack([pos, 1000.0 + pos], dim=1)  # token i: (i, 1000+i)
            prefix = torch.full((3, 2), -7.0)
            tokens = torch.cat([prefix, patches], dim=0).unsqueeze(0).expand(b, -1, -1)
            return SimpleNamespace(last_hidden_state=tokens)

    m = DinoV3SatSegmenter(
        _PosViT(), hidden_size=2, patch_size=16, num_prefix_tokens=3, num_classes=4
    )
    m.head = torch.nn.Identity()  # expose the reshaped grid itself
    out = m(torch.zeros(1, 3, 32, 64))  # 2x4 patch grid; non-square catches gh/gw swaps
    assert tuple(out.shape) == (1, 2, 32, 64)
    # bilinear corners of a 2x4 grid upsampled to 32x64 hit the grid corners exactly
    corners = [out[0, 0, 0, 0], out[0, 0, 0, -1], out[0, 0, -1, 0], out[0, 0, -1, -1]]
    assert [round(float(v)) for v in corners] == [0, 3, 4, 7]  # row-major, prefix gone
    assert round(float(out[0, 1, 0, 0])) == 1000  # channel identity survives the reshape
