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


def test_foundation_skips_and_logs_on_this_box():
    """windows_cpu + empty HF_TOKEN + no SAM extras/ckpt => both arms return None, no raise."""
    assert build_model("dinov3_sat") is None
    assert build_model("sam") is None


def test_foundation_gating_reasons(monkeypatch, tmp_path):
    from marsseg.models import foundation

    # all dinov3 gates pass when cuda + token are present
    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert foundation.gating_reasons("dinov3_sat") == []
    # each gate trips independently and is named in the reason
    monkeypatch.setenv("HF_TOKEN", "")
    assert any("HF_TOKEN" in r for r in foundation.gating_reasons("dinov3_sat"))
    monkeypatch.setattr(foundation, "has_cuda", lambda: False)
    assert any("cuda" in r for r in foundation.gating_reasons("dinov3_sat"))
    # sam: import + checkpoint gates (segment-anything is not installed on this box)
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
