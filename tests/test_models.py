"""Model zoo + loss + Lightning smoke tests (MS2). CPU, no network (scratch variants only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from aresseg.models.zoo import build_model
from aresseg.train.loss import CombinedLoss


@pytest.mark.parametrize(
    "name,kw",
    [
        ("majority", {}),
        ("tiny_unet", {}),
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


def test_majority_is_parameter_free_constant_soil():
    model = build_model("majority", num_classes=4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    logits = model(torch.randn(2, 3, 13, 17))
    assert tuple(logits.shape) == (2, 4, 13, 17)
    assert set(logits.argmax(1).unique().tolist()) == {0}


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope")


def test_segformer_pretrained_loads_verified_safetensors_encoder_only(
    monkeypatch, tmp_path, capsys
):
    import sys
    from types import SimpleNamespace

    from aresseg.models import zoo

    seen = {}
    test_token = "hf_test_mit_token_must_not_be_logged"
    monkeypatch.setenv("HF_TOKEN", test_token)
    payload = b"verified test safetensors payload"
    (tmp_path / "model.safetensors").write_bytes(payload)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(zoo.SEGFORMER_SAFETENSORS["b0"], "revision", "deadbeef")
    monkeypatch.setitem(
        zoo.SEGFORMER_SAFETENSORS["b0"], "sha256", hashlib.sha256(payload).hexdigest()
    )

    def fake_hf_hub_download(*, repo_id, filename, revision, token):
        assert repo_id == "nvidia/mit-b0"
        assert revision == "deadbeef"
        assert token == test_token
        return str(tmp_path / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    class Config:
        pass

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["model_id"] = str(model_id)
            seen.update(kwargs)
            return cls()

    class SemanticModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            seen["decoder_initialized"] = True
            self.segformer = torch.nn.Identity()
            self.config = config

    class ScratchConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake = SimpleNamespace(
        SegformerConfig=ScratchConfig,
        SegformerForSemanticSegmentation=SemanticModel,
        SegformerModel=Encoder,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    wrapped = build_model(
        "segformer",
        num_classes=4,
        backbone="b0",
        pretrained=True,
        revision="deadbeef",
    )
    assert seen == {
        "model_id": str(tmp_path),
        "use_safetensors": True,
        "local_files_only": True,
        "decoder_initialized": True,
    }
    assert isinstance(wrapped.model.segformer, Encoder)
    assert wrapped.model.config.num_labels == 4
    captured = capsys.readouterr()
    assert test_token not in captured.out
    assert test_token not in captured.err


@pytest.mark.parametrize("variant", ["b0", "b2"])
def test_segformer_config_matches_safetensors_pin(variant):
    from aresseg.models.zoo import SEGFORMER_SAFETENSORS

    config_path = Path(__file__).parents[1] / "configs" / "models" / f"segformer_{variant}.yaml"
    model = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    pin = SEGFORMER_SAFETENSORS[variant]
    assert model["revision"] == pin["revision"]
    assert model["weights_filename"] == "model.safetensors"
    assert model["weights_sha256"] == pin["sha256"]


def test_segformer_safe_loader_fails_closed(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from aresseg.models import zoo

    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_hf_hub_download(*, repo_id, filename, revision, token):
        return str(tmp_path / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )
    pin = zoo.SEGFORMER_SAFETENSORS["b0"]
    with pytest.raises(ValueError, match="requires safetensors revision"):
        zoo._verified_segformer_snapshot("b0", "unpinned")
    with pytest.raises(RuntimeError, match="safetensors sha256 mismatch"):
        zoo._verified_segformer_snapshot("b0", pin["revision"])


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

    from aresseg.train.lit import SegDataModule, SegLitModule

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


def test_dataloader_seed_controls_shuffle(tmp_path):
    from aresseg.train.lit import SegDataModule

    records = _make_records(tmp_path, n=8, size=16)
    first = SegDataModule(records, records[:1], batch_size=2, size=16, seed=7)
    same = SegDataModule(records, records[:1], batch_size=2, size=16, seed=7)
    different = SegDataModule(records, records[:1], batch_size=2, size=16, seed=8)
    for module in (first, same, different):
        module.setup()
    order_a = list(iter(first.train_dataloader().sampler))
    order_b = list(iter(same.train_dataloader().sampler))
    order_c = list(iter(different.train_dataloader().sampler))
    assert order_a == order_b
    assert order_a != order_c


def test_epoch_training_metrics_csv_artifact(tmp_path):
    import csv

    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger

    from aresseg.train.lit import SegDataModule, SegLitModule

    records = _make_records(tmp_path / "records", n=4, size=32)
    module = SegLitModule(model_name="tiny_unet", max_epochs=1, pretrained=False)
    data = SegDataModule(
        records[:3],
        records[3:],
        batch_size=2,
        size=32,
        aug={"hflip_p": 0.0, "rbc_p": 0.0},
        seed=29,
    )
    logger = CSVLogger(save_dir=str(tmp_path), name="logs", version="")
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, data)
    logger.save()
    source = Path(logger.log_dir) / "metrics.csv"
    artifact = tmp_path / "training_metrics.csv"
    artifact.write_bytes(source.read_bytes())
    with artifact.open(newline="", encoding="utf-8") as stream:
        fields = set(csv.DictReader(stream).fieldnames or [])
    expected = {
        "epoch",
        "step",
        "lr",
        "train_loss",
        "train_miou",
        "val_loss",
        "val_miou",
        *(f"train_iou_{name}" for name in ("soil", "bedrock", "sand", "big_rock")),
        *(f"val_iou_{name}" for name in ("soil", "bedrock", "sand", "big_rock")),
    }
    assert expected <= fields


def test_result_store_upsert_is_idempotent(tmp_path):
    from aresseg.utils.results import append_results, read_results

    parquet = tmp_path / "results.parquet"
    csv_path = tmp_path / "results.csv"
    row = {
        "run_id": "run",
        "model": "tiny_unet",
        "backbone": "none",
        "variant": "scratch",
        "scope": "ALL",
        "stratum": "all",
        "metric": "miou",
        "value": 0.1,
        "status": "ok",
        "profile": "gpu_full",
        "seed": 1414,
        "git_sha": "abc",
        "config_hash": "cfg",
        "total_parameters": 10,
        "trainable_parameters": 10,
    }
    append_results([row], parquet, csv_path)
    append_results([{**row, "value": 0.2}], parquet, csv_path)
    frame = read_results(parquet)
    assert len(frame) == 1
    assert frame.iloc[0]["value"] == pytest.approx(0.2)
    assert csv_path.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_h4_resume_requires_matching_source_checkpoint(monkeypatch, tmp_path):
    import json

    from scripts import run_experiment as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    experiments = tmp_path / "experiments"
    run_dir = experiments / "h4_run"
    run_dir.mkdir(parents=True)
    config = {"model": {"name": "unet"}}
    config_hash = runner.config_hash(config)
    checkpoint = str(tmp_path / "source.ckpt")
    manifest = {
        "run_id": run_dir.name,
        "timestamp_utc": "2026-07-10T00:00:00Z",
        "config_hash": config_hash,
        "git_sha": "git",
        "code_fingerprint": "code",
        "profile": "gpu_full",
        "model": "unet",
        "evaluation_split": "gold",
        "stages_completed": ["load_checkpoint", "eval_test_msl", "eval_test_mer"],
        "source_checkpoint": checkpoint,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_image.parquet").write_bytes(b"present")
    (run_dir / "per_image.csv").write_text("present\n", encoding="utf-8")
    experiments.mkdir(exist_ok=True)
    (experiments / "results_store.csv").write_text(
        f"run_id,config_hash,status\n{run_dir.name},{config_hash},ok\n",
        encoding="utf-8",
    )
    common = {
        "profile": "gpu_full",
        "model_name": "unet",
        "git_sha": "git",
        "code_fingerprint": "code",
        "h4": True,
    }
    assert runner._matching_complete_run(config, source_checkpoint=checkpoint, **common) == run_dir
    assert (
        runner._matching_complete_run(
            config, source_checkpoint=str(tmp_path / "other.ckpt"), **common
        )
        is None
    )

    tied = experiments / "h4_tied"
    tied.mkdir()
    tied_manifest = {**manifest, "run_id": tied.name}
    (tied / "manifest.json").write_text(json.dumps(tied_manifest), encoding="utf-8")
    (tied / "per_image.parquet").write_bytes(b"present")
    (tied / "per_image.csv").write_text("present\n", encoding="utf-8")
    (experiments / "results_store.csv").write_text(
        "run_id,config_hash,status\n"
        f"{run_dir.name},{config_hash},ok\n"
        f"{tied.name},{config_hash},ok\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="timestamp tie"):
        runner._matching_complete_run(config, source_checkpoint=checkpoint, **common)


def test_training_resume_allows_only_pinned_protocol_v3_provenance(monkeypatch, tmp_path):
    import json

    from scripts import run_experiment as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    experiments = tmp_path / "experiments"
    run_dir = experiments / "legacy_training"
    run_dir.mkdir(parents=True)
    config = {"model": {"name": "unet"}}
    config_hash = runner.config_hash(config)
    legacy_sha = next(iter(runner.PROTOCOL_V3_TRAINING_GIT_SHAS))
    manifest = {
        "run_id": run_dir.name,
        "timestamp_utc": "2026-07-10T00:00:00Z",
        "config_hash": config_hash,
        "git_sha": legacy_sha,
        "code_fingerprint": runner.PROTOCOL_V3_TRAINING_CODE_FINGERPRINT,
        "profile": "gpu_full",
        "model": "unet",
        "evaluation_split": "gold",
        "stages_completed": ["train", "eval_test_msl"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "per_image.parquet").write_bytes(b"present")
    (run_dir / "per_image.csv").write_text("present\n", encoding="utf-8")
    (run_dir / "best.ckpt").write_bytes(b"present")
    (run_dir / "training_metrics.csv").write_text("present\n", encoding="utf-8")
    (experiments / "results_store.csv").write_text(
        f"run_id,config_hash,status\n{run_dir.name},{config_hash},ok\n", encoding="utf-8"
    )
    common = {
        "profile": "gpu_full",
        "model_name": "unet",
        "git_sha": "current",
        "code_fingerprint": "current-code",
    }
    assert runner._matching_complete_run(config, **common) is None
    assert (
        runner._matching_complete_run(config, allow_protocol_v3_training=True, **common) == run_dir
    )
    manifest["code_fingerprint"] = "unrecognized"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runner._matching_complete_run(config, allow_protocol_v3_training=True, **common) is None


# --------------------------------------------------------------------------------------


class _RecLogger:
    """Captures log.warning calls (aresseg loggers set propagate=False, so caplog can't)."""

    def __init__(self):
        self.msgs = []

    def warning(self, msg, *args):
        self.msgs.append(msg % args if args else msg)


def test_foundation_skips_and_logs_when_gated(monkeypatch):
    """Gate B skip arm, deterministic on ANY box (cuda gate forced off): None + reason logged.

    No skipif needed and no network possible — the gate trips before any load path.
    """
    from aresseg.models import foundation

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
    from aresseg.models import foundation

    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    sentinel = torch.nn.Identity()
    seen = {}

    def fake_load(num_classes):
        seen["num_classes"] = num_classes
        return sentinel

    monkeypatch.setattr(foundation, "_load_dinov3", fake_load)
    foundation._LAST_UNAVAILABLE_REASON["dinov3_sat"] = "stale"
    assert build_model("dinov3_sat", num_classes=4) is sentinel
    assert foundation.last_unavailable_reason("dinov3_sat") is None
    assert seen["num_classes"] == 4

    class _StubSam(torch.nn.Module):
        def __init__(self, checkpoint, model_type="vit_b"):
            super().__init__()
            seen["ckpt"] = str(checkpoint)

    ckpt = tmp_path / "sam_vit_b.pth"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(foundation, "_sam_importable", lambda: True)
    monkeypatch.setattr(foundation, "SamRegionOracleUpperBound", _StubSam)
    foundation._LAST_UNAVAILABLE_REASON["sam"] = "stale"
    out = build_model("sam", sam_checkpoint=str(ckpt))
    assert foundation.last_unavailable_reason("sam") is None
    assert isinstance(out, _StubSam)
    assert seen["ckpt"] == str(ckpt)  # model.sam_checkpoint reaches the arm via the registry


def test_foundation_load_failure_skips_not_raises(monkeypatch):
    """Gates pass but the load blows up (e.g. HF 403 pending license) => skip-and-log, no crash."""
    from aresseg.models import foundation

    rec = _RecLogger()
    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(foundation, "log", rec)

    def boom(num_classes):
        raise foundation.FoundationUnavailable(
            "403 Forbidden: gated repo, license approval pending"
        )

    monkeypatch.setattr(foundation, "_load_dinov3", boom)
    assert build_model("dinov3_sat") is None
    assert "403 Forbidden" in foundation.last_unavailable_reason("dinov3_sat")
    assert any("load failed" in m and "403" in m for m in rec.msgs)


def test_foundation_programming_errors_propagate(monkeypatch):
    from aresseg.models import foundation

    monkeypatch.setattr(foundation, "has_cuda", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")

    def bug(_num_classes):
        raise TypeError("decoder contract bug")

    monkeypatch.setattr(foundation, "_load_dinov3", bug)
    with pytest.raises(TypeError, match="decoder contract bug"):
        build_model("dinov3_sat")


def test_foundation_gating_reasons(monkeypatch, tmp_path):
    from aresseg.models import foundation

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

    from aresseg.models.foundation import DinoV3SatSegmenter

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

    from aresseg.models.foundation import DinoV3SatSegmenter

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
