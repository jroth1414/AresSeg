"""Offline tests for the additive perspective-first MPBA experiment."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import pytest
import torch

from marsseg.experimental.mpba.data import (
    MATCHED_COHORT_EXCLUDED_PRODUCTS,
    RangeAwareSegDataset,
    apply_matched_range_cohort,
    attach_range_masks,
    cutoff_target_from_mask,
    resolve_range_mask_path,
)
from marsseg.experimental.mpba.lit import MPBALitModule, cutoff_smooth_l1
from marsseg.experimental.mpba.metrics import (
    MPBAValidationAccumulator,
    routing_utilization,
    small_big_rock_component_counts,
)
from marsseg.experimental.mpba.model import MarsPerspectiveScaleAdapter
from marsseg.models.zoo import build_model


def _mpba(
    architecture: str,
    *,
    router_mode: str = "content",
    coordinate_mode: str = "none",
) -> MarsPerspectiveScaleAdapter:
    backbone = "resnet34" if architecture == "unet" else "b0"
    base = build_model(
        architecture,
        num_classes=4,
        backbone=backbone,
        pretrained=False,
    )
    assert base is not None
    return MarsPerspectiveScaleAdapter(
        base,
        architecture=architecture,
        num_classes=4,
        projection_channels=128,
        router_mode=router_mode,
        coordinate_mode=coordinate_mode,
    )


@pytest.mark.parametrize(
    ("architecture", "channels"),
    [
        ("unet", (64, 128, 256, 512)),
        ("segformer", (32, 64, 160, 256)),
    ],
)
def test_real_encoder_adapters_and_forward_contract(architecture, channels):
    model = _mpba(architecture).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model.forward_with_aux(image)
        native = model.encoder_adapter(image)

    assert model.encoder_adapter.feature_channels == channels
    assert [feature.shape[-2:] for feature in native.features] == [
        (16, 16),
        (8, 8),
        (4, 4),
        (2, 2),
    ]
    assert output.final_logits.shape == output.base_logits.shape == (1, 4, 64, 64)
    assert output.routing_weights.shape == (1, 4, 16, 16)
    assert output.projected_embedding.shape == (1, 128, 16, 16)
    assert output.predicted_cutoff.shape == (1,)
    assert torch.isfinite(output.final_logits).all()
    assert torch.allclose(output.routing_weights.sum(1), torch.ones(1, 16, 16))
    # The zero-initialized adapter starts as the unchanged native decoder.
    assert torch.allclose(output.final_logits, output.base_logits)
    with torch.no_grad():
        assert torch.equal(model(image), output.final_logits)

    model.train()
    model.forward_with_aux(image).final_logits.mean().backward()
    assert model.residual_head[-1].weight.grad is not None
    assert torch.isfinite(model.residual_head[-1].weight.grad).all()


def test_static_routing_is_spatially_constant_and_parameter_matched():
    static = _mpba("unet", router_mode="static", coordinate_mode="none").eval()
    content = _mpba("unet", router_mode="content", coordinate_mode="raw_y").eval()
    assert sum(parameter.numel() for parameter in static.parameters()) == sum(
        parameter.numel() for parameter in content.parameters()
    )

    weights = static.forward_with_aux(torch.randn(2, 3, 64, 64)).routing_weights
    reference = weights[:, :, :1, :1].expand_as(weights)
    assert torch.allclose(weights, reference)


def test_routing_softmax_is_float32_under_reduced_precision():
    model = _mpba("unet").eval()
    projected = torch.randn(1, 4, 128, 5, 7)
    coordinate = torch.zeros(1, 1, 5, 7)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        weights = model._routing_weights(projected, coordinate)
    assert weights.dtype == torch.float32
    assert torch.allclose(weights.sum(1), torch.ones(1, 5, 7), atol=1e-6)


def test_coordinate_modes_have_expected_geometry():
    reference = torch.zeros(2, 128, 5, 3)
    cutoff = torch.tensor([0.0, 0.5])

    none = _mpba("unet", coordinate_mode="none")._coordinate(reference, cutoff)
    raw = _mpba("unet", coordinate_mode="raw_y")._coordinate(reference, cutoff)
    relative = _mpba("unet", coordinate_mode="range_cutoff")._coordinate(reference, cutoff)

    assert torch.count_nonzero(none) == 0
    assert torch.allclose(raw[0, 0, :, 0], torch.arange(5) / 5)
    assert torch.allclose(relative[0], raw[0])
    assert torch.allclose(relative[1, 0, :, 0], torch.tensor([-1.0, -0.6, -0.2, 0.2, 0.6]))


def test_checkpoint_round_trip_preserves_auxiliary_outputs():
    first = _mpba("unet", coordinate_mode="range_cutoff").eval()
    second = _mpba("unet", coordinate_mode="range_cutoff").eval()
    second.load_state_dict(first.state_dict())
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = first.forward_with_aux(image)
        b = second.forward_with_aux(image)
    assert torch.equal(a.final_logits, b.final_logits)
    assert torch.equal(a.routing_weights, b.routing_weights)
    assert torch.equal(a.predicted_cutoff, b.predicted_cutoff)


@pytest.mark.parametrize("architecture", ["unet", "segformer"])
def test_cpu_lightning_fast_dev_for_both_families(architecture):
    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            return {
                "image": torch.randn(3, 32, 32, generator=generator),
                "mask": torch.randint(0, 4, (32, 32), generator=generator),
            }

    module = MPBALitModule(
        _mpba(architecture),
        coordinate_mode="none",
        class_weights=[1.0] * 4,
        max_epochs=1,
    )
    loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=2)
    trainer = L.Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.state.finished


def _write_range_fixture(root: Path) -> dict[str, str]:
    image_dir = root / "images" / "edr"
    range_dir = root / "images" / "rng-30m"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    range_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[2:6, 2:6] = 3
    range_mask = np.zeros((8, 8), dtype=np.uint8)
    range_mask[:2] = 1
    image_path = image_dir / "NLA_1EDR_F000.JPG"
    label_path = label_dir / "NLA_1EDR_F000.png"
    range_path = range_dir / "NLA_1RNG_F000.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(label_path), semantic)
    assert cv2.imwrite(str(range_path), range_mask)
    return {
        "image": str(image_path),
        "label": str(label_path),
        "name": "fixture",
    }


def test_range_mask_resolution_targets_and_dataset(tmp_path):
    record = _write_range_fixture(tmp_path)
    resolved = resolve_range_mask_path(record)
    assert resolved is not None and resolved.name == "NLA_1RNG_F000.png"
    augmented = attach_range_masks([record])
    assert "range_mask" not in record
    assert augmented[0]["range_mask"] == str(resolved)

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:2] = 1
    assert cutoff_target_from_mask(mask) == pytest.approx(0.25)
    assert cutoff_target_from_mask(np.zeros_like(mask)) == 0.0
    assert cutoff_target_from_mask(np.ones_like(mask)) == 1.0

    from marsseg.data.transforms import eval_transform

    dataset = RangeAwareSegDataset([record], eval_transform(16), return_range_mask=True)
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == item["range_mask"].shape == (16, 16)
    assert item["cutoff_target"].item() == pytest.approx(0.25)


def test_range_mask_resolution_fails_closed(tmp_path):
    image = tmp_path / "images" / "edr" / "missingEDR_F000.JPG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-an-image")
    with pytest.raises(FileNotFoundError, match="no rng-30m partner"):
        resolve_range_mask_path(image)
    assert resolve_range_mask_path(image, strict=False) is None


def test_matched_range_cohort_excludes_only_pinned_training_product():
    excluded = MATCHED_COHORT_EXCLUDED_PRODUCTS[0]
    splits = {
        "train": [
            {"image": f"/source/edr/{excluded}", "name": "missing"},
            {"image": "/source/edr/kept.JPG", "name": "kept"},
        ],
        "val": [{"image": "/source/edr/validation.JPG", "name": "validation"}],
    }
    filtered, exclusions = apply_matched_range_cohort(splits)
    assert [record["name"] for record in filtered["train"]] == ["kept"]
    assert [record["name"] for record in filtered["val"]] == ["validation"]
    assert exclusions == [
        {
            "image": excluded,
            "record_name": "missing",
            "split": "train",
            "reason": "missing_rng_30m_partner_in_source_snapshot",
        }
    ]

    with pytest.raises(RuntimeError, match="source condition changed"):
        apply_matched_range_cohort({"train": filtered["train"], "val": filtered["val"]})


def test_cutoff_loss_and_mae_ignore_no_values():
    loss, mae = cutoff_smooth_l1(torch.tensor([0.1, 0.5]), torch.tensor([0.2, 0.3]))
    assert loss.item() == pytest.approx((0.5 * 0.1**2 + 0.5 * 0.2**2) / 2)
    assert mae.item() == pytest.approx(0.15)
    with pytest.raises(ValueError, match="batch sizes differ"):
        cutoff_smooth_l1(torch.tensor([0.1]), torch.tensor([0.1, 0.2]))


def test_range_cutoff_lightning_loss_adds_exact_weighted_auxiliary_term():
    module = MPBALitModule(
        _mpba("unet", coordinate_mode="range_cutoff"),
        coordinate_mode="range_cutoff",
        class_weights=[1.0] * 4,
        cutoff_loss_weight=0.1,
    )
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
        "cutoff_target": torch.tensor([0.2, 0.6]),
    }
    _output, logits, total, auxiliary = module._losses(batch)
    assert auxiliary is not None
    segmentation = module.loss_fn(logits, batch["mask"])
    assert torch.allclose(total, segmentation + 0.1 * auxiliary)


def test_small_component_recall_uses_eight_connectivity_and_coverage():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[4:8, 4:8] = 3  # 16 pixels: eligible.
    gt[20:23, 20:25] = 3  # 15 pixels: below the minimum.
    pred = np.zeros_like(gt)
    pred[4:6, 4:8] = 3  # exactly 50% coverage.
    assert small_big_rock_component_counts(pred, gt) == {"eligible": 1, "recalled": 1}
    pred[5, 4] = 0
    assert small_big_rock_component_counts(pred, gt) == {"eligible": 1, "recalled": 0}


def test_routing_utilization_and_validation_accumulator():
    weights = np.full((2, 4, 3, 3), 0.25, dtype=np.float32)
    utilization = routing_utilization(weights)
    assert utilization["n_active_scales"] == 4
    assert utilization["mean_weights"] == pytest.approx([0.25] * 4)

    gt = np.zeros((1, 32, 32), dtype=np.uint8)
    gt[:, 4:8, 4:8] = 3
    accumulator = MPBAValidationAccumulator()
    accumulator.update(
        gt,
        gt,
        routing_weights=weights[:1],
        predicted_cutoff=np.array([0.2]),
        cutoff_target=np.array([0.25]),
    )
    result = accumulator.compute()
    assert result["miou"] == pytest.approx(1.0)
    assert result["boundary_f1"] == pytest.approx(1.0)
    assert result["small_big_rock_component_recall"] == pytest.approx(1.0)
    assert result["cutoff_mae"] == pytest.approx(0.05)
    assert result["routing_utilization"]["n_active_scales"] == 4

    no_optional_metrics = MPBAValidationAccumulator()
    no_optional_metrics.update(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    optional_result = no_optional_metrics.compute()
    assert optional_result["small_big_rock_component_recall"] is None
    assert optional_result["cutoff_mae"] is None
    json.dumps(optional_result, allow_nan=False)
