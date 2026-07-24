"""Lightning training support for the isolated MPBA validation experiments.

This module intentionally does not modify the Protocol V3 training module.  It keeps the
segmentation objective identical (class-weighted CE + Dice) and adds only the preregistered
``0.1 * SmoothL1`` auxiliary loss when the learned range-cutoff coordinate is enabled.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

from marsseg.data.ai4mars import CLASSES
from marsseg.train.loss import CombinedLoss


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def output_field(output: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a dataclass-like or mapping auxiliary model output.

    Keeping this adapter small makes the training layer tolerant of serialized checkpoints made
    while the experimental output container evolves, without weakening the tensor-only public
    ``forward`` contract.
    """
    if isinstance(output, torch.Tensor):
        return (
            output
            if any(name in {"logits", "final_logits", "base_logits"} for name in names)
            else default
        )
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
        return default
    for name in names:
        if hasattr(output, name):
            return getattr(output, name)
    return default


def final_logits(output: Any) -> torch.Tensor:
    """Extract final segmentation logits from a model auxiliary output."""
    logits = output_field(output, "final_logits", "logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("forward_with_aux must expose tensor final_logits (or logits)")
    return logits


def _cutoff_target(batch: Mapping[str, Any]) -> torch.Tensor:
    for key in ("cutoff_target", "range_cutoff_target", "range_cutoff"):
        value = batch.get(key)
        if value is not None:
            return torch.as_tensor(value)
    raise KeyError("range_cutoff training requires a scalar cutoff_target in every dataset sample")


def cutoff_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SmoothL1 loss and MAE over finite per-image cutoff targets."""
    prediction = prediction.float().reshape(-1)
    target = target.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
    if prediction.numel() != target.numel():
        raise ValueError(
            "predicted cutoff and target batch sizes differ: "
            f"{prediction.numel()} != {target.numel()}"
        )
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("range_cutoff batch contains no finite cutoff targets")
    prediction, target = prediction[valid], target[valid]
    return F.smooth_l1_loss(prediction, target), F.l1_loss(prediction, target)


def routing_scale_means(weights: torch.Tensor, expected_scales: int = 4) -> torch.Tensor:
    """Reduce a routing tensor to one mean utilization value per scale."""
    if weights.ndim < 2:
        raise ValueError(f"routing weights must have at least two dimensions, got {weights.shape}")
    [axis for axis, size in enumerate(weights.shape) if size == expected_scales]
    # The model contract is B,S,H,W.  The final-axis fallback tolerates B,H,W,S diagnostics.
    scale_axis = 1 if weights.ndim > 1 and weights.shape[1] == expected_scales else None
    if scale_axis is None and weights.shape[-1] == expected_scales:
        scale_axis = weights.ndim - 1
    if scale_axis is None:
        raise ValueError(
            f"routing weights do not contain the expected {expected_scales} scales: {weights.shape}"
        )
    dims = tuple(axis for axis in range(weights.ndim) if axis != scale_axis)
    means = weights.float().mean(dim=dims)
    if means.numel() != expected_scales or not torch.isfinite(means).all():
        raise ValueError("routing utilization must contain four finite scale means")
    return means


class MPBALitModule(L.LightningModule):
    """Train a native decoder or its parameter-matched MPBA residual adapter."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_classes: int = 4,
        coordinate_mode: str = "none",
        class_weights: list[float] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dice_weight: float = 1.0,
        cutoff_loss_weight: float = 0.1,
        ignore_index: int = 255,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        if coordinate_mode not in {"none", "raw_y", "range_cutoff"}:
            raise ValueError(f"unsupported coordinate_mode {coordinate_mode!r}")
        if cutoff_loss_weight < 0:
            raise ValueError("cutoff_loss_weight must be non-negative")
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.coordinate_mode = coordinate_mode
        self.loss_fn = CombinedLoss(class_weights, ignore_index, dice_weight)
        metric_kwargs = dict(num_classes=num_classes, ignore_index=ignore_index)
        self.train_iou_per_class = MulticlassJaccardIndex(average=None, **metric_kwargs)
        self.train_miou = MulticlassJaccardIndex(average="macro", **metric_kwargs)
        self.val_iou_per_class = MulticlassJaccardIndex(average=None, **metric_kwargs)
        self.val_miou = MulticlassJaccardIndex(average="macro", **metric_kwargs)

    @property
    def uses_range_cutoff(self) -> bool:
        return self.coordinate_mode == "range_cutoff"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve the repository-wide tensor-only, input-resolution model contract."""
        return self.model(image)

    def forward_with_aux(self, image: torch.Tensor) -> Any:
        """Expose MPBA diagnostics while accepting a native model as the control arm."""
        method = getattr(self.model, "forward_with_aux", None)
        return method(image) if callable(method) else self.model(image)

    def _losses(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        output = self.forward_with_aux(batch["image"])
        logits = final_logits(output)
        segmentation_loss = self.loss_fn(logits, batch["mask"])
        cutoff_loss = None
        if self.uses_range_cutoff:
            prediction = output_field(output, "predicted_cutoff", "cutoff")
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("range_cutoff model output must expose tensor predicted_cutoff")
            cutoff_loss, _ = cutoff_smooth_l1(prediction, _cutoff_target(batch))
        total = segmentation_loss
        if cutoff_loss is not None:
            total = total + float(self.hparams.cutoff_loss_weight) * cutoff_loss
        return output, logits, total, cutoff_loss

    def training_step(self, batch: Mapping[str, Any], _batch_idx: int) -> torch.Tensor:
        _output, logits, loss, cutoff_loss = self._losses(batch)
        predictions = logits.argmax(1)
        self.train_iou_per_class.update(predictions, batch["mask"])
        self.train_miou.update(predictions, batch["mask"])
        batch_size = batch["image"].shape[0]
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if cutoff_loss is not None:
            self.log(
                "train_cutoff_loss",
                cutoff_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        per_class = self.train_iou_per_class.compute()
        self.log("train_miou", self.train_miou.compute(), prog_bar=True)
        for index, name in enumerate(CLASSES):
            self.log(f"train_iou_{name}", per_class[index])
        if self.trainer.optimizers:
            self.log("lr", float(self.trainer.optimizers[0].param_groups[0]["lr"]))
        self.train_iou_per_class.reset()
        self.train_miou.reset()

    def validation_step(self, batch: Mapping[str, Any], _batch_idx: int) -> None:
        output, logits, loss, cutoff_loss = self._losses(batch)
        predictions = logits.argmax(1)
        self.val_iou_per_class.update(predictions, batch["mask"])
        self.val_miou.update(predictions, batch["mask"])
        batch_size = batch["image"].shape[0]
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        if cutoff_loss is not None:
            prediction = output_field(output, "predicted_cutoff", "cutoff")
            _, cutoff_mae = cutoff_smooth_l1(prediction, _cutoff_target(batch))
            self.log(
                "val_cutoff_loss",
                cutoff_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val_cutoff_mae",
                cutoff_mae,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        weights = output_field(output, "routing_weights", "route_weights")
        if isinstance(weights, torch.Tensor):
            for index, utilization in enumerate(routing_scale_means(weights)):
                self.log(
                    f"val_routing_scale_{index}",
                    utilization,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

    def on_validation_epoch_end(self) -> None:
        per_class = self.val_iou_per_class.compute()
        self.log("val_miou", self.val_miou.compute(), prog_bar=True, sync_dist=True)
        for index, name in enumerate(CLASSES):
            self.log(f"val_iou_{name}", per_class[index], sync_dist=True)
        self.val_iou_per_class.reset()
        self.val_miou.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams.max_epochs),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class MPBADataModule(L.LightningDataModule):
    """Validation-only-screening data module with optional range-cutoff supervision."""

    def __init__(
        self,
        train_records: list[dict],
        val_records: list[dict],
        *,
        batch_size: int = 8,
        num_workers: int = 0,
        size: int = 512,
        aug: dict | None = None,
        seed: int = 1414,
        use_range_cutoff: bool = False,
    ) -> None:
        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.size = int(size)
        self.aug = dict(aug or {})
        self.seed = int(seed)
        self.use_range_cutoff = bool(use_range_cutoff)

    def setup(self, stage: str | None = None) -> None:
        from marsseg.data.dataset import SegDataset
        from marsseg.data.transforms import eval_transform, train_transform

        dataset_type = SegDataset
        if self.use_range_cutoff:
            from marsseg.experimental.mpba.data import RangeAwareSegDataset

            dataset_type = RangeAwareSegDataset
        self.train_ds = dataset_type(
            self.train_records,
            train_transform(self.size, self.aug),
        )
        self.val_ds = dataset_type(self.val_records, eval_transform(self.size))

    def _loader(self, dataset: Any, *, shuffle: bool, seed_offset: int) -> Any:
        from torch.utils.data import DataLoader

        generator = torch.Generator().manual_seed(self.seed + int(seed_offset))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=seed_dataloader_worker,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> Any:
        return self._loader(self.train_ds, shuffle=True, seed_offset=0)

    def val_dataloader(self) -> Any:
        return self._loader(self.val_ds, shuffle=False, seed_offset=1)
