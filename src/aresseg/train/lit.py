"""PyTorch Lightning orchestration for segmentation (MS2).

``SegLitModule`` wraps any model from the zoo with the CE+Dice loss and torchmetrics IoU
(``MulticlassJaccardIndex`` with ``ignore_index``); ``SegDataModule`` wraps ``SegDataset`` into
deterministic DataLoaders. Lightning handles the loop, AMP, gradient clipping, checkpointing,
early stopping, and DDP on the V100; metrics (mIoU + per-class IoU) are logged each epoch.
"""

from __future__ import annotations

import random

import lightning as L
import numpy as np
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from ..data.ai4mars import CLASSES
from ..models.zoo import build_model
from .loss import CombinedLoss


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python/NumPy in each worker from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class SegLitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "unet",
        num_classes: int = 4,
        backbone: str | None = None,
        pretrained: bool = True,
        revision: str | None = None,
        class_weights=None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(
            model_name,
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            revision=revision,
        )
        self.loss_fn = CombinedLoss(class_weights, ignore_index, dice_weight)
        mk = dict(num_classes=num_classes, ignore_index=ignore_index)
        self.train_iou_per_class = MulticlassJaccardIndex(average=None, **mk)
        self.train_miou = MulticlassJaccardIndex(average="macro", **mk)
        self.val_iou_per_class = MulticlassJaccardIndex(average=None, **mk)
        self.val_miou = MulticlassJaccardIndex(average="macro", **mk)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])
        preds = logits.argmax(1)
        self.train_iou_per_class.update(preds, batch["mask"])
        self.train_miou.update(preds, batch["mask"])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["image"].size(0),
        )
        return loss

    def on_train_epoch_end(self):
        per_class = self.train_iou_per_class.compute()
        self.log("train_miou", self.train_miou.compute(), prog_bar=True)
        for i, name in enumerate(CLASSES):
            self.log(f"train_iou_{name}", per_class[i])
        if self.trainer.optimizers:
            self.log("lr", float(self.trainer.optimizers[0].param_groups[0]["lr"]))
        self.train_iou_per_class.reset()
        self.train_miou.reset()

    def validation_step(self, batch, _):
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])
        preds = logits.argmax(1)
        self.val_iou_per_class.update(preds, batch["mask"])
        self.val_miou.update(preds, batch["mask"])
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["image"].size(0),
        )

    def on_validation_epoch_end(self):
        per_class = self.val_iou_per_class.compute()
        self.log("val_miou", self.val_miou.compute(), prog_bar=True)
        for i, name in enumerate(CLASSES):
            self.log(f"val_iou_{name}", per_class[i])
        self.val_iou_per_class.reset()
        self.val_miou.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


class SegDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_records: list[dict],
        val_records: list[dict],
        test_records: list[dict] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        size: int = 512,
        aug: dict | None = None,
        seed: int = 1414,
    ):
        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.aug = dict(aug or {})
        self.seed = int(seed)

    def setup(self, stage: str | None = None):
        from ..data.dataset import SegDataset
        from ..data.transforms import eval_transform, train_transform

        self.train_ds = SegDataset(self.train_records, train_transform(self.size, self.aug))
        self.val_ds = SegDataset(self.val_records, eval_transform(self.size))
        self.test_ds = (
            SegDataset(self.test_records, eval_transform(self.size)) if self.test_records else None
        )

    def _loader(self, ds, shuffle, seed_offset=0):
        from torch.utils.data import DataLoader

        if ds is None:
            return None
        generator = torch.Generator().manual_seed(self.seed + int(seed_offset))
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=seed_dataloader_worker,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, True, seed_offset=0)

    def val_dataloader(self):
        return self._loader(self.val_ds, False, seed_offset=1)

    def test_dataloader(self):
        return self._loader(self.test_ds, False, seed_offset=2)
