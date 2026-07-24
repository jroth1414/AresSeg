"""Albumentations transforms for AI4Mars segmentation (Phase MS1).

Geometric ops apply nearest-neighbour to the mask automatically (labels are not interpolated).
Augmentation preserves the horizon (horizontal flip + photometric only — NO vertical flip).
Grayscale images are replicated to 3 channels and ImageNet-normalized for pretrained encoders.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


DEFAULT_AUG = {
    "hflip_p": 0.5,
    "vflip": False,
    "vflip_p": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "rbc_p": 0.3,
    "scale_crop": False,
    "scale_limit": 0.1,
    "scale_crop_p": 0.3,
}


def train_transform(size: int = 512, aug: dict | None = None) -> A.Compose:
    """Build training transforms from the resolved augmentation configuration.

    Defaults preserve the preregistered behavior. Passing the resolved mapping makes changes to
    configs/data.yaml affect execution as well as the config hash.
    """
    acfg = {**DEFAULT_AUG, **(aug or {})}
    transforms: list = [A.Resize(size, size)]
    if acfg.get("scale_crop"):
        # Disabled by the frozen protocol, but implemented so this advertised knob is truthful.
        transforms.append(
            A.Affine(
                scale=(1.0 - float(acfg["scale_limit"]), 1.0 + float(acfg["scale_limit"])),
                translate_percent=0,
                rotate=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=float(acfg["scale_crop_p"]),
            )
        )
    transforms.append(A.HorizontalFlip(p=float(acfg["hflip_p"])))
    if acfg.get("vflip"):
        transforms.append(A.VerticalFlip(p=float(acfg["vflip_p"])))
    transforms.extend(
        [
            A.RandomBrightnessContrast(
                brightness_limit=float(acfg["brightness_limit"]),
                contrast_limit=float(acfg["contrast_limit"]),
                p=float(acfg["rbc_p"]),
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def eval_transform(size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
