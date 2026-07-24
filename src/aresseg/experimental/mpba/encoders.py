"""Native-decoder encoder adapters for the MPBA experiment.

The adapters expose the four feature scales used by MPBA while computing the native
decoder logits in the same pass. They intentionally use duck typing so importing the
experimental package does not eagerly import optional SMP or Transformers dependencies.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class EncoderAdapterOutput:
    """Native logits and encoder features ordered from ``/4`` through ``/32``."""

    base_logits: Tensor
    features: tuple[Tensor, Tensor, Tensor, Tensor]


class EncoderAdapter(nn.Module, ABC):
    """Interface shared by native architecture adapters."""

    feature_strides: ClassVar[tuple[int, int, int, int]] = (4, 8, 16, 32)

    def __init__(self, feature_channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        if len(feature_channels) != len(self.feature_strides):
            raise ValueError("MPBA requires exactly four encoder feature scales")
        if any(channel <= 0 for channel in feature_channels):
            raise ValueError("encoder feature channels must all be positive")
        self.feature_channels = tuple(int(channel) for channel in feature_channels)

    @abstractmethod
    def forward(self, images: Tensor) -> EncoderAdapterOutput:
        """Return native decoder logits and ``/4``, ``/8``, ``/16``, ``/32`` features."""


def _four_features(features: object, architecture: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(features, (tuple, list)) or len(features) < 4:
        raise RuntimeError(f"{architecture} encoder did not return a four-stage feature pyramid")
    selected = tuple(features[-4:])
    if not all(isinstance(feature, Tensor) and feature.ndim == 4 for feature in selected):
        raise RuntimeError(f"{architecture} encoder features must be BCHW tensors")
    return selected  # type: ignore[return-value]


class SMPEncoderAdapter(EncoderAdapter):
    """Expose SMP U-Net encoder stages without replacing its native decoder."""

    def __init__(self, model: nn.Module) -> None:
        missing = [
            name for name in ("encoder", "decoder", "segmentation_head") if not hasattr(model, name)
        ]
        if missing:
            raise TypeError(f"SMP model is missing required attributes: {', '.join(missing)}")

        out_channels = getattr(model.encoder, "out_channels", None)
        if not isinstance(out_channels, (tuple, list)) or len(out_channels) < 4:
            raise TypeError("SMP encoder must expose at least four entries in out_channels")
        super().__init__(tuple(int(channel) for channel in out_channels[-4:]))
        self.model = model

        # SMP <=0.3 used ``decoder(*features)``; SMP >=0.4 uses ``decoder(features)``.
        # Resolve the convention once rather than catching execution-time TypeErrors.
        parameters = inspect.signature(model.decoder.forward).parameters.values()
        self._decoder_uses_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )

    def forward(self, images: Tensor) -> EncoderAdapterOutput:
        check_input_shape = getattr(self.model, "check_input_shape", None)
        if callable(check_input_shape):
            check_input_shape(images)

        all_features = self.model.encoder(images)
        if self._decoder_uses_varargs:
            decoded = self.model.decoder(*all_features)
        else:
            decoded = self.model.decoder(all_features)
        base_logits = self.model.segmentation_head(decoded)
        features = _four_features(all_features, "SMP")
        return EncoderAdapterOutput(base_logits=base_logits, features=features)


def _unwrap_segformer(model: nn.Module) -> nn.Module:
    """Return a Hugging Face ``SegformerForSemanticSegmentation``-like module."""

    candidate = getattr(model, "model", model)
    if not hasattr(candidate, "segformer") or not hasattr(candidate, "decode_head"):
        raise TypeError(
            "SegFormer adapter expects the aresseg wrapper or a "
            "SegformerForSemanticSegmentation-like module"
        )
    return candidate


class SegFormerEncoderAdapter(EncoderAdapter):
    """Expose MiT-B0 stages while retaining the native SegFormer decode head."""

    def __init__(self, model: nn.Module) -> None:
        hf_model = _unwrap_segformer(model)
        config = getattr(hf_model, "config", None)
        hidden_sizes = getattr(config, "hidden_sizes", None)
        if not isinstance(hidden_sizes, (tuple, list)) or len(hidden_sizes) != 4:
            raise TypeError("SegFormer config must expose four hidden_sizes")
        super().__init__(tuple(int(channel) for channel in hidden_sizes))
        self.model = model
        # Do not register the same module twice. It remains reachable through ``self.model``.
        self.__dict__["_hf_model"] = hf_model

    @property
    def hf_model(self) -> nn.Module:
        return self.__dict__["_hf_model"]

    def forward(self, images: Tensor) -> EncoderAdapterOutput:
        outputs = self.hf_model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        features = _four_features(hidden_states, "SegFormer")
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, Tensor):
            raise RuntimeError("SegFormer native decoder did not return tensor logits")
        base_logits = F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return EncoderAdapterOutput(base_logits=base_logits, features=features)


def build_encoder_adapter(model: nn.Module, architecture: str | None = None) -> EncoderAdapter:
    """Build the appropriate native decoder adapter for an SMP U-Net or SegFormer-B0."""

    hint = architecture.lower() if architecture is not None else None
    if hint in {"unet", "smp", "smp_unet"}:
        return SMPEncoderAdapter(model)
    if hint in {"segformer", "segformer_b0", "mit-b0", "b0"}:
        return SegFormerEncoderAdapter(model)
    if hint is not None:
        raise ValueError(f"unsupported MPBA architecture {architecture!r}")

    if all(hasattr(model, name) for name in ("encoder", "decoder", "segmentation_head")):
        return SMPEncoderAdapter(model)
    candidate = getattr(model, "model", model)
    if hasattr(candidate, "segformer") and hasattr(candidate, "decode_head"):
        return SegFormerEncoderAdapter(model)
    raise TypeError("could not infer an SMP U-Net or SegFormer encoder adapter")
