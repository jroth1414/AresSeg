"""Structured outputs for the Mars perspective routing experiment."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PerspectiveOutput:
    """Auxiliary tensors produced by :class:`MarsPerspectiveScaleAdapter`.

    All logits are at the input image resolution. ``routing_weights`` has shape
    ``(B, 4, H/4, W/4)`` and ``projected_embedding`` is the routed 128-channel
    embedding at the same spatial resolution. ``predicted_cutoff`` is an image-level
    fraction in ``[0, 1]`` with shape ``(B,)``.
    """

    final_logits: Tensor
    base_logits: Tensor
    routing_weights: Tensor
    projected_embedding: Tensor
    predicted_cutoff: Tensor
