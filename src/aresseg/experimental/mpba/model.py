"""Perspective-conditioned multi-scale routing for Mars terrain segmentation."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoders import EncoderAdapter, build_encoder_adapter
from .outputs import PerspectiveOutput

RouterMode = Literal["static", "content"]
CoordinateMode = Literal["none", "raw_y", "range_cutoff"]


def _group_count(channels: int) -> int:
    """Choose the largest conventional GroupNorm group count that divides ``channels``."""

    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class MarsPerspectiveScaleAdapter(nn.Module):
    """Add a parameter-matched multi-scale residual adapter to a native segmenter.

    Four encoder stages are projected to ``projection_channels`` at ``/4`` resolution.
    A shared scorer produces either spatially constant learned fusion weights or
    content-dependent pixel weights. The coordinate mode always uses the same scorer
    shape: ``none`` supplies zeros, ``raw_y`` supplies normalized image row, and
    ``range_cutoff`` supplies a row coordinate relative to an image-only cutoff estimate.

    The cutoff predictor and every routing module are instantiated in all modes. This
    makes static/content and coordinate ablations parameter matched for a fixed backbone.
    """

    VALID_ROUTER_MODES = frozenset({"static", "content"})
    VALID_COORDINATE_MODES = frozenset({"none", "raw_y", "range_cutoff"})

    def __init__(
        self,
        base_model: nn.Module | EncoderAdapter,
        *,
        num_classes: int = 4,
        projection_channels: int = 128,
        router_mode: RouterMode = "content",
        coordinate_mode: CoordinateMode = "none",
        architecture: str | None = None,
    ) -> None:
        super().__init__()
        if router_mode not in self.VALID_ROUTER_MODES:
            raise ValueError(f"unsupported router_mode {router_mode!r}")
        if coordinate_mode not in self.VALID_COORDINATE_MODES:
            raise ValueError(f"unsupported coordinate_mode {coordinate_mode!r}")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if projection_channels <= 0:
            raise ValueError("projection_channels must be positive")

        self.encoder_adapter = (
            base_model
            if isinstance(base_model, EncoderAdapter)
            else build_encoder_adapter(base_model, architecture=architecture)
        )
        self.num_classes = int(num_classes)
        self.projection_channels = int(projection_channels)
        self.router_mode = router_mode
        self.coordinate_mode = coordinate_mode

        groups = _group_count(self.projection_channels)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, self.projection_channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, self.projection_channels),
                nn.GELU(),
            )
            for channels in self.encoder_adapter.feature_channels
        )
        self.scale_embeddings = nn.Parameter(torch.empty(4, self.projection_channels))
        router_hidden = max(self.projection_channels // 2, 1)
        self.scorer = nn.Sequential(
            nn.Conv2d(self.projection_channels + 1, router_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(router_hidden, 1, kernel_size=1),
        )

        deepest_channels = self.encoder_adapter.feature_channels[-1]
        self.cutoff_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(deepest_channels, self.projection_channels),
            nn.GELU(),
            nn.Linear(self.projection_channels, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(
                self.projection_channels,
                self.projection_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.projection_channels),
            nn.GELU(),
            nn.Conv2d(self.projection_channels, self.num_classes, kernel_size=1),
        )

        nn.init.normal_(self.scale_embeddings, mean=0.0, std=0.02)
        # Begin as the unchanged native decoder and learn only a residual correction.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _raw_row(reference: Tensor) -> Tensor:
        """Return the requested ``y / H`` row coordinate at ``reference`` resolution."""

        batch, _, height, width = reference.shape
        rows = (
            torch.arange(
                height,
                dtype=reference.dtype,
                device=reference.device,
            )
            / height
        )
        return rows.view(1, 1, height, 1).expand(batch, 1, height, width)

    def _coordinate(self, reference: Tensor, predicted_cutoff: Tensor) -> Tensor:
        rows = self._raw_row(reference)
        if self.coordinate_mode == "none":
            return torch.zeros_like(rows)
        if self.coordinate_mode == "raw_y":
            return rows

        cutoff = predicted_cutoff.to(dtype=reference.dtype).view(-1, 1, 1, 1)
        epsilon = torch.finfo(reference.dtype).eps
        relative = (rows - cutoff) / (1.0 - cutoff).clamp_min(epsilon)
        return relative.clamp(min=-1.0, max=1.0)

    def _project_features(self, features: tuple[Tensor, ...]) -> Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features, strict=True):
            embedding = projection(feature)
            if embedding.shape[-2:] != target_size:
                embedding = F.interpolate(
                    embedding,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(embedding)
        return torch.stack(projected, dim=1)

    def _routing_weights(self, projected: Tensor, coordinate: Tensor) -> Tensor:
        batch, scales, channels, height, width = projected.shape
        scale_embeddings = self.scale_embeddings.view(1, scales, channels, 1, 1)

        if self.router_mode == "static":
            # This path remains spatially and image independent, including when a caller
            # combines it with a non-``none`` coordinate mode.
            routing_features = scale_embeddings.expand(batch, scales, channels, height, width)
            routing_coordinate = torch.zeros(
                (batch, scales, 1, height, width),
                dtype=projected.dtype,
                device=projected.device,
            )
        else:
            routing_features = projected + scale_embeddings
            routing_coordinate = coordinate.unsqueeze(1).expand(-1, scales, -1, -1, -1)

        scorer_input = torch.cat((routing_features, routing_coordinate), dim=2)
        scorer_input = scorer_input.reshape(
            batch * scales,
            channels + 1,
            height,
            width,
        )
        scores = self.scorer(scorer_input).reshape(batch, scales, height, width)
        # Keep the diagnostic probability contract stable even under direct half inference.
        return scores.float().softmax(dim=1)

    def forward_with_aux(self, images: Tensor) -> PerspectiveOutput:
        """Return final logits together with native logits and routing diagnostics."""

        native = self.encoder_adapter(images)
        if native.base_logits.shape[:2] != (images.shape[0], self.num_classes):
            raise RuntimeError(
                "native decoder logits do not match the configured batch size and class count"
            )
        if native.base_logits.shape[-2:] != images.shape[-2:]:
            raise RuntimeError("native decoder must return logits at the input image resolution")

        projected = self._project_features(native.features)
        predicted_cutoff = torch.sigmoid(self.cutoff_head(native.features[-1])).squeeze(1)
        coordinate = self._coordinate(projected[:, 0], predicted_cutoff)
        routing_weights = self._routing_weights(projected, coordinate)
        routed_embedding = (routing_weights.to(dtype=projected.dtype).unsqueeze(2) * projected).sum(
            dim=1
        )

        residual = self.residual_head(routed_embedding)
        residual = F.interpolate(
            residual,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        final_logits = native.base_logits + residual
        return PerspectiveOutput(
            final_logits=final_logits,
            base_logits=native.base_logits,
            routing_weights=routing_weights,
            projected_embedding=routed_embedding,
            predicted_cutoff=predicted_cutoff,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return input-resolution logits, preserving the standard model-zoo contract."""

        return self.forward_with_aux(images).final_logits


def build_mpba_model(
    model_name: str,
    *,
    backbone: str | None = None,
    num_classes: int = 4,
    pretrained: bool = True,
    revision: str | None = None,
    router_mode: RouterMode = "content",
    coordinate_mode: CoordinateMode = "none",
    projection_channels: int = 128,
) -> MarsPerspectiveScaleAdapter:
    """Build an MPBA-wrapped SMP U-Net/ResNet-34 or SegFormer/MiT-B0."""

    normalized_name = model_name.lower()
    if normalized_name not in {"unet", "segformer"}:
        raise ValueError("MPBA currently supports only 'unet' and 'segformer'")
    resolved_backbone = backbone or ("resnet34" if normalized_name == "unet" else "b0")
    if normalized_name == "segformer" and resolved_backbone.lower() != "b0":
        raise ValueError("the perspective-first MPBA cycle is restricted to SegFormer/MiT-B0")

    # Import lazily to keep this experimental module importable without optional model deps.
    from ...models.zoo import build_model

    base_model = build_model(
        normalized_name,
        num_classes=num_classes,
        backbone=resolved_backbone,
        pretrained=pretrained,
        revision=revision,
    )
    if base_model is None:
        raise RuntimeError(f"base model {normalized_name!r} was unavailable")
    return MarsPerspectiveScaleAdapter(
        base_model,
        num_classes=num_classes,
        projection_channels=projection_channels,
        router_mode=router_mode,
        coordinate_mode=coordinate_mode,
        architecture=normalized_name,
    )
