"""Source-only, leaf-first DINOv2 classifier for plant disease recognition.

The model deliberately separates context and leaf evidence while using one shared
backbone instance.  The context image and the leaf-focused view are concatenated
along the batch dimension, so DINOv2 is invoked once per forward pass.

The default leaf view crops the supplied foreground mask (with a small margin),
resizes the leaf to the model input, and neutralises everything outside the
mask.  A pre-cropped/resized leaf view can instead be supplied through
``leaf_image`` (and optionally ``leaf_mask``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["FinalModelV4", "PINNED_DINOV2_HUB_REPO"]

PINNED_DINOV2_HUB_REPO = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)


def _group_count(channels: int, requested_groups: int) -> int:
    """Return the largest valid GroupNorm group count up to the request."""
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DINOv2MultiLayerBackbone(nn.Module):
    """One DINOv2 backbone exposing patch maps and CLS tokens from many layers.

    Parameters
    ----------
    model_name:
        A model exported by the DINOv2 torch hub, for example
        ``dinov2_vits14`` or ``dinov2_vitb14``.  The value is passed to
        ``torch.hub.load`` and is not hard-coded.
    num_intermediate_layers:
        Number of final transformer blocks returned by DINOv2.
    hub_repo:
        Torch hub repository.  Pinning or vendoring this repository is
        recommended for fully reproducible deployments.
    backbone_model:
        Optional already-created DINOv2 module.  This is useful for offline
        construction and tests; when supplied, torch hub is not called.
    freeze:
        Freeze every backbone parameter.  This is the default to preserve the
        broad pretrained representation during source-only training.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        num_intermediate_layers: int = 4,
        hub_repo: str = PINNED_DINOV2_HUB_REPO,
        backbone_model: Optional[nn.Module] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if num_intermediate_layers < 1:
            raise ValueError("num_intermediate_layers must be at least 1")

        self.model_name = model_name
        self.num_intermediate_layers = int(num_intermediate_layers)
        self.dinov2 = (
            backbone_model
            if backbone_model is not None
            else torch.hub.load(hub_repo, model_name)
        )

        if not hasattr(self.dinov2, "embed_dim"):
            raise AttributeError("The supplied backbone does not expose embed_dim")
        if not hasattr(self.dinov2, "get_intermediate_layers"):
            raise AttributeError(
                "The supplied backbone does not expose get_intermediate_layers"
            )

        self.embed_dim = int(self.dinov2.embed_dim)
        self.patch_size = self._read_patch_size(self.dinov2)

        blocks = getattr(self.dinov2, "blocks", None)
        if blocks is not None and self.num_intermediate_layers > len(blocks):
            raise ValueError(
                "num_intermediate_layers cannot exceed the number of backbone "
                f"blocks ({len(blocks)})"
            )

        if freeze:
            self.freeze_all()

    @staticmethod
    def _read_patch_size(backbone: nn.Module) -> Tuple[int, int]:
        patch_embed = getattr(backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            raise AttributeError("The supplied DINOv2 backbone has no patch_size")
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, Sequence) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        raise TypeError(f"Unsupported patch_size: {patch_size!r}")

    def freeze_all(self) -> None:
        """Freeze all DINOv2 parameters."""
        for parameter in self.dinov2.parameters():
            parameter.requires_grad = False
        self.dinov2.eval()

    def unfreeze_last_blocks(
        self,
        num_blocks: int,
        *,
        unfreeze_norm: bool = True,
    ) -> List[nn.Parameter]:
        """Unfreeze the final transformer blocks and return newly trainable params.

        The returned list can be placed in a dedicated optimizer group with a
        lower learning rate.  Passing zero is a no-op.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        blocks = getattr(self.dinov2, "blocks", None)
        if blocks is None:
            raise AttributeError("The supplied backbone does not expose blocks")
        if num_blocks > len(blocks):
            raise ValueError(
                f"Requested {num_blocks} blocks, but backbone has {len(blocks)}"
            )
        if num_blocks == 0:
            return []

        newly_trainable: List[nn.Parameter] = []
        for block in blocks[-num_blocks:]:
            for parameter in block.parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad = True
                    newly_trainable.append(parameter)

        if unfreeze_norm:
            final_norm = getattr(self.dinov2, "norm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    if not parameter.requires_grad:
                        parameter.requires_grad = True
                        newly_trainable.append(parameter)

        if self.training:
            self.dinov2.train()
        return newly_trainable

    def forward(self, image: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Return per-layer patch maps and CLS tokens.

        ``image`` may contain both context and leaf views concatenated in its
        batch dimension.  Spatial dimensions need not be square, but DINOv2's
        patch token count must match the inferred patch grid.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image [B,C,H,W], got {tuple(image.shape)}")

        outputs = self.dinov2.get_intermediate_layers(
            image,
            n=self.num_intermediate_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        if len(outputs) != self.num_intermediate_layers:
            raise RuntimeError(
                f"Expected {self.num_intermediate_layers} intermediate outputs, "
                f"received {len(outputs)}"
            )

        grid_h = image.shape[-2] // self.patch_size[0]
        grid_w = image.shape[-1] // self.patch_size[1]
        expected_tokens = grid_h * grid_w

        patch_maps: List[Tensor] = []
        class_tokens: List[Tensor] = []
        for layer_output in outputs:
            if not isinstance(layer_output, (tuple, list)) or len(layer_output) != 2:
                raise RuntimeError(
                    "DINOv2 must return (patch_tokens, class_token) when "
                    "return_class_token=True"
                )
            patch_tokens, class_token = layer_output
            if patch_tokens.ndim != 3:
                raise RuntimeError(
                    "Expected patch tokens [B,N,C], got "
                    f"{tuple(patch_tokens.shape)}"
                )
            if patch_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "Patch-token count does not match the input grid: "
                    f"got {patch_tokens.shape[1]}, expected {expected_tokens} "
                    f"for patch size {self.patch_size}"
                )
            if class_token.ndim == 3 and class_token.shape[1] == 1:
                class_token = class_token[:, 0]
            if class_token.ndim != 2:
                raise RuntimeError(
                    f"Expected CLS token [B,C], got {tuple(class_token.shape)}"
                )

            patch_map = patch_tokens.transpose(1, 2).reshape(
                image.shape[0], self.embed_dim, grid_h, grid_w
            )
            patch_maps.append(patch_map)
            class_tokens.append(class_token)

        return patch_maps, class_tokens


class ChannelAttention(nn.Module):
    """Channel attention gate used inside residual lesion attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """Spatial attention gate based on channel average and maximum."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class ResidualLesionAttention(nn.Module):
    """CBAM-like attention with a bounded, learnable residual scale.

    The residual formulation preserves pretrained evidence when attention is
    initially uncertain and avoids the double-sigmoid attenuation used in V3.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x: Tensor) -> Tensor:
        channel_gate = self.channel_attention(x)
        spatial_gate = self.spatial_attention(x * channel_gate)
        scale = torch.tanh(self.residual_scale)
        return x * (1.0 + scale * channel_gate * spatial_gate)


class MaskedGeMPool2d(nn.Module):
    """Generalized-mean pooling restricted to a soft foreground mask."""

    def __init__(self, initial_p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(initial_p)))
        self.eps = float(eps)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(
                mask, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        mask = mask.clamp(0.0, 1.0)
        mask_mass = mask.sum(dim=(2, 3), keepdim=True)
        mask = torch.where(mask_mass > self.eps, mask, torch.ones_like(mask))

        p = self.p.clamp(1.0, 6.0)
        numerator = (x.clamp_min(self.eps).pow(p) * mask).sum(dim=(2, 3))
        denominator = mask.sum(dim=(2, 3)).clamp_min(self.eps)
        return (numerator / denominator).clamp_min(self.eps).pow(1.0 / p)


class LocalTopKPool(nn.Module):
    """Learned top-k token pooling constrained by the leaf mask."""

    def __init__(
        self,
        channels: int,
        topk_ratio: float = 0.15,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        hidden_channels = max(channels // 4, 32)
        self.score_head = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.topk_ratio = float(topk_ratio)
        self.temperature = float(temperature)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(
                mask, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        mask = mask.clamp(0.0, 1.0)
        mask_mass = mask.sum(dim=(2, 3), keepdim=True)
        mask = torch.where(mask_mass > 1e-6, mask, torch.ones_like(mask))

        score = self.score_head(x)
        score = score + mask.clamp_min(1e-6).log()
        score_flat = score.flatten(1)
        token_features = x.flatten(2).transpose(1, 2)

        num_tokens = token_features.shape[1]
        k = max(1, min(num_tokens, math.ceil(num_tokens * self.topk_ratio)))
        top_scores, top_indices = torch.topk(score_flat, k=k, dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, x.shape[1])
        selected_features = torch.gather(token_features, 1, gather_index)
        weights = F.softmax(top_scores / self.temperature, dim=1)
        pooled = (selected_features * weights.unsqueeze(-1)).sum(dim=1)

        attention_flat = torch.zeros_like(score_flat).scatter(1, top_indices, weights)
        attention_map = attention_flat.reshape(
            x.shape[0], 1, x.shape[-2], x.shape[-1]
        )
        return pooled, attention_map


class HighFrequencyDetailBranch(nn.Module):
    """Optional small branch for masked, high-frequency lesion texture.

    It is disabled by default.  The branch uses a local high-pass residual and
    GroupNorm only; it does not inject random feature noise.
    """

    def __init__(
        self,
        output_dim: int = 128,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        channels_1, channels_2 = 32, 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels_1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(
                _group_count(channels_1, group_norm_groups), channels_1
            ),
            nn.GELU(),
            nn.Conv2d(
                channels_1,
                channels_2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(channels_2, group_norm_groups), channels_2
            ),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels_2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        local_mean = F.avg_pool2d(
            image,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        detail = (image - local_mean).abs()
        detail_map = self.encoder(detail)

        # Pool away from the segmentation edge so optional detail features do
        # not learn SAM boundaries or crop-resize halos as disease texture.
        original_mask = mask.clamp(0.0, 1.0)
        eroded_mask = 1.0 - F.max_pool2d(
            1.0 - original_mask, kernel_size=7, stride=1, padding=3
        )
        mask = F.interpolate(
            eroded_mask,
            size=detail_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        fallback_mask = F.interpolate(
            original_mask,
            size=detail_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        mask_mass = mask.sum(dim=(2, 3), keepdim=True)
        mask = torch.where(mask_mass > 1e-6, mask, fallback_mask)
        pooled = (detail_map * mask).sum(dim=(2, 3))
        pooled = pooled / mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return self.projection(pooled)


class CosineClassifier(nn.Module):
    """Normalized linear classifier with a learnable bounded logit scale."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        initial_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        normalized_x = F.normalize(x, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * F.linear(normalized_x, normalized_weight)


class FinalModelV4(nn.Module):
    """Leaf-first, source-only plant disease model built on one DINOv2.

    Parameters
    ----------
    num_disease_classes:
        Number of joint plant-disease classes (38 for PlantVillage).
    num_plant_classes:
        Number of plant-species classes (14 for PlantVillage).
    num_health_classes:
        Number of health-state classes, normally healthy/diseased = 2.
    model_name:
        DINOv2 torch hub model name.  Unlike V3, this argument is honored.
    num_intermediate_layers:
        Number of final DINOv2 layers fused for patch and CLS features.
    feature_dim:
        Width of fused leaf/context features.
    projection_dim:
        Width of normalized contrastive representation ``z``.
    use_detail_branch:
        Enable the optional lightweight high-frequency branch.
    leaf_background_value:
        Fill value outside the mask.  Zero is neutral for ImageNet-normalized
        images and is therefore the default.
    crop_leaf_by_mask:
        Crop the automatic leaf view to the foreground bounding box before it
        is resized.  This keeps small field leaves large enough for lesion and
        vein tokens to survive the DINOv2 patch embedding.
    leaf_crop_margin:
        Fractional margin around the foreground bounding box.
    class_to_plant, class_to_health:
        Optional source-label hierarchy.  When supplied, plant and health heads
        softly calibrate joint disease logits; they never hard-mask classes.
    backbone_model:
        Optional pre-created DINOv2 model for offline use/testing.
    """

    def __init__(
        self,
        num_disease_classes: int = 38,
        num_plant_classes: int = 14,
        num_health_classes: int = 2,
        *,
        model_name: str = "dinov2_vits14",
        num_intermediate_layers: int = 4,
        feature_dim: int = 512,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        topk_ratio: float = 0.15,
        dropout: float = 0.2,
        group_norm_groups: int = 32,
        use_detail_branch: bool = False,
        detail_dim: int = 128,
        use_cosine_heads: bool = True,
        leaf_background_value: float = 0.0,
        crop_leaf_by_mask: bool = True,
        leaf_crop_margin: float = 0.10,
        class_to_plant: Optional[Sequence[int]] = None,
        class_to_health: Optional[Sequence[int]] = None,
        plant_hierarchy_logit_weight: float = 0.35,
        health_hierarchy_logit_weight: float = 0.15,
        hub_repo: str = PINNED_DINOV2_HUB_REPO,
        backbone_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or projection_dim < 1 or projection_hidden_dim < 1:
            raise ValueError("Feature and projection dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= leaf_crop_margin <= 0.5:
            raise ValueError("leaf_crop_margin must be in [0, 0.5]")
        if plant_hierarchy_logit_weight < 0.0 or health_hierarchy_logit_weight < 0.0:
            raise ValueError("hierarchy logit weights must be non-negative")

        self.num_disease_classes = int(num_disease_classes)
        self.num_plant_classes = int(num_plant_classes)
        self.num_health_classes = int(num_health_classes)
        self.feature_dim = int(feature_dim)
        self.use_detail_branch = bool(use_detail_branch)
        self.leaf_background_value = float(leaf_background_value)
        self.crop_leaf_by_mask = bool(crop_leaf_by_mask)
        self.leaf_crop_margin = float(leaf_crop_margin)
        self.plant_hierarchy_logit_weight = float(plant_hierarchy_logit_weight)
        self.health_hierarchy_logit_weight = float(health_hierarchy_logit_weight)

        if (class_to_plant is None) != (class_to_health is None):
            raise ValueError("Provide both class_to_plant and class_to_health, or neither")
        if class_to_plant is None:
            plant_lookup = torch.empty(0, dtype=torch.long)
            health_lookup = torch.empty(0, dtype=torch.long)
        else:
            assert class_to_health is not None
            if len(class_to_plant) != self.num_disease_classes or len(
                class_to_health
            ) != self.num_disease_classes:
                raise ValueError("Hierarchy lookup length must equal disease classes")
            plant_lookup = torch.as_tensor(class_to_plant, dtype=torch.long)
            health_lookup = torch.as_tensor(class_to_health, dtype=torch.long)
            if torch.any((plant_lookup < 0) | (plant_lookup >= self.num_plant_classes)):
                raise ValueError("class_to_plant contains an invalid plant index")
            if torch.any((health_lookup < 0) | (health_lookup >= self.num_health_classes)):
                raise ValueError("class_to_health contains an invalid health index")
        self.register_buffer("class_to_plant", plant_lookup, persistent=True)
        self.register_buffer("class_to_health", health_lookup, persistent=True)

        self.backbone = DINOv2MultiLayerBackbone(
            model_name=model_name,
            num_intermediate_layers=num_intermediate_layers,
            hub_repo=hub_repo,
            backbone_model=backbone_model,
            freeze=True,
        )
        multi_layer_dim = self.backbone.embed_dim * num_intermediate_layers

        self.patch_fusion = nn.Sequential(
            nn.Conv2d(multi_layer_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(
                _group_count(feature_dim, group_norm_groups), feature_dim
            ),
            nn.GELU(),
        )
        self.cls_fusion = nn.Sequential(
            nn.Linear(multi_layer_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        # Disease evidence must remain leaf-dominant.  Context is only a small
        # residual correction and is capped at 25%; D3 consistency can drive it
        # even lower when background is unreliable.
        self.context_residual_logit = nn.Parameter(torch.tensor(-2.20))
        self.residual_attention = ResidualLesionAttention(feature_dim)
        self.foreground_pool = MaskedGeMPool2d(initial_p=3.0)
        self.local_pool = LocalTopKPool(
            feature_dim, topk_ratio=topk_ratio, temperature=1.0
        )

        if self.use_detail_branch:
            self.detail_branch: Optional[HighFrequencyDetailBranch] = (
                HighFrequencyDetailBranch(
                    output_dim=detail_dim,
                    group_norm_groups=min(group_norm_groups, 8),
                )
            )
        else:
            self.detail_branch = None

        combined_dim = feature_dim * 3 + (detail_dim if use_detail_branch else 0)
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

        classifier_type = CosineClassifier if use_cosine_heads else nn.Linear
        self.disease_head = classifier_type(feature_dim, num_disease_classes)
        self.plant_head = classifier_type(feature_dim, num_plant_classes)
        self.health_head = classifier_type(feature_dim, num_health_classes)

    @staticmethod
    def _prepare_mask(
        mask: Tensor,
        *,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Normalize a binary/soft mask to [B,1,H,W] in the image domain."""
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}")
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"Image batch is {batch_size}, but mask batch is {mask.shape[0]}"
            )
        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)

        per_sample_max = mask.amax(dim=(1, 2, 3), keepdim=True)
        mask = torch.where(
            per_sample_max > 1.0,
            mask / per_sample_max.clamp_min(1e-6),
            mask,
        ).clamp(0.0, 1.0)
        if mask.shape[-2:] != spatial_size:
            mask = F.interpolate(
                mask, size=spatial_size, mode="bilinear", align_corners=False
            )

        # An empty segmentation means "mask unavailable", not "empty image".
        valid = mask.amax(dim=(1, 2, 3), keepdim=True) > 1e-6
        return torch.where(valid, mask, torch.ones_like(mask))

    def _make_leaf_view(
        self,
        image: Tensor,
        mask: Tensor,
        leaf_image: Optional[Tensor],
        leaf_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if leaf_image is None:
            if self.crop_leaf_by_mask:
                focused_image, focused_mask = self._crop_batch_to_mask(image, mask)
            else:
                focused_mask = mask
                focused_image = image
        else:
            if leaf_image.ndim != 4 or leaf_image.shape[0] != image.shape[0]:
                raise ValueError(
                    "leaf_image must have shape [B,3,H,W] and match image batch"
                )
            if leaf_image.shape[1] != image.shape[1]:
                raise ValueError("leaf_image and image must have equal channels")
            focused_image = leaf_image.to(device=image.device, dtype=image.dtype)
            if focused_image.shape[-2:] != image.shape[-2:]:
                focused_image = F.interpolate(
                    focused_image,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if leaf_mask is None:
                focused_mask = torch.ones_like(mask)
            else:
                focused_mask = self._prepare_mask(
                    leaf_mask,
                    batch_size=image.shape[0],
                    spatial_size=image.shape[-2:],
                    device=image.device,
                    dtype=image.dtype,
                )

        fill = torch.as_tensor(
            self.leaf_background_value, device=image.device, dtype=image.dtype
        )
        leaf_view = focused_image * focused_mask + fill * (1.0 - focused_mask)
        return leaf_view, focused_mask, focused_image

    def _crop_batch_to_mask(
        self,
        image: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Crop each sample to its mask box and resize back to input size.

        Bounding-box coordinates are intentionally non-differentiable; masks
        are supervision/geometry rather than learned tensors.  The image crop
        and resize remain differentiable with respect to image features.
        """

        output_size = image.shape[-2:]
        cropped_images: List[Tensor] = []
        cropped_masks: List[Tensor] = []
        height, width = output_size
        for sample_image, sample_mask in zip(image, mask):
            foreground = sample_mask[0] > 0.05
            coordinates = foreground.nonzero(as_tuple=False)
            if coordinates.numel() == 0:
                y0, y1, x0, x1 = 0, height, 0, width
            else:
                y_min, x_min = coordinates.amin(dim=0)
                y_max, x_max = coordinates.amax(dim=0)
                box_height = int((y_max - y_min + 1).item())
                box_width = int((x_max - x_min + 1).item())
                margin_y = max(2, round(box_height * self.leaf_crop_margin))
                margin_x = max(2, round(box_width * self.leaf_crop_margin))
                y0 = max(0, int(y_min.item()) - margin_y)
                y1 = min(height, int(y_max.item()) + 1 + margin_y)
                x0 = max(0, int(x_min.item()) - margin_x)
                x1 = min(width, int(x_max.item()) + 1 + margin_x)

            image_crop = sample_image[:, y0:y1, x0:x1].unsqueeze(0)
            mask_crop = sample_mask[:, y0:y1, x0:x1].unsqueeze(0)
            cropped_images.append(
                F.interpolate(
                    image_crop,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            cropped_masks.append(
                F.interpolate(
                    mask_crop,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            )
        return torch.cat(cropped_images, dim=0), torch.cat(cropped_masks, dim=0)

    def unfreeze_last_blocks(
        self,
        num_blocks: int = 1,
        *,
        unfreeze_norm: bool = True,
    ) -> List[nn.Parameter]:
        """Unfreeze final DINOv2 blocks for a low-learning-rate optimizer group."""
        return self.backbone.unfreeze_last_blocks(
            num_blocks, unfreeze_norm=unfreeze_norm
        )

    def train(self, mode: bool = True) -> "FinalModelV4":
        """Keep a fully frozen backbone deterministic while training the heads."""
        super().train(mode)
        if mode and not any(
            parameter.requires_grad for parameter in self.backbone.dinov2.parameters()
        ):
            self.backbone.dinov2.eval()
        return self

    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        *,
        leaf_image: Optional[Tensor] = None,
        leaf_mask: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        """Classify disease, plant species, and health state.

        Parameters
        ----------
        image:
            Context image ``[B,3,H,W]``, normally ImageNet-normalized.
        mask:
            Leaf foreground mask ``[B,1,H,W]``.  Values may be in [0,1] or
            [0,255].  Empty masks safely fall back to all-foreground.
        leaf_image:
            Optional pre-cropped leaf image.  It is resized to the context input
            size and replaces the automatically masked leaf view.
        leaf_mask:
            Optional mask aligned with ``leaf_image``.  If omitted, a supplied
            crop is treated as fully foreground.
        return_aux:
            Include branch features and the local top-k attention map.

        Returns
        -------
        dict
            Always contains hierarchy-calibrated ``disease_logits``,
            ``raw_disease_logits``, ``plant_logits``, ``health_logits``,
            normalized ``z``, and fused ``features``.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected RGB image [B,3,H,W], got {tuple(image.shape)}"
            )
        prepared_mask = self._prepare_mask(
            mask,
            batch_size=image.shape[0],
            spatial_size=image.shape[-2:],
            device=image.device,
            dtype=image.dtype,
        )
        leaf_view, focused_mask, unmasked_leaf_view = self._make_leaf_view(
            image, prepared_mask, leaf_image, leaf_mask
        )

        batch_size = image.shape[0]
        # One backbone instance and one invocation for both semantic views.
        joint_views = torch.cat((image, leaf_view), dim=0)
        patch_maps, class_tokens = self.backbone(joint_views)

        context_cls = torch.cat(
            [class_token[:batch_size] for class_token in class_tokens], dim=1
        )
        leaf_cls = torch.cat(
            [class_token[batch_size:] for class_token in class_tokens], dim=1
        )
        context_features = self.cls_fusion(context_cls)
        leaf_global_features = self.cls_fusion(leaf_cls)
        context_weight = 0.25 * torch.sigmoid(self.context_residual_logit)
        global_features = leaf_global_features + context_weight * (
            context_features - leaf_global_features
        )

        leaf_multilayer_map = torch.cat(
            [patch_map[batch_size:] for patch_map in patch_maps], dim=1
        )
        leaf_feature_map = self.patch_fusion(leaf_multilayer_map)
        leaf_feature_map = self.residual_attention(leaf_feature_map)

        token_mask = F.interpolate(
            focused_mask,
            size=leaf_feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        foreground_features = self.foreground_pool(leaf_feature_map, token_mask)
        local_features, local_attention = self.local_pool(
            leaf_feature_map, token_mask
        )

        components = [global_features, foreground_features, local_features]
        detail_features: Optional[Tensor] = None
        if self.detail_branch is not None:
            detail_features = self.detail_branch(unmasked_leaf_view, focused_mask)
            components.append(detail_features)

        features = self.feature_fusion(torch.cat(components, dim=1))
        z = F.normalize(self.projection_head(features), dim=1)

        raw_disease_logits = self.disease_head(features)
        plant_logits = self.plant_head(features)
        health_logits = self.health_head(features)
        disease_logits = raw_disease_logits
        if self.class_to_plant.numel():
            disease_logits = disease_logits + self.plant_hierarchy_logit_weight * (
                F.log_softmax(plant_logits, dim=1)[:, self.class_to_plant]
            )
            disease_logits = disease_logits + self.health_hierarchy_logit_weight * (
                F.log_softmax(health_logits, dim=1)[:, self.class_to_health]
            )

        output: Dict[str, Tensor] = {
            "disease_logits": disease_logits,
            "raw_disease_logits": raw_disease_logits,
            "plant_logits": plant_logits,
            "health_logits": health_logits,
            "z": z,
            "features": features,
        }
        if return_aux:
            output.update(
                {
                    "global_features": global_features,
                    "leaf_global_features": leaf_global_features,
                    "context_features": context_features,
                    "context_weight": context_weight.expand(batch_size, 1),
                    "foreground_features": foreground_features,
                    "local_features": local_features,
                    "local_attention": local_attention,
                    "token_mask": token_mask,
                }
            )
            if detail_features is not None:
                output["detail_features"] = detail_features
        return output
