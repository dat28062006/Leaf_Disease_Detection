"""Leaf-only parallel plant morphology and disease-symptom classifier.

``FinalModelV5`` deliberately gives SAM a narrow role: an external CPU data
pipeline uses the SAM mask to locate/crop the leaf and remove its background.
The model then receives only that prepared leaf crop and its aligned mask.  It
has no context-image input, no internal bounding-box crop, and no path through
which field or laboratory background can influence a prediction.

One shared DINOv2 invocation produces semantic leaf tokens.  Two branches read
those tokens in parallel:

* the morphology branch pools global, low-frequency leaf evidence and encodes
  mask shape to supervise plant identity;
* the symptom branch uses lesion attention, an eroded foreground mask, and
  local top-k pooling to supervise visual disease type.

The final 38-class prediction is made directly from feature-level joint fusion
of the two branches.  Plant and symptom logits are auxiliary outputs only: they
are never added to, multiplied with, used to gate, or used to mask the final
disease logits.  This is therefore not a sequential plant-first classifier.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # Support package imports and ``python new_model/file.py`` workflows.
    from .final_model_v4 import (
        PINNED_DINOV2_HUB_REPO,
        CosineClassifier,
        DINOv2MultiLayerBackbone,
        HighFrequencyDetailBranch,
        LocalTopKPool,
        MaskedGeMPool2d,
        ResidualLesionAttention,
    )
except ImportError:
    from final_model_v4 import (
        PINNED_DINOV2_HUB_REPO,
        CosineClassifier,
        DINOv2MultiLayerBackbone,
        HighFrequencyDetailBranch,
        LocalTopKPool,
        MaskedGeMPool2d,
        ResidualLesionAttention,
    )


__all__ = ["FinalModelV5", "PINNED_DINOV2_HUB_REPO"]


def _group_count(channels: int, requested_groups: int = 8) -> int:
    """Return the largest GroupNorm group count that divides ``channels``."""

    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class MaskShapeEncoder(nn.Module):
    """Encode the aligned SAM silhouette into a compact morphology feature.

    The crop pipeline should preserve aspect ratio; otherwise a square stretch
    would destroy precisely the silhouette information this encoder is meant to
    capture.  The encoder is intentionally small so imperfect SAM boundaries do
    not dominate the semantic DINOv2 representation.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("shape_dim must be positive")

        channels = (16, 32, 64)
        layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.GELU(),
                ]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, mask: Tensor) -> Tensor:
        encoded = self.encoder(mask)
        pooled = F.adaptive_avg_pool2d(encoded, output_size=1).flatten(1)
        return self.projection(pooled)


class FinalModelV5(nn.Module):
    """Leaf-only DINOv2 model with parallel morphology and symptom branches.

    Parameters
    ----------
    num_disease_classes:
        Number of final joint plant-disease labels, normally 38.
    num_plant_classes:
        Number of auxiliary plant-species labels, normally 14.
    num_symptom_classes:
        Number of auxiliary disease-type labels.  The current PlantVillage
        taxonomy has 21 unique ``disease_name`` values, including healthy.
    model_name, num_intermediate_layers, hub_repo:
        Frozen DINOv2 trunk configuration.  ``hub_repo`` defaults to the pinned
        V4 revision so a saved model config can reconstruct strictly.
    trunk_dim:
        Width of the shared patch/CLS representation and final joint feature.
    morphology_dim, symptom_dim:
        Widths of the two parallel branch features.
    interaction_dim:
        Common width used for feature-level joint fusion.
    projection_dim:
        Width of normalized contrastive representation ``z``.
    shape_dim:
        Width of the compact SAM-mask shape representation.
    use_detail_branch:
        Add V4's eroded-mask high-frequency image branch to symptom evidence.
        It is disabled by default because mask boundaries can create artificial
        high-frequency cues and should be enabled only after an ablation.
    leaf_background_value:
        Fill outside ``leaf_mask`` in normalized image space.  Zero corresponds
        to ImageNet mean and is the expected neutral background.
    mask_erosion_kernel:
        Odd image-domain kernel used to remove SAM/crop boundaries from symptom
        pooling.  If erosion empties a sample, its original foreground mask is
        used safely for that sample.
    backbone_model:
        Optional pre-created DINOv2-compatible module for offline use/tests.
        It is intentionally a runtime dependency and should not be serialized
        into ``model_config``.

    Notes
    -----
    ``forward`` accepts only ``leaf_image`` and ``leaf_mask``.  CPU preprocessing
    must first crop the SAM bounding box with margin, resize it while preserving
    aspect ratio, remove the background, and normalize the result.
    """

    def __init__(
        self,
        num_disease_classes: int = 38,
        num_plant_classes: int = 14,
        num_symptom_classes: int = 21,
        *,
        model_name: str = "dinov2_vits14",
        num_intermediate_layers: int = 4,
        trunk_dim: int = 512,
        morphology_dim: int = 512,
        symptom_dim: int = 512,
        interaction_dim: int = 256,
        projection_dim: int = 128,
        topk_ratio: float = 0.15,
        dropout: float = 0.20,
        shape_dim: int = 64,
        use_detail_branch: bool = False,
        detail_dim: int = 128,
        use_cosine_heads: bool = True,
        leaf_background_value: float = 0.0,
        mask_erosion_kernel: int = 7,
        hub_repo: str = PINNED_DINOV2_HUB_REPO,
        backbone_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        class_counts = (
            num_disease_classes,
            num_plant_classes,
            num_symptom_classes,
        )
        if any(int(count) < 1 for count in class_counts):
            raise ValueError("All class counts must be positive")
        dimensions = (
            trunk_dim,
            morphology_dim,
            symptom_dim,
            interaction_dim,
            projection_dim,
            shape_dim,
            detail_dim,
        )
        if any(int(dimension) < 1 for dimension in dimensions):
            raise ValueError("All feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if mask_erosion_kernel < 1 or mask_erosion_kernel % 2 == 0:
            raise ValueError("mask_erosion_kernel must be a positive odd integer")

        self.num_disease_classes = int(num_disease_classes)
        self.num_plant_classes = int(num_plant_classes)
        self.num_symptom_classes = int(num_symptom_classes)
        self.trunk_dim = int(trunk_dim)
        self.morphology_dim = int(morphology_dim)
        self.symptom_dim = int(symptom_dim)
        self.interaction_dim = int(interaction_dim)
        self.projection_dim = int(projection_dim)
        self.use_detail_branch = bool(use_detail_branch)
        self.leaf_background_value = float(leaf_background_value)
        self.mask_erosion_kernel = int(mask_erosion_kernel)

        self.backbone = DINOv2MultiLayerBackbone(
            model_name=model_name,
            num_intermediate_layers=num_intermediate_layers,
            hub_repo=hub_repo,
            backbone_model=backbone_model,
            freeze=True,
        )
        multilayer_dim = self.backbone.embed_dim * int(num_intermediate_layers)
        self.patch_fusion = nn.Sequential(
            nn.Conv2d(multilayer_dim, trunk_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(trunk_dim, 32), trunk_dim),
            nn.GELU(),
        )
        self.cls_fusion = nn.Sequential(
            nn.Linear(multilayer_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
        )

        # Morphology branch: broad semantic evidence plus a deliberately small
        # shape code.  Blur pooling discourages reliance on isolated lesions.
        self.morphology_pool = MaskedGeMPool2d(initial_p=3.0)
        self.shape_encoder = MaskShapeEncoder(output_dim=shape_dim)
        morphology_input_dim = trunk_dim * 2 + shape_dim
        self.morphology_branch = nn.Sequential(
            nn.LayerNorm(morphology_input_dim),
            nn.Linear(morphology_input_dim, morphology_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Symptom branch: lesion-aware map and local evidence pooled away from
        # the mask edge.  Its auxiliary classifier remains independent of the
        # direct joint disease classifier below.
        self.symptom_attention = ResidualLesionAttention(trunk_dim)
        self.symptom_foreground_pool = MaskedGeMPool2d(initial_p=3.0)
        self.symptom_local_pool = LocalTopKPool(
            trunk_dim, topk_ratio=topk_ratio, temperature=1.0
        )
        if self.use_detail_branch:
            self.detail_branch: Optional[HighFrequencyDetailBranch] = (
                HighFrequencyDetailBranch(
                    output_dim=detail_dim,
                    group_norm_groups=8,
                )
            )
        else:
            self.detail_branch = None
        symptom_input_dim = trunk_dim * 2 + (
            detail_dim if self.use_detail_branch else 0
        )
        self.symptom_branch = nn.Sequential(
            nn.LayerNorm(symptom_input_dim),
            nn.Linear(symptom_input_dim, symptom_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        classifier_type = CosineClassifier if use_cosine_heads else nn.Linear
        self.plant_head = classifier_type(morphology_dim, num_plant_classes)
        self.symptom_head = classifier_type(symptom_dim, num_symptom_classes)

        self.morphology_to_interaction = nn.Sequential(
            nn.LayerNorm(morphology_dim),
            nn.Linear(morphology_dim, interaction_dim),
            nn.GELU(),
        )
        self.symptom_to_interaction = nn.Sequential(
            nn.LayerNorm(symptom_dim),
            nn.Linear(symptom_dim, interaction_dim),
            nn.GELU(),
        )
        self.joint_fusion = nn.Sequential(
            nn.LayerNorm(interaction_dim * 3),
            nn.Linear(interaction_dim * 3, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.disease_head = classifier_type(trunk_dim, num_disease_classes)

        projection_hidden_dim = max(projection_dim, trunk_dim // 2)
        self.projection_head = nn.Sequential(
            nn.Linear(trunk_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    @staticmethod
    def _prepare_mask(
        mask: Tensor,
        *,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Normalize an aligned leaf mask to soft ``[B,1,H,W]`` in ``[0,1]``."""

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected leaf_mask [B,1,H,W], got {tuple(mask.shape)}")
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"leaf_image batch is {batch_size}, but leaf_mask batch is "
                f"{mask.shape[0]}"
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
                mask,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

        # Leaf-only V5 is fail-closed: an empty segmentation must never turn a
        # full laboratory/field frame into an all-foreground model input.
        valid = mask.amax(dim=(1, 2, 3)) > 1e-6
        if not bool(valid.all()):
            bad = (~valid).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Empty leaf_mask for batch indices {bad}")
        return mask

    def _erode_symptom_mask(self, mask: Tensor) -> Tensor:
        """Remove SAM/crop boundaries and fall back per sample if erosion empties."""

        if self.mask_erosion_kernel == 1:
            return mask
        radius = self.mask_erosion_kernel // 2
        background = 1.0 - mask.clamp(0.0, 1.0)
        padded_background = F.pad(
            background,
            (radius, radius, radius, radius),
            mode="constant",
            value=1.0,
        )
        eroded = 1.0 - F.max_pool2d(
            padded_background,
            kernel_size=self.mask_erosion_kernel,
            stride=1,
            padding=0,
        )
        valid = eroded.sum(dim=(1, 2, 3), keepdim=True) > 1e-6
        return torch.where(valid, eroded, mask)

    def unfreeze_last_blocks(
        self,
        num_blocks: int = 1,
        *,
        unfreeze_norm: bool = True,
    ) -> List[nn.Parameter]:
        """Unfreeze final DINOv2 blocks for a low-learning-rate optimizer group."""

        return self.backbone.unfreeze_last_blocks(
            num_blocks,
            unfreeze_norm=unfreeze_norm,
        )

    def train(self, mode: bool = True) -> "FinalModelV5":
        """Keep a fully frozen DINOv2 trunk deterministic while heads train."""

        super().train(mode)
        if mode and not any(
            parameter.requires_grad for parameter in self.backbone.dinov2.parameters()
        ):
            self.backbone.dinov2.eval()
        return self

    def forward(
        self,
        leaf_image: Tensor,
        leaf_mask: Tensor,
        *,
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        """Classify one externally cropped, background-removed leaf batch.

        Parameters
        ----------
        leaf_image:
            ImageNet-normalized SAM leaf crops shaped ``[B,3,H,W]``.
        leaf_mask:
            Masks aligned with ``leaf_image``, shaped ``[B,1,H,W]`` (or
            ``[B,H,W]``), with values in either ``[0,1]`` or ``[0,255]``.
        return_aux:
            Include diagnostic pooling features and attention/mask maps.

        Returns
        -------
        dict
            Always contains ``disease_logits``, ``plant_logits``,
            ``symptom_logits``, ``morphology_features``, ``symptom_features``,
            joint ``features``, and normalized projection ``z``.  The final
            disease logits come only from joint feature fusion; auxiliary
            logits cannot modify them.
        """

        if leaf_image.ndim != 4 or leaf_image.shape[1] != 3:
            raise ValueError(
                "Expected leaf_image RGB tensor [B,3,H,W], got "
                f"{tuple(leaf_image.shape)}"
            )
        prepared_mask = self._prepare_mask(
            leaf_mask,
            batch_size=leaf_image.shape[0],
            spatial_size=leaf_image.shape[-2:],
            device=leaf_image.device,
            dtype=leaf_image.dtype,
        )
        fill = torch.as_tensor(
            self.leaf_background_value,
            device=leaf_image.device,
            dtype=leaf_image.dtype,
        )
        leaf_view = leaf_image * prepared_mask + fill * (1.0 - prepared_mask)

        # Exactly one leaf-only backbone invocation.  Unlike V4, there is no
        # context view, concatenated double batch, or context residual weight.
        patch_maps, class_tokens = self.backbone(leaf_view)
        shared_map = self.patch_fusion(torch.cat(patch_maps, dim=1))
        leaf_cls = self.cls_fusion(torch.cat(class_tokens, dim=1))

        token_mask = F.interpolate(
            prepared_mask,
            size=shared_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        morphology_map = F.avg_pool2d(
            shared_map,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        morphology_global = self.morphology_pool(morphology_map, token_mask)
        shape_features = self.shape_encoder(prepared_mask)
        morphology_features = self.morphology_branch(
            torch.cat((leaf_cls, morphology_global, shape_features), dim=1)
        )
        plant_logits = self.plant_head(morphology_features)

        eroded_mask = self._erode_symptom_mask(prepared_mask)
        eroded_token_mask = F.interpolate(
            eroded_mask,
            size=shared_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        symptom_map = self.symptom_attention(shared_map)
        symptom_global = self.symptom_foreground_pool(
            symptom_map,
            eroded_token_mask,
        )
        symptom_local, symptom_attention_map = self.symptom_local_pool(
            symptom_map,
            eroded_token_mask,
        )
        symptom_components = [symptom_global, symptom_local]
        detail_features: Optional[Tensor] = None
        if self.detail_branch is not None:
            # V4's detail branch performs its own edge erosion; pass the original
            # foreground rather than eroding twice.
            detail_features = self.detail_branch(leaf_view, prepared_mask)
            symptom_components.append(detail_features)
        symptom_features = self.symptom_branch(
            torch.cat(symptom_components, dim=1)
        )
        symptom_logits = self.symptom_head(symptom_features)

        morphology_interaction = F.normalize(
            self.morphology_to_interaction(morphology_features),
            dim=1,
        )
        symptom_interaction = F.normalize(
            self.symptom_to_interaction(symptom_features),
            dim=1,
        )
        joint_input = torch.cat(
            (
                morphology_interaction,
                symptom_interaction,
                morphology_interaction * symptom_interaction,
            ),
            dim=1,
        )
        features = self.joint_fusion(joint_input)
        z = F.normalize(self.projection_head(features), dim=1)

        # Direct joint classifier.  Keep this assignment isolated so it is
        # structurally clear that auxiliary logits never calibrate or gate it.
        disease_logits = self.disease_head(features)
        output: Dict[str, Tensor] = {
            "disease_logits": disease_logits,
            "plant_logits": plant_logits,
            "symptom_logits": symptom_logits,
            "morphology_features": morphology_features,
            "symptom_features": symptom_features,
            "features": features,
            "z": z,
        }
        if return_aux:
            output.update(
                {
                    "leaf_view": leaf_view,
                    "token_mask": token_mask,
                    "eroded_token_mask": eroded_token_mask,
                    "morphology_global_features": morphology_global,
                    "shape_features": shape_features,
                    "symptom_global_features": symptom_global,
                    "symptom_local_features": symptom_local,
                    # Stable trainer-facing name shared with the V4 loss API.
                    "local_attention": symptom_attention_map,
                    "symptom_attention_map": symptom_attention_map,
                    "morphology_interaction_features": morphology_interaction,
                    "symptom_interaction_features": symptom_interaction,
                }
            )
            if detail_features is not None:
                output["detail_features"] = detail_features
        return output
