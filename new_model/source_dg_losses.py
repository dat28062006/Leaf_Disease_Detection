"""Reusable losses and training utilities for source-only domain generalization.

The helpers in this module deliberately have no dependency on the dataset or
training entry point.  They are suitable for the weak/strong-view training used
by the V4 plant-disease model, while remaining useful in smaller ablations.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


Reduction = Literal["none", "mean", "sum"]
ConsistencyKind = Literal["js", "symmetric_kl"]
TeacherView = Literal["weak", "strong", "none"]
BufferMode = Literal["copy", "ema", "none"]

__all__ = [
    "ConfidenceGatedTeacherLoss",
    "CrossBatchSupervisedContrastiveLoss",
    "ClassificationMetricsAccumulator",
    "ForegroundAttentionLoss",
    "HierarchicalLabelMapping",
    "ModelEMA",
    "SupervisedContrastiveLoss",
    "WeakStrongConsistencyLoss",
    "build_hierarchical_label_mapping",
    "infer_plant_name",
    "is_healthy_label",
    "jensen_shannon_divergence",
    "normalize_class_name",
    "symmetric_kl_divergence",
    "update_ema_model",
]


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            f"reduction must be 'none', 'mean', or 'sum'; got {reduction!r}"
        )


def _reduce_valid_losses(
    losses: Tensor,
    valid: Tensor,
    reduction: Reduction,
    *,
    reference: Tensor,
) -> Tensor:
    """Reduce losses while making the all-invalid case differentiable."""

    valid = valid.to(device=losses.device, dtype=torch.bool)
    if reduction == "none":
        return torch.where(valid, losses, torch.zeros_like(losses))

    selected = losses[valid]
    if selected.numel() == 0:
        return reference.sum() * 0.0
    if reduction == "sum":
        return selected.sum()
    return selected.mean()


class WeakStrongConsistencyLoss(nn.Module):
    """Consistency loss between weakly and strongly augmented predictions.

    Args:
        divergence: ``"js"`` for Jensen-Shannon divergence or
            ``"symmetric_kl"`` for the mean of both KL directions.
        temperature: Softmax temperature.  Values above one soften targets.
        teacher: Which view acts as the teacher when ``detach_teacher`` is true.
            ``"weak"`` is the usual source-DG setting.  ``"none"`` leaves both
            branches attached regardless of ``detach_teacher``.
        detach_teacher: Stop gradients through the selected teacher view.
        reduction: Reduction over all non-class dimensions.
        scale_by_temperature: Multiply by ``temperature ** 2`` to retain a
            comparable gradient scale when using soft targets.
        class_dim: Dimension containing class logits.

    Both divergences are computed from log-softmax values and are stable under
    mixed precision.  The returned loss is exactly zero for identical logits,
    up to floating-point roundoff.
    """

    def __init__(
        self,
        divergence: ConsistencyKind = "js",
        temperature: float = 1.0,
        teacher: TeacherView = "weak",
        detach_teacher: bool = True,
        reduction: Reduction = "mean",
        scale_by_temperature: bool = True,
        class_dim: int = -1,
    ) -> None:
        super().__init__()
        if divergence not in {"js", "symmetric_kl"}:
            raise ValueError(f"unsupported divergence: {divergence!r}")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        if teacher not in {"weak", "strong", "none"}:
            raise ValueError(f"unsupported teacher view: {teacher!r}")
        _validate_reduction(reduction)

        self.divergence = divergence
        self.temperature = float(temperature)
        self.teacher = teacher
        self.detach_teacher = bool(detach_teacher)
        self.reduction = reduction
        self.scale_by_temperature = bool(scale_by_temperature)
        self.class_dim = int(class_dim)

    def forward(self, weak_logits: Tensor, strong_logits: Tensor) -> Tensor:
        if weak_logits.shape != strong_logits.shape:
            raise ValueError(
                "weak_logits and strong_logits must have identical shapes; "
                f"got {tuple(weak_logits.shape)} and {tuple(strong_logits.shape)}"
            )
        if not weak_logits.is_floating_point() or not strong_logits.is_floating_point():
            raise TypeError("consistency logits must be floating-point tensors")
        if weak_logits.ndim == 0:
            raise ValueError("logits must have at least one dimension")

        weak_scaled = weak_logits.float() / self.temperature
        strong_scaled = strong_logits.float() / self.temperature
        weak_log_prob = F.log_softmax(weak_scaled, dim=self.class_dim)
        strong_log_prob = F.log_softmax(strong_scaled, dim=self.class_dim)
        weak_prob = weak_log_prob.exp()
        strong_prob = strong_log_prob.exp()

        if self.detach_teacher and self.teacher == "weak":
            weak_log_prob = weak_log_prob.detach()
            weak_prob = weak_prob.detach()
        elif self.detach_teacher and self.teacher == "strong":
            strong_log_prob = strong_log_prob.detach()
            strong_prob = strong_prob.detach()

        if self.divergence == "symmetric_kl":
            weak_to_strong = (
                weak_prob * (weak_log_prob - strong_log_prob)
            ).sum(dim=self.class_dim)
            strong_to_weak = (
                strong_prob * (strong_log_prob - weak_log_prob)
            ).sum(dim=self.class_dim)
            losses = 0.5 * (weak_to_strong + strong_to_weak)
        else:
            midpoint = 0.5 * (weak_prob + strong_prob)
            midpoint_log = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
            weak_js = (weak_prob * (weak_log_prob - midpoint_log)).sum(
                dim=self.class_dim
            )
            strong_js = (strong_prob * (strong_log_prob - midpoint_log)).sum(
                dim=self.class_dim
            )
            losses = 0.5 * (weak_js + strong_js)

        # Tiny negative values are possible for JS because of roundoff.
        losses = losses.clamp_min(0.0)
        if self.scale_by_temperature:
            losses = losses * (self.temperature**2)

        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


def jensen_shannon_divergence(
    weak_logits: Tensor,
    strong_logits: Tensor,
    *,
    temperature: float = 1.0,
    teacher: TeacherView = "weak",
    detach_teacher: bool = True,
    reduction: Reduction = "mean",
    scale_by_temperature: bool = True,
    class_dim: int = -1,
) -> Tensor:
    """Functional Jensen-Shannon weak/strong consistency loss."""

    return WeakStrongConsistencyLoss(
        divergence="js",
        temperature=temperature,
        teacher=teacher,
        detach_teacher=detach_teacher,
        reduction=reduction,
        scale_by_temperature=scale_by_temperature,
        class_dim=class_dim,
    )(weak_logits, strong_logits)


def symmetric_kl_divergence(
    weak_logits: Tensor,
    strong_logits: Tensor,
    *,
    temperature: float = 1.0,
    teacher: TeacherView = "weak",
    detach_teacher: bool = True,
    reduction: Reduction = "mean",
    scale_by_temperature: bool = True,
    class_dim: int = -1,
) -> Tensor:
    """Functional symmetric-KL weak/strong consistency loss."""

    return WeakStrongConsistencyLoss(
        divergence="symmetric_kl",
        temperature=temperature,
        teacher=teacher,
        detach_teacher=detach_teacher,
        reduction=reduction,
        scale_by_temperature=scale_by_temperature,
        class_dim=class_dim,
    )(weak_logits, strong_logits)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss supporting multiple views per sample.

    ``features`` must have shape ``[batch, views, feature_dim...]``.  With two
    views, every valid anchor has its paired view as a positive even when all
    labels in the batch are unique.  With one view, anchors that have no
    positive are ignored.  If no valid positives exist at all, the method
    returns a differentiable zero rather than NaN.

    Passing ``labels=None`` produces an instance-discrimination objective where
    only other views of the same input are positives.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float | None = None,
        contrast_mode: Literal["all", "one"] = "all",
        reduction: Reduction = "mean",
        normalize: bool = True,
        ignore_index: int | None = -1,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        if base_temperature is not None and (
            not math.isfinite(base_temperature) or base_temperature <= 0.0
        ):
            raise ValueError("base_temperature must be finite and positive")
        if contrast_mode not in {"all", "one"}:
            raise ValueError("contrast_mode must be 'all' or 'one'")
        _validate_reduction(reduction)

        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature or temperature)
        self.contrast_mode = contrast_mode
        self.reduction = reduction
        self.normalize = bool(normalize)
        self.ignore_index = ignore_index

    def forward(self, features: Tensor, labels: Tensor | None = None) -> Tensor:
        if features.ndim < 3:
            raise ValueError(
                "features must have shape [batch, views, feature_dim...]; "
                f"got {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")

        batch_size, num_views = features.shape[:2]
        if batch_size == 0 or num_views == 0:
            return features.sum() * 0.0

        features = features.reshape(batch_size, num_views, -1)
        if labels is not None:
            labels = labels.reshape(-1).to(device=features.device)
            if labels.numel() != batch_size:
                raise ValueError(
                    f"expected {batch_size} labels, got {labels.numel()}"
                )
            if self.ignore_index is not None:
                keep = labels != self.ignore_index
                features = features[keep]
                labels = labels[keep]
                batch_size = int(keep.sum().item())
                if batch_size == 0:
                    return features.sum() * 0.0

        work_features = features.float()
        if self.normalize:
            work_features = F.normalize(work_features, dim=-1)

        if labels is None:
            positive_mask = torch.eye(
                batch_size, device=features.device, dtype=work_features.dtype
            )
        else:
            positive_mask = labels[:, None].eq(labels[None, :]).to(
                dtype=work_features.dtype
            )

        # View-major layout: [view0_batch, view1_batch, ...].
        contrast_features = torch.cat(torch.unbind(work_features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_features = work_features[:, 0]
            anchor_count = 1
        else:
            anchor_features = contrast_features
            anchor_count = num_views

        logits = anchor_features @ contrast_features.T
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        positive_mask = positive_mask.repeat(anchor_count, num_views)
        logits_mask = torch.ones_like(positive_mask)
        self_indices = torch.arange(
            batch_size * anchor_count, device=features.device
        ).view(-1, 1)
        logits_mask.scatter_(1, self_indices, 0.0)
        positive_mask = positive_mask * logits_mask

        exp_logits = logits.exp() * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(exp_logits.dtype).tiny
            )
        )

        positive_count = positive_mask.sum(dim=1)
        valid_anchors = positive_count > 0
        mean_log_prob_positive = (
            (positive_mask * log_prob).sum(dim=1)
            / positive_count.clamp_min(1.0)
        )
        losses = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_positive
        return _reduce_valid_losses(
            losses,
            valid_anchors,
            self.reduction,
            reference=features,
        )


class CrossBatchSupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive learning with a detached, checkpointable queue.

    The regular :class:`SupervisedContrastiveLoss` is reliable when a batch has
    many examples, but V5 deliberately uses micro-batches of two leaf crops for
    memory safety.  Gradient accumulation does not enlarge a contrastive batch,
    so it cannot provide enough class negatives or same-class positives.  This
    module uses student anchors and detached momentum-teacher keys, then adds a
    FIFO dictionary of older source keys as cross-batch positives and negatives.

    The queue contains source-only features and labels.  It never receives
    PlantDoc data and therefore remains valid for the source-only protocol.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        queue_size: int = 256,
        temperature: float = 0.10,
        base_temperature: float | None = None,
        normalize: bool = True,
        ignore_index: int | None = -1,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        if base_temperature is not None and (
            not math.isfinite(base_temperature) or base_temperature <= 0.0
        ):
            raise ValueError("base_temperature must be finite and positive")

        self.feature_dim = int(feature_dim)
        self.queue_size = int(queue_size)
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature or temperature)
        self.normalize = bool(normalize)
        self.ignore_index = ignore_index
        self.register_buffer(
            "queue_features", torch.zeros(queue_size, feature_dim), persistent=True
        )
        self.register_buffer(
            "queue_labels",
            torch.full((queue_size,), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))

    @property
    def stored_features(self) -> int:
        """Number of valid feature keys currently held in the FIFO queue."""

        return int(self.queue_count.item())

    @torch.no_grad()
    def enqueue(self, keys: Tensor, labels: Tensor) -> None:
        """Append detached teacher keys after a successful optimizer step."""
        if keys.ndim != 2 or keys.shape[1] != self.feature_dim:
            raise ValueError(
                "queue keys must have shape [batch, feature_dim]; got "
                f"{tuple(keys.shape)}"
            )
        labels = labels.reshape(-1).to(device=keys.device, dtype=torch.long)
        if labels.numel() != keys.shape[0]:
            raise ValueError("queue key and label counts differ")
        if keys.numel() == 0:
            return

        keys = keys.detach().to(
            device=self.queue_features.device, dtype=self.queue_features.dtype
        )
        labels = labels.detach().to(device=self.queue_labels.device)
        if self.normalize:
            keys = F.normalize(keys, dim=1)

        # Retain the newest keys if a caller ever passes more than one queue.
        if keys.shape[0] >= self.queue_size:
            keys = keys[-self.queue_size :]
            labels = labels[-self.queue_size :]

        count = int(keys.shape[0])
        pointer = int(self.queue_pointer.item())
        first = min(count, self.queue_size - pointer)
        self.queue_features[pointer : pointer + first].copy_(keys[:first])
        self.queue_labels[pointer : pointer + first].copy_(labels[:first])
        remaining = count - first
        if remaining:
            self.queue_features[:remaining].copy_(keys[first:])
            self.queue_labels[:remaining].copy_(labels[first:])
        self.queue_pointer.fill_((pointer + count) % self.queue_size)
        self.queue_count.fill_(min(self.queue_size, self.stored_features + count))

    def forward(
        self,
        anchor_features: Tensor,
        labels: Tensor,
        *,
        key_features: Tensor,
    ) -> Tensor:
        """Return source-only supervised MoCo-style contrastive loss.

        ``anchor_features`` are trainable student vectors from a strong view;
        ``key_features`` are no-gradient EMA-teacher vectors from its paired
        weak view.  Both have shape ``[batch, feature_dim]``.  Each paired key
        is a positive by construction, even when the micro-batch has one sample.
        """

        if (
            anchor_features.ndim != 2
            or key_features.ndim != 2
            or anchor_features.shape != key_features.shape
            or anchor_features.shape[1] != self.feature_dim
        ):
            raise ValueError(
                "anchor_features and key_features must match [batch, feature_dim]; "
                f"got {tuple(anchor_features.shape)} and {tuple(key_features.shape)}"
            )
        if not anchor_features.is_floating_point() or not key_features.is_floating_point():
            raise TypeError("contrastive features must be floating point")

        batch_size = anchor_features.shape[0]
        labels = labels.reshape(-1).to(device=anchor_features.device, dtype=torch.long)
        if labels.numel() != batch_size:
            raise ValueError(f"expected {batch_size} labels, got {labels.numel()}")
        if self.ignore_index is not None:
            keep = labels != self.ignore_index
            anchor_features = anchor_features[keep]
            key_features = key_features[keep]
            labels = labels[keep]
            batch_size = int(keep.sum().item())
        if batch_size == 0:
            return anchor_features.sum() * 0.0

        anchors = anchor_features.float()
        keys = key_features.detach().float()
        if self.normalize:
            anchors = F.normalize(anchors, dim=-1)
            keys = F.normalize(keys, dim=-1)

        queue_count = self.stored_features
        queue_features = self.queue_features[:queue_count].detach().to(
            device=anchors.device, dtype=anchors.dtype
        )
        queue_labels = self.queue_labels[:queue_count].detach().to(anchors.device)
        contrast_features = torch.cat((keys, queue_features), dim=0)
        contrast_labels = torch.cat((labels, queue_labels), dim=0)

        logits = anchors @ contrast_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_mask = labels[:, None].eq(contrast_labels[None, :]).to(
            dtype=anchors.dtype
        )
        # Anchors and keys are distinct tensors, so their paired position stays
        # a valid positive instead of being masked as a self-comparison.
        exp_logits = logits.exp()
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(exp_logits.dtype).tiny
            )
        )
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        losses = -(
            self.temperature / self.base_temperature
        ) * (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
        result = _reduce_valid_losses(
            losses, valid, "mean", reference=anchor_features
        )
        return result


class ConfidenceGatedTeacherLoss(nn.Module):
    """KL distillation from an EMA teacher with confidence and label safeguards.

    The gate is intentionally evaluated against known source labels.  This keeps
    the objective source-only: an uncertain or wrong EMA pseudo-label is ignored
    instead of reinforcing a training error.  The returned coverage is the
    fraction of samples accepted by the gate and should be logged during runs.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.5,
        confidence_threshold: float = 0.70,
        require_label_agreement: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.require_label_agreement = bool(require_label_agreement)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
            raise ValueError(
                "student_logits and teacher_logits must have matching [batch, classes] "
                f"shapes; got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )
        if labels is not None:
            labels = labels.reshape(-1).to(device=student_logits.device, dtype=torch.long)
            if labels.numel() != student_logits.shape[0]:
                raise ValueError("teacher labels do not match batch size")

        teacher_float = teacher_logits.detach().float()
        confidence_probabilities = teacher_float.softmax(dim=1)
        confidence, pseudo_labels = confidence_probabilities.max(dim=1)
        valid = confidence >= self.confidence_threshold
        if self.require_label_agreement and labels is not None:
            valid = valid & pseudo_labels.eq(labels)

        temperature = self.temperature
        teacher_log_prob = F.log_softmax(teacher_float / temperature, dim=1)
        teacher_prob = teacher_log_prob.exp()
        student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=1)
        per_sample = (
            teacher_prob * (teacher_log_prob - student_log_prob)
        ).sum(dim=1) * (temperature**2)
        loss = _reduce_valid_losses(
            per_sample,
            valid,
            "mean",
            reference=student_logits,
        )
        coverage = valid.float().mean()
        return loss, coverage


class ForegroundAttentionLoss(nn.Module):
    """Penalize spatial attention mass that is not on the leaf foreground.

    Args:
        from_logits: Apply a spatial softmax to each attention map.  Set false
            for non-negative sigmoid/probability attention maps.
        threshold: Optional threshold used to binarize the resized foreground
            mask.  ``None`` preserves a soft mask.
        ignore_empty_masks: Ignore empty masks.  When every mask is empty, a
            differentiable zero is returned.
        reduction: Reduction over batch and attention-map dimensions.
        eps: Numerical floor used when normalizing non-logit attention.

    Attention may have shape ``[B, H, W]`` or ``[B, ..., H, W]``.  All middle
    dimensions are treated as independent maps.  The mask must have shape
    ``[B, H, W]`` or ``[B, 1|maps, H, W]``.
    """

    def __init__(
        self,
        *,
        from_logits: bool = False,
        threshold: float | None = 0.5,
        ignore_empty_masks: bool = True,
        reduction: Reduction = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1] or be None")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        _validate_reduction(reduction)

        self.from_logits = bool(from_logits)
        self.threshold = threshold
        self.ignore_empty_masks = bool(ignore_empty_masks)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, attention: Tensor, foreground_mask: Tensor) -> Tensor:
        if attention.ndim < 3:
            raise ValueError("attention must have shape [B, ..., H, W]")
        if foreground_mask.ndim not in {3, 4}:
            raise ValueError("foreground_mask must have shape [B, H, W] or [B, C, H, W]")
        if attention.shape[0] != foreground_mask.shape[0]:
            raise ValueError("attention and foreground_mask batch sizes differ")
        if not attention.is_floating_point():
            raise TypeError("attention must be floating point")

        batch_size, height, width = (
            attention.shape[0],
            attention.shape[-2],
            attention.shape[-1],
        )
        maps = attention.reshape(batch_size, -1, height, width).float()
        num_maps = maps.shape[1]

        mask = foreground_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=attention.device, dtype=torch.float32)
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(mask, size=(height, width), mode="nearest")
        if mask.shape[1] == 1:
            mask = mask.expand(-1, num_maps, -1, -1)
        elif mask.shape[1] != num_maps:
            raise ValueError(
                "mask channels must be one or match flattened attention maps; "
                f"got {mask.shape[1]} and {num_maps}"
            )
        mask = mask.clamp(0.0, 1.0)
        if self.threshold is not None:
            mask = (mask >= self.threshold).to(dtype=maps.dtype)

        flat_maps = maps.flatten(2)
        flat_mask = mask.flatten(2)
        if self.from_logits:
            weights = F.softmax(flat_maps, dim=-1)
        else:
            weights = flat_maps.clamp_min(0.0)

        total_mass = weights.sum(dim=-1)
        foreground_mass = (weights * flat_mask).sum(dim=-1)
        losses = 1.0 - foreground_mass / total_mass.clamp_min(self.eps)
        losses = losses.clamp(0.0, 1.0)

        nonempty = flat_mask.sum(dim=-1) > 0
        valid = nonempty if self.ignore_empty_masks else torch.ones_like(nonempty)
        return _reduce_valid_losses(
            losses,
            valid,
            self.reduction,
            reference=attention,
        )


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    source_model: nn.Module,
    *,
    decay: float = 0.999,
    buffer_mode: BufferMode = "copy",
) -> None:
    """Update ``ema_model`` in place from ``source_model``.

    Floating-point parameters use ``ema = decay * ema + (1-decay) * source``.
    Buffers are copied by default, which is safest for BatchNorm counters.  With
    ``buffer_mode='ema'``, floating buffers are averaged and integer buffers are
    still copied.  Model structures must match exactly.
    """

    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be finite and lie in [0, 1]")
    if buffer_mode not in {"copy", "ema", "none"}:
        raise ValueError("buffer_mode must be 'copy', 'ema', or 'none'")

    ema_parameters = dict(ema_model.named_parameters())
    source_parameters = dict(source_model.named_parameters())
    if ema_parameters.keys() != source_parameters.keys():
        missing = sorted(source_parameters.keys() - ema_parameters.keys())
        extra = sorted(ema_parameters.keys() - source_parameters.keys())
        raise ValueError(
            f"model parameter structures differ; missing={missing}, extra={extra}"
        )

    for name, ema_parameter in ema_parameters.items():
        source_parameter = source_parameters[name].detach().to(
            device=ema_parameter.device, dtype=ema_parameter.dtype
        )
        if ema_parameter.is_floating_point() or ema_parameter.is_complex():
            ema_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
        else:
            ema_parameter.copy_(source_parameter)

    if buffer_mode == "none":
        return

    ema_buffers = dict(ema_model.named_buffers())
    source_buffers = dict(source_model.named_buffers())
    if ema_buffers.keys() != source_buffers.keys():
        missing = sorted(source_buffers.keys() - ema_buffers.keys())
        extra = sorted(ema_buffers.keys() - source_buffers.keys())
        raise ValueError(
            f"model buffer structures differ; missing={missing}, extra={extra}"
        )

    for name, ema_buffer in ema_buffers.items():
        source_buffer = source_buffers[name].detach().to(
            device=ema_buffer.device, dtype=ema_buffer.dtype
        )
        can_average = ema_buffer.is_floating_point() or ema_buffer.is_complex()
        if buffer_mode == "ema" and can_average:
            ema_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            ema_buffer.copy_(source_buffer)


class ModelEMA(nn.Module):
    """Evaluation-only exponential moving average copy of a model.

    ``warmup_updates`` gradually raises the effective decay from zero to the
    configured value.  The wrapper's regular ``state_dict`` includes both the
    averaged model and the update counter, so it can be checkpointed directly.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        warmup_updates: int = 0,
        device: torch.device | str | None = None,
        buffer_mode: BufferMode = "copy",
    ) -> None:
        super().__init__()
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite and lie in [0, 1)")
        if warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")
        if buffer_mode not in {"copy", "ema", "none"}:
            raise ValueError("buffer_mode must be 'copy', 'ema', or 'none'")

        self.module = copy.deepcopy(model)
        if device is not None:
            self.module.to(device)
        self.module.eval()
        self.module.requires_grad_(False)

        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.buffer_mode = buffer_mode
        self.register_buffer(
            "num_updates", torch.zeros((), dtype=torch.long), persistent=True
        )
        super().train(False)

    def train(self, mode: bool = True) -> ModelEMA:
        """Keep the averaged model in evaluation mode."""

        del mode
        super().train(False)
        self.module.eval()
        return self

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def effective_decay(self) -> float:
        """Return decay used by the next/current update schedule."""

        if self.warmup_updates == 0:
            return self.decay
        updates = max(int(self.num_updates.item()), 1)
        return self.decay * (1.0 - math.exp(-updates / self.warmup_updates))

    @torch.no_grad()
    def update(self, source_model: nn.Module) -> float:
        """Update the EMA weights and return the effective decay."""

        self.num_updates.add_(1)
        decay = self.effective_decay()
        update_ema_model(
            self.module,
            source_model,
            decay=decay,
            buffer_mode=self.buffer_mode,
        )
        self.module.eval()
        return decay

    @torch.no_grad()
    def copy_from(self, source_model: nn.Module) -> None:
        """Replace EMA weights with an exact copy of ``source_model``."""

        update_ema_model(
            self.module, source_model, decay=0.0, buffer_mode="copy"
        )


def normalize_class_name(name: str) -> str:
    """Trim and collapse whitespace without changing label semantics."""

    if not isinstance(name, str):
        raise TypeError(f"class name must be str, got {type(name).__name__}")
    normalized = " ".join(name.strip().split())
    if not normalized:
        raise ValueError("class name must not be empty")
    return normalized


def infer_plant_name(class_name: str) -> str:
    """Infer the plant component from common PlantVillage label formats.

    Supports folder-style labels such as ``Apple___Black_rot`` and display
    labels such as ``Apple Black Rot`` or ``Corn_(maize) Common Rust``.
    """

    normalized = normalize_class_name(class_name)
    if "___" in normalized:
        plant = normalized.split("___", 1)[0]
    else:
        plant = normalized.split(maxsplit=1)[0]
    plant = re.sub(r"_+", " ", plant).strip()
    if not plant:
        raise ValueError(f"could not infer plant from {class_name!r}")
    return plant


def is_healthy_label(
    class_name: str, *, healthy_tokens: Sequence[str] = ("healthy",)
) -> bool:
    """Return whether a class name contains a whole healthy-status token."""

    normalized = normalize_class_name(class_name).casefold()
    words = set(re.findall(r"[a-z0-9]+", normalized))
    tokens = {
        token.casefold().strip()
        for token in healthy_tokens
        if token and token.strip()
    }
    if not tokens:
        raise ValueError("healthy_tokens must contain at least one non-empty token")
    return bool(words & tokens)


@dataclass(frozen=True)
class HierarchicalLabelMapping:
    """Lookup tables from fine-grained class IDs to plant and health IDs.

    Health IDs are ``0 = diseased`` and ``1 = healthy``.
    """

    class_names: tuple[str, ...]
    plant_names: tuple[str, ...]
    class_to_plant: tuple[int, ...]
    class_to_health: tuple[int, ...]
    health_names: tuple[str, str] = ("diseased", "healthy")

    def __post_init__(self) -> None:
        size = len(self.class_names)
        if size == 0:
            raise ValueError("mapping must contain at least one class")
        if len(self.class_to_plant) != size or len(self.class_to_health) != size:
            raise ValueError("class lookup tables must match class_names length")
        if any(index < 0 or index >= len(self.plant_names) for index in self.class_to_plant):
            raise ValueError("class_to_plant contains an invalid plant index")
        if any(index not in {0, 1} for index in self.class_to_health):
            raise ValueError("class_to_health values must be zero or one")

    def map_targets(self, class_targets: Tensor) -> tuple[Tensor, Tensor]:
        """Map class-index tensor to plant-index and health-index tensors."""

        if class_targets.is_floating_point() or class_targets.is_complex():
            raise TypeError("class_targets must contain integer indices")
        targets = class_targets.long()
        if targets.numel() > 0:
            minimum = int(targets.min().item())
            maximum = int(targets.max().item())
            if minimum < 0 or maximum >= len(self.class_names):
                raise IndexError(
                    f"class target range [{minimum}, {maximum}] is outside "
                    f"[0, {len(self.class_names) - 1}]"
                )
        plant_lut = torch.tensor(
            self.class_to_plant, device=targets.device, dtype=torch.long
        )
        health_lut = torch.tensor(
            self.class_to_health, device=targets.device, dtype=torch.long
        )
        return plant_lut[targets], health_lut[targets]


def build_hierarchical_label_mapping(
    class_names: Sequence[str],
    *,
    plant_overrides: Mapping[str, str] | None = None,
    healthy_tokens: Sequence[str] = ("healthy",),
) -> HierarchicalLabelMapping:
    """Build deterministic class-to-plant and class-to-health lookup tables.

    ``plant_overrides`` may use either the original or normalized class name as
    a key.  Plant IDs are assigned by case-insensitive alphabetical order.
    """

    normalized_names = tuple(normalize_class_name(name) for name in class_names)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("class_names contain duplicates after normalization")
    if not normalized_names:
        raise ValueError("class_names must not be empty")

    overrides = dict(plant_overrides or {})
    inferred_plants: list[str] = []
    for original, normalized in zip(class_names, normalized_names):
        plant = overrides.get(original, overrides.get(normalized))
        inferred_plants.append(
            normalize_class_name(plant) if plant is not None else infer_plant_name(normalized)
        )

    plant_names = tuple(sorted(set(inferred_plants), key=str.casefold))
    plant_to_index = {name: index for index, name in enumerate(plant_names)}
    class_to_plant = tuple(plant_to_index[name] for name in inferred_plants)
    class_to_health = tuple(
        int(is_healthy_label(name, healthy_tokens=healthy_tokens))
        for name in normalized_names
    )
    return HierarchicalLabelMapping(
        class_names=normalized_names,
        plant_names=plant_names,
        class_to_plant=class_to_plant,
        class_to_health=class_to_health,
    )


class ClassificationMetricsAccumulator:
    """Streaming confusion-matrix metrics for single-label classification.

    Rows of the confusion matrix are targets and columns are predictions.
    ``compute`` returns accuracy, macro-F1, and balanced accuracy without a
    scikit-learn dependency.  Empty target classes are excluded by default,
    matching balanced-accuracy semantics for an observed evaluation set.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ignore_index: int | None = -1,
        include_empty_classes: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.include_empty_classes = bool(include_empty_classes)
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=device,
        )

    @torch.no_grad()
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Add a batch of class IDs or logits/probabilities.

        Accepted prediction layouts are class IDs with the same shape as
        ``targets``, channel-first logits ``[B, C, ...]``, or class-last logits
        ``[..., C]``.
        """

        predictions = predictions.detach()
        targets = targets.detach()
        if predictions.shape == targets.shape:
            predicted_ids = predictions
        elif (
            predictions.ndim == targets.ndim + 1
            and predictions.shape[0] == targets.shape[0]
            and predictions.shape[1] == self.num_classes
            and predictions.shape[2:] == targets.shape[1:]
        ):
            predicted_ids = predictions.argmax(dim=1)
        elif (
            predictions.ndim == targets.ndim + 1
            and predictions.shape[-1] == self.num_classes
            and predictions.shape[:-1] == targets.shape
        ):
            predicted_ids = predictions.argmax(dim=-1)
        else:
            raise ValueError(
                "predictions must be class IDs matching targets or logits with "
                f"{self.num_classes} classes; got {tuple(predictions.shape)} and "
                f"targets {tuple(targets.shape)}"
            )

        predicted_ids = predicted_ids.reshape(-1).long().to(
            self.confusion_matrix.device
        )
        target_ids = targets.reshape(-1).long().to(self.confusion_matrix.device)
        keep = torch.ones_like(target_ids, dtype=torch.bool)
        if self.ignore_index is not None:
            keep &= target_ids != self.ignore_index
        predicted_ids = predicted_ids[keep]
        target_ids = target_ids[keep]
        if target_ids.numel() == 0:
            return

        if (
            target_ids.min() < 0
            or target_ids.max() >= self.num_classes
            or predicted_ids.min() < 0
            or predicted_ids.max() >= self.num_classes
        ):
            raise ValueError("target or prediction contains an out-of-range class ID")

        encoded = target_ids * self.num_classes + predicted_ids
        counts = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix.add_(counts)

    def reset(self) -> None:
        """Clear accumulated counts."""

        self.confusion_matrix.zero_()

    def compute(self) -> dict[str, float]:
        """Compute scalar metrics from accumulated counts."""

        matrix = self.confusion_matrix.to(dtype=torch.float64)
        true_positive = matrix.diag()
        target_support = matrix.sum(dim=1)
        predicted_support = matrix.sum(dim=0)
        total = matrix.sum()

        precision = true_positive / predicted_support.clamp_min(1.0)
        recall = true_positive / target_support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(
            torch.finfo(matrix.dtype).eps
        )

        if self.include_empty_classes:
            included = torch.ones(
                self.num_classes, dtype=torch.bool, device=matrix.device
            )
        else:
            included = target_support > 0

        if bool(included.any()):
            macro_f1 = f1[included].mean()
            balanced_accuracy = recall[included].mean()
        else:
            macro_f1 = matrix.new_zeros(())
            balanced_accuracy = matrix.new_zeros(())
        accuracy = true_positive.sum() / total.clamp_min(1.0)

        return {
            "accuracy": float(accuracy.item()),
            "macro_f1": float(macro_f1.item()),
            "balanced_accuracy": float(balanced_accuracy.item()),
        }

    def per_class(self) -> dict[str, Tensor]:
        """Return support, precision, recall, and F1 tensors per class."""

        matrix = self.confusion_matrix.to(dtype=torch.float64)
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(
            torch.finfo(matrix.dtype).eps
        )
        return {
            "support": support.clone(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def merge_(self, other: ClassificationMetricsAccumulator) -> None:
        """Merge counts from a compatible accumulator."""

        if self.num_classes != other.num_classes:
            raise ValueError("cannot merge accumulators with different class counts")
        self.confusion_matrix.add_(
            other.confusion_matrix.to(self.confusion_matrix.device)
        )

    @torch.no_grad()
    def synchronize_between_processes(self) -> None:
        """All-reduce counts when torch.distributed is initialized."""

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.confusion_matrix, op=dist.ReduceOp.SUM)

    def state_dict(self) -> dict[str, Tensor]:
        """Return a checkpoint-safe state dictionary."""

        return {"confusion_matrix": self.confusion_matrix.clone()}

    def load_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Restore accumulated counts with validation."""

        if "confusion_matrix" not in state_dict:
            raise KeyError("state_dict is missing 'confusion_matrix'")
        matrix = state_dict["confusion_matrix"]
        expected_shape = (self.num_classes, self.num_classes)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"expected confusion matrix shape {expected_shape}, got {tuple(matrix.shape)}"
            )
        self.confusion_matrix.copy_(
            matrix.to(device=self.confusion_matrix.device, dtype=torch.int64)
        )
