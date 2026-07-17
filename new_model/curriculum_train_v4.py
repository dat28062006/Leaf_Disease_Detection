"""Source-only domain-generalization training for :mod:`final_model_v4`.

This trainer deliberately uses only PlantVillage ``train.csv``.  It never
imports, reads, normalizes from, pseudo-labels, or selects checkpoints with
PlantDoc/test data.  PlantDoc evaluation belongs in a separate command after
all hyperparameters and the source-DG checkpoint have been frozen.

The default experiment trains on four synthetic PlantVillage domains:

* D0: ordinary source views;
* D1: illumination/colour changes;
* D2: camera/compression changes;
* D3: geometry and mask-aware background randomization.

D4 mask corruption is kept as an internal proxy test.  Checkpoint selection
uses a clean PlantVillage validation partition plus D1--D3 transformations of
a disjoint OOD-validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Avoid network version checks inside DataLoader workers.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:  # Support both ``python file.py`` and package-style imports.
    from .final_model_v4 import FinalModelV4, PINNED_DINOV2_HUB_REPO
    from .source_dg_data import (
        MANIFEST_ALGORITHM_VERSION,
        PROXY_DOMAINS,
        SourceDGDataset,
        build_source_splits,
        seed_worker,
    )
    from .source_dg_losses import (
        ClassificationMetricsAccumulator,
        ForegroundAttentionLoss,
        ModelEMA,
        SupervisedContrastiveLoss,
        WeakStrongConsistencyLoss,
    )
except ImportError:
    from final_model_v4 import FinalModelV4, PINNED_DINOV2_HUB_REPO
    from source_dg_data import (
        MANIFEST_ALGORITHM_VERSION,
        PROXY_DOMAINS,
        SourceDGDataset,
        build_source_splits,
        seed_worker,
    )
    from source_dg_losses import (
        ClassificationMetricsAccumulator,
        ForegroundAttentionLoss,
        ModelEMA,
        SupervisedContrastiveLoss,
        WeakStrongConsistencyLoss,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class TrainConfig:
    """Configuration persisted in every V4 checkpoint."""

    project_root: str = str(PROJECT_ROOT)
    train_csv: str = str(PROJECT_ROOT / "train.csv")
    output_dir: str = str(PROJECT_ROOT / "new_model" / "checkpoints_source_dg_v4")
    manifest_path: str = ""

    seed: int = 2026
    epochs: int = 50
    image_size: int = 336
    batch_size: int = 2
    validation_batch_size: int = 4
    gradient_accumulation: int = 8
    num_workers: int = 4

    model_name: str = "dinov2_vits14"
    feature_dim: int = 512
    projection_dim: int = 128
    use_detail_branch: bool = False
    crop_leaf_by_mask: bool = True
    leaf_crop_margin: float = 0.10

    train_domains: tuple[str, ...] = ("D0", "D1", "D2", "D3")
    selection_domains: tuple[str, ...] = ("D1", "D2", "D3")
    proxy_test_domain: str = "D4"
    background_donor_probability: float = 0.75
    balance_fraction: float = 0.50
    hash_exact_duplicates: bool = True

    head_lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    plant_loss_weight: float = 0.30
    health_loss_weight: float = 0.20
    plant_hierarchy_logit_weight: float = 0.35
    health_hierarchy_logit_weight: float = 0.15
    consistency_loss_weight: float = 0.30
    consistency_ramp_epochs: int = 5
    supcon_loss_weight: float = 0.05
    supcon_start_epoch: int = 5
    attention_loss_weight: float = 0.05
    max_grad_norm: float = 1.0

    ema_decay: float = 0.999
    ema_warmup_updates: int = 100
    robust_eval_interval: int = 5

    amp: bool = True
    deterministic: bool = False
    resume: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    def __post_init__(self) -> None:
        self.train_domains = tuple(self.train_domains)
        self.selection_domains = tuple(self.selection_domains)
        if not self.manifest_path:
            self.manifest_path = str(
                Path(self.output_dir) / f"source_manifest_seed{self.seed}.csv"
            )
        unknown = (set(self.train_domains) | set(self.selection_domains)) - set(
            PROXY_DOMAINS
        )
        if self.proxy_test_domain not in PROXY_DOMAINS:
            unknown.add(self.proxy_test_domain)
        if unknown:
            raise ValueError(f"Unknown proxy domains: {sorted(unknown)}")
        if not self.train_domains or not self.selection_domains:
            raise ValueError("train_domains and selection_domains cannot be empty")
        if self.proxy_test_domain in set(self.train_domains) | set(
            self.selection_domains
        ):
            raise ValueError(
                "proxy_test_domain must be disjoint from train/selection domains"
            )
        if not 0.0 <= self.balance_fraction <= 1.0:
            raise ValueError("balance_fraction must lie in [0, 1]")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch sizes must be positive")
        if self.gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.robust_eval_interval <= 0:
            raise ValueError("robust_eval_interval must be positive")
        if not 0.0 <= self.leaf_crop_margin <= 0.5:
            raise ValueError("leaf_crop_margin must be in [0, 0.5]")
        if (
            self.plant_hierarchy_logit_weight < 0.0
            or self.health_hierarchy_logit_weight < 0.0
        ):
            raise ValueError("hierarchy logit weights must be non-negative")


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    name: str
    start: int
    end: int
    unfrozen_blocks: int
    head_lr_factor: float
    backbone_lr_factor: float


PHASES: tuple[CurriculumPhase, ...] = (
    CurriculumPhase("head_warmup", 0, 5, 0, 1.00, 0.00),
    CurriculumPhase("background_invariance", 5, 15, 1, 0.70, 1.00),
    CurriculumPhase("domain_randomization", 15, 35, 2, 0.40, 0.50),
    CurriculumPhase("hard_generalization", 35, 45, 4, 0.20, 0.25),
    CurriculumPhase("consolidation", 45, 10**9, 2, 0.10, 0.10),
)


def phase_for_epoch(epoch: int) -> CurriculumPhase:
    for phase in PHASES:
        if phase.start <= epoch < phase.end:
            return phase
    raise RuntimeError(f"No curriculum phase covers epoch {epoch}")


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except AttributeError:
        pass


def _jsonable_config(config: TrainConfig) -> dict[str, Any]:
    result = asdict(config)
    result["train_domains"] = list(config.train_domains)
    result["selection_domains"] = list(config.selection_domains)
    return result


def _model_config(
    config: TrainConfig,
    *,
    num_disease_classes: int,
    num_plant_classes: int,
    class_to_plant: Sequence[int],
    class_to_health: Sequence[int],
) -> dict[str, Any]:
    """Persist every architecture argument needed for strict reconstruction."""

    return {
        "num_disease_classes": int(num_disease_classes),
        "num_plant_classes": int(num_plant_classes),
        "num_health_classes": 2,
        "model_name": config.model_name,
        "num_intermediate_layers": 4,
        "feature_dim": config.feature_dim,
        "projection_dim": config.projection_dim,
        "projection_hidden_dim": 256,
        "topk_ratio": 0.15,
        "dropout": 0.20,
        "group_norm_groups": 32,
        "use_detail_branch": config.use_detail_branch,
        "detail_dim": 128,
        "use_cosine_heads": True,
        "leaf_background_value": 0.0,
        "crop_leaf_by_mask": config.crop_leaf_by_mask,
        "leaf_crop_margin": config.leaf_crop_margin,
        "class_to_plant": list(class_to_plant),
        "class_to_health": list(class_to_health),
        "plant_hierarchy_logit_weight": config.plant_hierarchy_logit_weight,
        "health_hierarchy_logit_weight": config.health_hierarchy_logit_weight,
        "hub_repo": PINNED_DINOV2_HUB_REPO,
    }


def _file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".meta.json")


def _validate_source_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "image_rel",
        "plant_disease",
        "plant_name",
        "disease_type_name",
        "class_index",
        "plant_index",
        "disease_type_index",
        "health_index",
        "group_id",
        "split",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    if manifest.empty or manifest["image_rel"].duplicated().any():
        raise ValueError("Source manifest is empty or contains duplicate image paths")
    safe_paths = manifest["image_rel"].astype(str).str.replace("\\", "/", regex=False)
    if not safe_paths.str.casefold().str.startswith("train/").all():
        raise ValueError("Source manifest contains a path outside Train/")
    expected_splits = {"train", "id_val", "ood_val", "proxy_test"}
    actual_splits = set(manifest["split"].astype(str))
    if actual_splits != expected_splits:
        raise ValueError(
            f"Source manifest split mismatch: {sorted(actual_splits)}"
        )
    if int(manifest.groupby("group_id")["split"].nunique().max()) != 1:
        raise ValueError("A capture/exact group crosses source partitions")
    all_classes = set(manifest["class_index"].astype(int))
    for split_name, split_frame in manifest.groupby("split"):
        if set(split_frame["class_index"].astype(int)) != all_classes:
            raise ValueError(f"Split {split_name!r} does not contain every class")


def load_or_create_manifest(config: TrainConfig) -> pd.DataFrame:
    """Load a matching split manifest or create one exclusively from train.csv."""

    csv_path = Path(config.train_csv).resolve()
    if csv_path.name.casefold().startswith("test"):
        raise ValueError("V4 source-only training refuses test CSV files")
    manifest_path = Path(config.manifest_path).resolve()
    metadata_path = _manifest_metadata_path(manifest_path)
    source_digest = _file_digest(csv_path)

    if manifest_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "source_csv_digest": source_digest,
            "seed": config.seed,
            "hash_exact_duplicates": config.hash_exact_duplicates,
            "manifest_algorithm_version": MANIFEST_ALGORITHM_VERSION,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            manifest = pd.read_csv(manifest_path)
            _validate_source_manifest(manifest)
            return manifest

    print("Creating deterministic PlantVillage source-only split manifest...")
    manifest = build_source_splits(
        csv_path,
        project_root=config.project_root,
        seed=config.seed,
        hash_exact_duplicates=config.hash_exact_duplicates,
    )
    _validate_source_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)
    metadata = {
        "source_csv": str(csv_path),
        "source_csv_digest": source_digest,
        "seed": config.seed,
        "hash_exact_duplicates": config.hash_exact_duplicates,
        "manifest_algorithm_version": MANIFEST_ALGORITHM_VERSION,
        "rows": len(manifest),
        "split_counts": {
            str(key): int(value)
            for key, value in manifest["split"].value_counts().items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _ordered_names(frame: pd.DataFrame, name_column: str, index_column: str) -> list[str]:
    pairs = frame[[name_column, index_column]].drop_duplicates()
    if pairs[index_column].duplicated().any():
        raise ValueError(f"{index_column} maps to multiple {name_column} values")
    pairs = pairs.sort_values(index_column)
    expected = list(range(len(pairs)))
    actual = pairs[index_column].astype(int).tolist()
    if actual != expected:
        raise ValueError(f"{index_column} is not contiguous: {actual}")
    return pairs[name_column].astype(str).tolist()


def class_and_plant_names(manifest: pd.DataFrame) -> tuple[list[str], list[str]]:
    return (
        _ordered_names(manifest, "plant_disease", "class_index"),
        _ordered_names(manifest, "plant_name", "plant_index"),
    )


def class_hierarchy(manifest: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return checkpoint-stable class -> plant/health lookup tables."""

    rows = (
        manifest[["class_index", "plant_index", "health_index"]]
        .drop_duplicates()
        .sort_values("class_index")
    )
    if rows["class_index"].duplicated().any():
        raise ValueError("A disease class maps to multiple hierarchy labels")
    expected = list(range(len(rows)))
    if rows["class_index"].astype(int).tolist() != expected:
        raise ValueError("Disease class indices are not contiguous")
    return (
        rows["plant_index"].astype(int).tolist(),
        rows["health_index"].astype(int).tolist(),
    )


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            _set_dataset_epoch(child, epoch)


def hybrid_sample_weights(labels: Sequence[int], balance_fraction: float) -> Tensor:
    """Mixture of natural and class-uniform sampling probabilities.

    This is one sampling mechanism, not inverse weighting layered on top of
    focal loss.  ``balance_fraction=0`` is natural sampling and ``1`` is fully
    class-uniform sampling.
    """

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or not len(label_array):
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    classes, counts = np.unique(label_array, return_counts=True)
    count_lookup = {int(label): int(count) for label, count in zip(classes, counts)}
    total = float(len(label_array))
    num_classes = float(len(classes))
    weights = [
        (1.0 - balance_fraction) / total
        + balance_fraction / (num_classes * count_lookup[int(label)])
        for label in label_array
    ]
    return torch.as_tensor(weights, dtype=torch.double)


def curriculum_domain_mix(phase: CurriculumPhase) -> dict[str, float]:
    """Ramp synthetic-domain difficulty instead of exposing all shifts at epoch 0."""

    if phase.name == "head_warmup":
        return {"D0": 0.70, "D1": 0.30, "D2": 0.00, "D3": 0.00}
    if phase.name == "background_invariance":
        return {"D0": 0.35, "D1": 0.25, "D2": 0.15, "D3": 0.25}
    return {"D0": 0.25, "D1": 0.25, "D2": 0.25, "D3": 0.25}


def set_curriculum_sampler_mix(
    sampler: WeightedRandomSampler,
    train_dataset: ConcatDataset,
    config: TrainConfig,
    phase: CurriculumPhase,
) -> dict[str, float]:
    if len(train_dataset.datasets) != len(config.train_domains):
        raise RuntimeError("Train-domain datasets and config are out of sync")
    first_frame = train_dataset.datasets[0].frame
    base = hybrid_sample_weights(
        first_frame["class_index"].astype(int).tolist(), config.balance_fraction
    )
    requested = curriculum_domain_mix(phase)
    raw = np.asarray(
        [requested.get(domain, 0.0) for domain in config.train_domains],
        dtype=np.float64,
    )
    if float(raw.sum()) <= 0.0:
        raw = np.ones(len(config.train_domains), dtype=np.float64)
    probabilities = raw / raw.sum()
    sampler.weights = torch.cat(
        [base * float(probability) for probability in probabilities]
    )
    return {
        domain: float(probability)
        for domain, probability in zip(config.train_domains, probabilities)
    }


@dataclass(slots=True)
class LoaderBundle:
    train: DataLoader
    train_sampler: WeightedRandomSampler
    id_validation: DataLoader
    ood_validation: dict[str, DataLoader]
    proxy_test: DataLoader
    train_dataset: ConcatDataset


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def build_loaders(config: TrainConfig, manifest: pd.DataFrame) -> LoaderBundle:
    common = dict(
        project_root=config.project_root,
        image_size=config.image_size,
        seed=config.seed,
        missing_mask_policy="error",
        background_donor_probability=config.background_donor_probability,
        leaf_crop_margin=config.leaf_crop_margin,
    )
    train_children = [
        SourceDGDataset(
            manifest,
            partition="train",
            proxy_domain=domain,
            training=True,
            **common,
        )
        for domain in config.train_domains
    ]
    train_dataset = ConcatDataset(train_children)
    train_frame = train_children[0].frame
    base_weights = hybrid_sample_weights(
        train_frame["class_index"].astype(int).tolist(), config.balance_fraction
    )
    weights = base_weights.repeat(len(train_children))
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_frame),
        replacement=True,
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = _loader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=pin_memory,
    )
    id_dataset = SourceDGDataset(
        manifest,
        partition="id_val",
        proxy_domain="D0",
        training=False,
        **common,
    )
    id_loader = _loader(
        id_dataset,
        batch_size=config.validation_batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    ood_loaders: dict[str, DataLoader] = {}
    for domain in config.selection_domains:
        dataset = SourceDGDataset(
            manifest,
            partition="ood_val",
            proxy_domain=domain,
            training=False,
            **common,
        )
        ood_loaders[domain] = _loader(
            dataset,
            batch_size=config.validation_batch_size,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
    proxy_dataset = SourceDGDataset(
        manifest,
        partition="proxy_test",
        proxy_domain=config.proxy_test_domain,
        training=False,
        **common,
    )
    proxy_loader = _loader(
        proxy_dataset,
        batch_size=config.validation_batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return LoaderBundle(
        train=train_loader,
        train_sampler=sampler,
        id_validation=id_loader,
        ood_validation=ood_loaders,
        proxy_test=proxy_loader,
        train_dataset=train_dataset,
    )


def _split_weight_decay_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    head_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []
    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "logit_scale" in name
        if is_backbone and no_decay:
            backbone_no_decay.append(parameter)
        elif is_backbone:
            backbone_decay.append(parameter)
        elif no_decay:
            head_no_decay.append(parameter)
        else:
            head_decay.append(parameter)
    return head_decay, head_no_decay, backbone_decay, backbone_no_decay


def build_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    """Build stable groups containing every parameter from the start.

    Frozen DINO parameters remain in the optimizer without gradients.  Later
    unfreezing only changes ``requires_grad`` and therefore cannot create the
    V3 resume-time param-group mismatch.
    """

    head_decay, head_no_decay, backbone_decay, backbone_no_decay = (
        _split_weight_decay_parameters(model)
    )
    groups = [
        {
            "params": head_decay,
            "lr": config.head_lr,
            "weight_decay": config.weight_decay,
            "role": "head",
        },
        {
            "params": head_no_decay,
            "lr": config.head_lr,
            "weight_decay": 0.0,
            "role": "head",
        },
        {
            "params": backbone_decay,
            "lr": 0.0,
            "weight_decay": config.weight_decay,
            "role": "backbone",
        },
        {
            "params": backbone_no_decay,
            "lr": 0.0,
            "weight_decay": 0.0,
            "role": "backbone",
        },
    ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("An optimizer parameter group is unexpectedly empty")
    return AdamW(groups)


def apply_curriculum_phase(
    model: FinalModelV4,
    optimizer: AdamW,
    epoch: int,
    config: TrainConfig,
) -> CurriculumPhase:
    phase = phase_for_epoch(epoch)
    model.backbone.freeze_all()
    if phase.unfrozen_blocks:
        model.unfreeze_last_blocks(phase.unfrozen_blocks, unfreeze_norm=True)
    for group in optimizer.param_groups:
        role = group.get("role")
        if role == "head":
            group["lr"] = config.head_lr * phase.head_lr_factor
        elif role == "backbone":
            group["lr"] = config.backbone_lr * phase.backbone_lr_factor
        else:
            raise RuntimeError(f"Optimizer group has invalid role: {role!r}")
    return phase


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def _autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled,
    )


@dataclass(slots=True)
class LossBundle:
    consistency: WeakStrongConsistencyLoss = field(
        default_factory=lambda: WeakStrongConsistencyLoss(
            divergence="js", teacher="weak", detach_teacher=True
        )
    )
    supervised_contrastive: SupervisedContrastiveLoss = field(
        default_factory=lambda: SupervisedContrastiveLoss(temperature=0.1)
    )
    foreground_attention: ForegroundAttentionLoss = field(
        default_factory=lambda: ForegroundAttentionLoss(
            from_logits=False, threshold=0.5, ignore_empty_masks=True
        )
    )

    def to(self, device: torch.device) -> "LossBundle":
        self.consistency.to(device)
        self.supervised_contrastive.to(device)
        self.foreground_attention.to(device)
        return self


def _classification_loss(
    output: Mapping[str, Tensor],
    labels: Tensor,
    plant_labels: Tensor,
    health_labels: Tensor,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    disease = F.cross_entropy(
        output["disease_logits"], labels, label_smoothing=config.label_smoothing
    )
    plant = F.cross_entropy(
        output["plant_logits"],
        plant_labels,
        label_smoothing=config.label_smoothing,
    )
    health = F.cross_entropy(
        output["health_logits"],
        health_labels,
        label_smoothing=config.label_smoothing,
    )
    total = disease + config.plant_loss_weight * plant + config.health_loss_weight * health
    return total, {"disease": disease, "plant": plant, "health": health}


def _consistency_weight(config: TrainConfig, epoch: int, progress: float) -> float:
    if config.consistency_ramp_epochs <= 0:
        return config.consistency_loss_weight
    completed = epoch + progress
    ramp = min(1.0, max(0.0, completed / config.consistency_ramp_epochs))
    return config.consistency_loss_weight * ramp


def _strong_classification_weight(epoch: int) -> float:
    phase = phase_for_epoch(epoch)
    if phase.name == "head_warmup":
        return 0.25
    if phase.name == "background_invariance":
        return 0.40
    return 0.50


def train_one_epoch(
    model: FinalModelV4,
    ema: ModelEMA,
    loader: DataLoader,
    sampler: WeightedRandomSampler,
    optimizer: AdamW,
    scaler: GradScaler,
    losses: LossBundle,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    _set_dataset_epoch(loader.dataset, epoch)
    if sampler.generator is not None:
        sampler.generator.manual_seed(config.seed + epoch)

    totals: dict[str, float] = {
        "loss": 0.0,
        "classification": 0.0,
        "disease": 0.0,
        "plant": 0.0,
        "health": 0.0,
        "consistency": 0.0,
        "supcon": 0.0,
        "attention": 0.0,
    }
    optimizer.zero_grad(set_to_none=True)
    processed = 0
    optimizer_steps = 0
    total_batches = len(loader)
    if config.max_train_batches is not None:
        total_batches = min(total_batches, config.max_train_batches)

    progress_bar = tqdm(
        loader,
        total=total_batches,
        desc=f"Train {epoch + 1}/{config.epochs}",
        leave=False,
    )
    for batch_index, raw_batch in enumerate(progress_bar):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        batch = _to_device(raw_batch, device)
        progress = batch_index / max(total_batches, 1)
        consistency_weight = _consistency_weight(config, epoch, progress)
        strong_classification_weight = _strong_classification_weight(epoch)
        supcon_weight = (
            config.supcon_loss_weight if epoch >= config.supcon_start_epoch else 0.0
        )

        with _autocast_context(device, config.amp and device.type in {"cuda", "cpu"}):
            weak_leaf_kwargs = (
                {
                    "leaf_image": batch["leaf_image_weak"],
                    "leaf_mask": batch["leaf_mask_weak"],
                }
                if config.crop_leaf_by_mask
                else {}
            )
            strong_leaf_kwargs = (
                {
                    "leaf_image": batch["leaf_image_strong"],
                    "leaf_mask": batch["leaf_mask_strong"],
                }
                if config.crop_leaf_by_mask
                else {}
            )
            weak = model(
                batch["image_weak"],
                batch["mask_weak"],
                return_aux=True,
                **weak_leaf_kwargs,
            )
            strong = model(
                batch["image_strong"],
                batch["mask_strong"],
                return_aux=True,
                **strong_leaf_kwargs,
            )
            weak_cls, weak_parts = _classification_loss(
                weak,
                batch["label"],
                batch["plant_label"],
                batch["health_label"],
                config,
            )
            strong_cls, strong_parts = _classification_loss(
                strong,
                batch["label"],
                batch["plant_label"],
                batch["health_label"],
                config,
            )
            classification = (
                (1.0 - strong_classification_weight) * weak_cls
                + strong_classification_weight * strong_cls
            )
            consistency = losses.consistency(
                weak["disease_logits"], strong["disease_logits"]
            )
            if supcon_weight > 0.0:
                contrastive_features = torch.stack(
                    (weak["z"], strong["z"]), dim=1
                )
                supcon = losses.supervised_contrastive(
                    contrastive_features, batch["label"]
                )
            else:
                # Avoid building the O(B^2) contrastive graph during warm-up.
                supcon = weak["z"].sum() * 0.0
            attention = 0.5 * (
                losses.foreground_attention(
                    weak["local_attention"], weak["token_mask"]
                )
                + losses.foreground_attention(
                    strong["local_attention"], strong["token_mask"]
                )
            )
            total_loss = (
                classification
                + consistency_weight * consistency
                + supcon_weight * supcon
                + config.attention_loss_weight * attention
            )
            window_start = (
                batch_index // config.gradient_accumulation
            ) * config.gradient_accumulation
            accumulation_window = min(
                config.gradient_accumulation, total_batches - window_start
            )
            scaled_loss = total_loss / accumulation_window

        scaler.scale(scaled_loss).backward()
        is_last = batch_index + 1 >= total_batches
        should_step = (
            (batch_index + 1) % config.gradient_accumulation == 0 or is_last
        )
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # GradScaler lowers its scale when non-finite gradients caused the
            # optimizer step to be skipped.  EMA must track only real updates.
            step_was_skipped = scaler.get_scale() < scale_before_step
            if not step_was_skipped:
                ema.update(model)
                optimizer_steps += 1

        batch_size = int(batch["label"].shape[0])
        processed += batch_size
        averaged_parts = {
            key: (
                (1.0 - strong_classification_weight) * weak_parts[key]
                + strong_classification_weight * strong_parts[key]
            )
            for key in weak_parts
        }
        values = {
            "loss": total_loss,
            "classification": classification,
            "disease": averaged_parts["disease"],
            "plant": averaged_parts["plant"],
            "health": averaged_parts["health"],
            "consistency": consistency,
            "supcon": supcon,
            "attention": attention,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item()) * batch_size
        progress_bar.set_postfix(
            loss=f"{float(total_loss.detach().item()):.3f}",
            cons=f"{float(consistency.detach().item()):.3f}",
        )

    if processed == 0 or optimizer_steps == 0:
        raise RuntimeError("Training epoch processed no optimizer steps")
    metrics = {key: value / processed for key, value in totals.items()}
    metrics["samples"] = float(processed)
    metrics["optimizer_steps"] = float(optimizer_steps)
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    *,
    num_disease_classes: int,
    num_plant_classes: int,
    domain_name: str,
) -> dict[str, float]:
    model.eval()
    disease_metrics = ClassificationMetricsAccumulator(num_disease_classes)
    plant_metrics = ClassificationMetricsAccumulator(num_plant_classes)
    health_metrics = ClassificationMetricsAccumulator(2)
    loss_sum = 0.0
    sample_count = 0
    total_batches = len(loader)
    if config.max_val_batches is not None:
        total_batches = min(total_batches, config.max_val_batches)

    progress_bar = tqdm(
        loader,
        total=total_batches,
        desc=f"Eval {domain_name}",
        leave=False,
    )
    for batch_index, raw_batch in enumerate(progress_bar):
        if config.max_val_batches is not None and batch_index >= config.max_val_batches:
            break
        batch = _to_device(raw_batch, device)
        with _autocast_context(device, config.amp and device.type in {"cuda", "cpu"}):
            leaf_kwargs = (
                {
                    "leaf_image": batch["leaf_image_strong"],
                    "leaf_mask": batch["leaf_mask_strong"],
                }
                if config.crop_leaf_by_mask
                else {}
            )
            output = model(
                batch["image_strong"], batch["mask_strong"], **leaf_kwargs
            )
            disease_loss = F.cross_entropy(output["disease_logits"], batch["label"])
        disease_metrics.update(output["disease_logits"].float().cpu(), batch["label"].cpu())
        plant_metrics.update(
            output["plant_logits"].float().cpu(), batch["plant_label"].cpu()
        )
        health_metrics.update(
            output["health_logits"].float().cpu(), batch["health_label"].cpu()
        )
        batch_size = int(batch["label"].shape[0])
        loss_sum += float(disease_loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(f"Evaluation domain {domain_name} had no samples")
    result = {
        "loss": loss_sum / sample_count,
        "samples": float(sample_count),
    }
    result.update(
        {f"disease_{key}": value for key, value in disease_metrics.compute().items()}
    )
    result.update(
        {f"plant_{key}": value for key, value in plant_metrics.compute().items()}
    )
    result.update(
        {f"health_{key}": value for key, value in health_metrics.compute().items()}
    )
    return result


def robust_selection_score(
    id_metrics: Mapping[str, float],
    ood_metrics: Mapping[str, Mapping[str, float]],
) -> float:
    id_score = float(id_metrics["disease_macro_f1"])
    domain_scores = [
        float(metrics["disease_macro_f1"]) for metrics in ood_metrics.values()
    ]
    if not domain_scores:
        raise ValueError("At least one OOD domain is required for robust selection")
    return 0.20 * id_score + 0.50 * float(np.mean(domain_scores)) + 0.30 * min(
        domain_scores
    )


def build_run_identity(
    config: TrainConfig,
    *,
    class_names: Sequence[str],
    plant_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_health: Sequence[int],
) -> dict[str, Any]:
    """Fingerprint split/model/training semantics while allowing epoch extension."""

    critical = _jsonable_config(config)
    for key in (
        "project_root",
        "train_csv",
        "output_dir",
        "manifest_path",
        "epochs",
        "resume",
        "num_workers",
        "validation_batch_size",
    ):
        critical.pop(key, None)
    identity_payload = {
        "manifest_algorithm_version": MANIFEST_ALGORITHM_VERSION,
        "source_csv_sha256": _file_digest(Path(config.train_csv).resolve()),
        "manifest_sha256": _file_digest(Path(config.manifest_path).resolve()),
        "critical_config": critical,
        "model_config": _model_config(
            config,
            num_disease_classes=len(class_names),
            num_plant_classes=len(plant_names),
            class_to_plant=class_to_plant,
            class_to_health=class_to_health,
        ),
        "class_names": list(class_names),
        "plant_names": list(plant_names),
        "class_to_plant": list(class_to_plant),
        "class_to_health": list(class_to_health),
    }
    serialized = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
        **identity_payload,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    best_score: float,
    model: FinalModelV4,
    ema: ModelEMA,
    optimizer: AdamW,
    scaler: GradScaler,
    config: TrainConfig,
    class_names: Sequence[str],
    plant_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_health: Sequence[int],
    run_identity: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 4,
            "protocol": "PlantVillage-only source domain generalization",
            "epoch": epoch,
            "best_robust_score": best_score,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _rng_state(),
            "config": _jsonable_config(config),
            "model_config": _model_config(
                config,
                num_disease_classes=len(class_names),
                num_plant_classes=len(plant_names),
                class_to_plant=class_to_plant,
                class_to_health=class_to_health,
            ),
            "class_names": list(class_names),
            "plant_names": list(plant_names),
            "health_names": ["diseased", "healthy"],
            "class_to_plant": list(class_to_plant),
            "class_to_health": list(class_to_health),
            "run_identity": dict(run_identity),
        },
        path,
    )


def save_best_checkpoint(
    path: Path,
    *,
    epoch: int,
    robust_score: float,
    ema: ModelEMA,
    config: TrainConfig,
    class_names: Sequence[str],
    plant_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_health: Sequence[int],
    run_identity: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 4,
            "protocol": "PlantVillage-only source domain generalization",
            "epoch": epoch,
            "robust_score": robust_score,
            "model_state_dict": ema.module.state_dict(),
            "config": _jsonable_config(config),
            "model_config": _model_config(
                config,
                num_disease_classes=len(class_names),
                num_plant_classes=len(plant_names),
                class_to_plant=class_to_plant,
                class_to_health=class_to_health,
            ),
            "class_names": list(class_names),
            "plant_names": list(plant_names),
            "health_names": ["diseased", "healthy"],
            "class_to_plant": list(class_to_plant),
            "class_to_health": list(class_to_health),
            "run_identity": dict(run_identity),
            "validation": dict(validation),
        },
        path,
    )


def append_history(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_resume(
    path: Path,
    *,
    model: FinalModelV4,
    ema: ModelEMA,
    optimizer: AdamW,
    scaler: GradScaler,
    class_names: Sequence[str],
    plant_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_health: Sequence[int],
    expected_run_identity: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, float]:
    if not path.is_file():
        return 0, -math.inf
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 4:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if list(checkpoint.get("class_names", [])) != list(class_names):
        raise ValueError("Checkpoint disease class mapping differs from manifest")
    if list(checkpoint.get("plant_names", [])) != list(plant_names):
        raise ValueError("Checkpoint plant mapping differs from manifest")
    if list(checkpoint.get("class_to_plant", [])) != list(class_to_plant):
        raise ValueError("Checkpoint class-to-plant mapping differs from manifest")
    if list(checkpoint.get("class_to_health", [])) != list(class_to_health):
        raise ValueError("Checkpoint class-to-health mapping differs from manifest")
    checkpoint_identity = checkpoint.get("run_identity", {})
    if checkpoint_identity.get("fingerprint") != expected_run_identity.get(
        "fingerprint"
    ):
        raise ValueError(
            "Resume checkpoint fingerprint differs from the requested seed, "
            "manifest, model, domains, or optimization config. Use a fresh "
            "output directory or restore the original configuration."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict", {})
    if scaler.is_enabled() and scaler_state:
        scaler.load_state_dict(scaler_state)
    _restore_rng_state(checkpoint.get("rng_state"))
    start = int(checkpoint["epoch"]) + 1
    best = float(checkpoint.get("best_robust_score", -math.inf))
    print(f"Resumed source-DG V4 at epoch {start + 1}; best score={best:.4f}")
    return start, best


def train(config: TrainConfig) -> dict[str, Any]:
    set_global_seed(config.seed, config.deterministic)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_or_create_manifest(config)
    class_names, plant_names = class_and_plant_names(manifest)
    class_to_plant, class_to_health = class_hierarchy(manifest)
    run_identity = build_run_identity(
        config,
        class_names=class_names,
        plant_names=plant_names,
        class_to_plant=class_to_plant,
        class_to_health=class_to_health,
    )
    print(
        f"PlantVillage manifest: {len(manifest):,} images, "
        f"{len(class_names)} diseases, {len(plant_names)} plants"
    )
    print("Split counts:", manifest["split"].value_counts().to_dict())
    loaders = build_loaders(config, manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; image={config.image_size}; batch={config.batch_size} "
          f"x accum={config.gradient_accumulation}")
    model = FinalModelV4(
        num_disease_classes=len(class_names),
        num_plant_classes=len(plant_names),
        num_health_classes=2,
        model_name=config.model_name,
        feature_dim=config.feature_dim,
        projection_dim=config.projection_dim,
        use_detail_branch=config.use_detail_branch,
        crop_leaf_by_mask=config.crop_leaf_by_mask,
        leaf_crop_margin=config.leaf_crop_margin,
        class_to_plant=class_to_plant,
        class_to_health=class_to_health,
        plant_hierarchy_logit_weight=config.plant_hierarchy_logit_weight,
        health_hierarchy_logit_weight=config.health_hierarchy_logit_weight,
    ).to(device)
    optimizer = build_optimizer(model, config)
    scaler = GradScaler(
        device="cuda",
        enabled=config.amp and device.type == "cuda",
        # The cosine heads can overflow with the default 2^16 scale on the
        # first mixed-precision step before GradScaler has calibrated itself.
        init_scale=4096.0,
        growth_interval=1000,
    )
    ema = ModelEMA(
        model,
        decay=config.ema_decay,
        warmup_updates=config.ema_warmup_updates,
        device=device,
        buffer_mode="copy",
    )
    losses = LossBundle().to(device)

    resume_path = output_dir / "resume_checkpoint.pth"
    best_path = output_dir / "best_source_dg_v4.pth"
    history_path = output_dir / "history.jsonl"
    if not config.resume and (resume_path.exists() or best_path.exists()):
        raise FileExistsError(
            "--no-resume refuses to mix a fresh run with existing V4 "
            "checkpoints; choose a new output directory"
        )
    start_epoch, best_score = (0, -math.inf)
    if config.resume:
        start_epoch, best_score = maybe_resume(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            class_names=class_names,
            plant_names=plant_names,
            class_to_plant=class_to_plant,
            class_to_health=class_to_health,
            expected_run_identity=run_identity,
            device=device,
        )
    if start_epoch > 0 and not best_path.is_file():
        if start_epoch >= config.epochs:
            raise FileNotFoundError(
                "Resume state exists but the selected best checkpoint is missing"
            )
        print(
            "Warning: selected best checkpoint is missing; the next robust "
            "evaluation will establish a new best."
        )
        best_score = -math.inf

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **_jsonable_config(config),
                "run_fingerprint": run_identity["fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    last_validation: dict[str, Any] = {}
    for epoch in range(start_epoch, config.epochs):
        phase = apply_curriculum_phase(model, optimizer, epoch, config)
        domain_mix = set_curriculum_sampler_mix(
            loaders.train_sampler, loaders.train_dataset, config, phase
        )
        head_lr = next(
            float(group["lr"])
            for group in optimizer.param_groups
            if group["role"] == "head"
        )
        backbone_lr = next(
            float(group["lr"])
            for group in optimizer.param_groups
            if group["role"] == "backbone"
        )
        print(
            f"\nEpoch {epoch + 1}/{config.epochs} [{phase.name}] "
            f"unfrozen={phase.unfrozen_blocks}, lr={head_lr:.2e}/{backbone_lr:.2e}; "
            "domains=" + ",".join(f"{key}:{value:.2f}" for key, value in domain_mix.items())
        )
        started = time.time()
        train_metrics = train_one_epoch(
            model,
            ema,
            loaders.train,
            loaders.train_sampler,
            optimizer,
            scaler,
            losses,
            device,
            config,
            epoch,
        )

        run_robust_eval = (
            (epoch + 1) % config.robust_eval_interval == 0
            or epoch + 1 == config.epochs
            or config.max_train_batches is not None
        )
        validation: dict[str, Any] = {}
        if run_robust_eval:
            id_metrics = evaluate(
                ema.module,
                loaders.id_validation,
                device,
                config,
                num_disease_classes=len(class_names),
                num_plant_classes=len(plant_names),
                domain_name="ID-D0",
            )
            ood_metrics = {
                domain: evaluate(
                    ema.module,
                    loader,
                    device,
                    config,
                    num_disease_classes=len(class_names),
                    num_plant_classes=len(plant_names),
                    domain_name=f"OOD-{domain}",
                )
                for domain, loader in loaders.ood_validation.items()
            }
            score = robust_selection_score(id_metrics, ood_metrics)
            validation = {"id": id_metrics, "ood": ood_metrics, "robust_score": score}
            last_validation = validation
            print(
                f"Robust score={score:.4f}; ID macro-F1="
                f"{id_metrics['disease_macro_f1']:.4f}; OOD="
                + ", ".join(
                    f"{domain}:{metrics['disease_macro_f1']:.4f}"
                    for domain, metrics in ood_metrics.items()
                )
            )
            if score > best_score:
                best_score = score
                save_best_checkpoint(
                    best_path,
                    epoch=epoch,
                    robust_score=score,
                    ema=ema,
                    config=config,
                    class_names=class_names,
                    plant_names=plant_names,
                    class_to_plant=class_to_plant,
                    class_to_health=class_to_health,
                    run_identity=run_identity,
                    validation=validation,
                )
                print(f"Saved new source-only best checkpoint: {best_path.name}")

        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            best_score=best_score,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            class_names=class_names,
            plant_names=plant_names,
            class_to_plant=class_to_plant,
            class_to_health=class_to_health,
            run_identity=run_identity,
        )
        record = {
            "epoch": epoch,
            "phase": phase.name,
            "head_lr": head_lr,
            "backbone_lr": backbone_lr,
            "domain_mix": domain_mix,
            "strong_classification_weight": _strong_classification_weight(epoch),
            "seconds": time.time() - started,
            "train": train_metrics,
            "validation": validation,
            "best_robust_score": best_score,
        }
        append_history(history_path, record)
        print(
            f"Train loss={train_metrics['loss']:.4f}; "
            f"time={record['seconds']:.1f}s"
        )

    if not best_path.is_file():
        raise RuntimeError("Training completed without a robust-evaluation checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if best_checkpoint.get("run_identity", {}).get("fingerprint") != run_identity[
        "fingerprint"
    ]:
        raise ValueError("Best checkpoint fingerprint differs from the active run")
    ema.module.load_state_dict(best_checkpoint["model_state_dict"])
    proxy_metrics = evaluate(
        ema.module,
        loaders.proxy_test,
        device,
        config,
        num_disease_classes=len(class_names),
        num_plant_classes=len(plant_names),
        domain_name=f"PROXY-{config.proxy_test_domain}",
    )
    summary = {
        "best_robust_score": float(best_checkpoint["robust_score"]),
        "best_epoch": int(best_checkpoint["epoch"]),
        "proxy_test_domain": config.proxy_test_domain,
        "proxy_test": proxy_metrics,
        "last_validation": last_validation,
        "plantdoc_used": False,
    }
    (output_dir / "source_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Internal unseen {config.proxy_test_domain}: accuracy="
        f"{proxy_metrics['disease_accuracy']:.4f}, macro-F1="
        f"{proxy_metrics['disease_macro_f1']:.4f}"
    )
    print("PlantDoc has not been read or used by this trainer.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leaf-first V4 using PlantVillage source-only DG"
    )
    parser.add_argument("--train-csv", default=str(PROJECT_ROOT / "train.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "new_model" / "checkpoints_source_dg_v4"),
    )
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument(
        "--detail-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional masked high-frequency branch (default: off; enable only "
            "after a PlantVillage proxy-OOD ablation)"
        ),
    )
    parser.add_argument(
        "--leaf-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop and enlarge the SAM leaf region before DINOv2 (default: on)",
    )
    parser.add_argument("--leaf-crop-margin", type=float, default=0.10)
    parser.add_argument("--plant-hierarchy-weight", type=float, default=0.35)
    parser.add_argument("--health-hierarchy-weight", type=float, default=0.15)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-exact-hash", action="store_true")
    parser.add_argument("--robust-eval-interval", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        project_root=str(PROJECT_ROOT),
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        seed=args.seed,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_batch_size=args.val_batch_size,
        gradient_accumulation=args.grad_accum,
        num_workers=args.workers,
        model_name=args.model_name,
        use_detail_branch=args.detail_branch,
        crop_leaf_by_mask=args.leaf_crop,
        leaf_crop_margin=args.leaf_crop_margin,
        plant_hierarchy_logit_weight=args.plant_hierarchy_weight,
        health_hierarchy_logit_weight=args.health_hierarchy_weight,
        hash_exact_duplicates=not args.skip_exact_hash,
        amp=not args.no_amp,
        deterministic=args.deterministic,
        resume=not args.no_resume,
        robust_eval_interval=args.robust_eval_interval,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    train(config_from_args(parse_args()))
