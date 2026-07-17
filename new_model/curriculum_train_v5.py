"""Source-only curriculum trainer for :class:`FinalModelV5`.

This module deliberately has no PlantDoc dependency.  It builds every split,
label mapping, augmentation domain, checkpoint-selection metric, and acceptance
report from ``train.csv`` (PlantVillage) only.  The final evaluator is the sole
place where a target-domain accuracy gate may be applied.

V5 predicts the 38-way joint plant-disease label directly from fused leaf
features.  Plant and symptom heads are auxiliary supervision only: neither the
trainer nor the serialized inference rule applies class masking, factorized
routing, logit gating, or target-domain priors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:  # Package imports.
    from .curriculum_train_v4 import (
        _autocast_context,
        _restore_rng_state,
        _rng_state,
        _set_dataset_epoch,
        append_history,
        atomic_torch_save,
        hybrid_sample_weights,
        load_or_create_manifest,
        set_global_seed,
    )
    from .final_model_v5 import FinalModelV5, PINNED_DINOV2_HUB_REPO
    from .source_dg_data import (
        MANIFEST_ALGORITHM_VERSION,
        PROXY_DOMAINS,
        SourceDGDataset,
        seed_worker,
    )
    from .source_dg_losses import (
        ClassificationMetricsAccumulator,
        ConfidenceGatedTeacherLoss,
        CrossBatchSupervisedContrastiveLoss,
        ForegroundAttentionLoss,
        ModelEMA,
        SupervisedContrastiveLoss,
        WeakStrongConsistencyLoss,
    )
except ImportError:  # Direct execution: python new_model/curriculum_train_v5.py
    from curriculum_train_v4 import (  # type: ignore
        _autocast_context,
        _restore_rng_state,
        _rng_state,
        _set_dataset_epoch,
        append_history,
        atomic_torch_save,
        hybrid_sample_weights,
        load_or_create_manifest,
        set_global_seed,
    )
    from final_model_v5 import FinalModelV5, PINNED_DINOV2_HUB_REPO  # type: ignore
    from source_dg_data import (  # type: ignore
        MANIFEST_ALGORITHM_VERSION,
        PROXY_DOMAINS,
        SourceDGDataset,
        seed_worker,
    )
    from source_dg_losses import (  # type: ignore
        ClassificationMetricsAccumulator,
        ConfidenceGatedTeacherLoss,
        CrossBatchSupervisedContrastiveLoss,
        ForegroundAttentionLoss,
        ModelEMA,
        SupervisedContrastiveLoss,
        WeakStrongConsistencyLoss,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 5
PROTOCOL = "PlantVillage-only source domain generalization V5"
INFERENCE_RULE = "direct_joint_feature_fusion"
BEST_CHECKPOINT_NAME = "best_source_dg_v5.pt"
FINAL_CHECKPOINT_NAME = "final_source_dg_v5.pt"
RESUME_CHECKPOINT_NAME = "resume_source_dg_v5.pt"
SOURCE_ACCEPTANCE_NAME = "source_acceptance_v5.json"
REQUIRED_OUTPUT_KEYS = frozenset(
    {
        "disease_logits",
        "plant_logits",
        "symptom_logits",
        "morphology_features",
        "symptom_features",
        "features",
        "z",
        "local_attention",
        "token_mask",
    }
)


@dataclass(slots=True)
class TrainConfig:
    """Training configuration persisted in every V5 checkpoint."""

    project_root: str = str(PROJECT_ROOT)
    train_csv: str = str(PROJECT_ROOT / "train.csv")
    output_dir: str = str(PROJECT_ROOT / "new_model" / "checkpoints_curriculum_v5")
    manifest_path: str = ""

    seed: int = 2026
    epochs: int = 55
    image_size: int = 336
    batch_size: int = 2
    validation_batch_size: int = 4
    gradient_accumulation: int = 8
    num_workers: int = 4

    model_name: str = "dinov2_vits14"
    trunk_dim: int = 512
    morphology_dim: int = 512
    symptom_dim: int = 512
    interaction_dim: int = 256
    projection_dim: int = 128
    shape_dim: int = 64
    use_detail_branch: bool = False
    detail_dim: int = 128
    use_cosine_heads: bool = True
    leaf_background_value: float = 0.0
    mask_erosion_kernel: int = 7
    leaf_crop_margin: float = 0.10

    # D3 is intentionally forbidden: V5 consumes an already-cropped leaf view,
    # so background replacement is neither necessary nor semantically useful.
    train_domains: tuple[str, ...] = ("D0", "D1", "D2")
    selection_domains: tuple[str, ...] = ("D1", "D2")
    proxy_test_domain: str = "D4"
    balance_fraction: float = 0.50
    hash_exact_duplicates: bool = True

    head_lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    plant_loss_weight: float = 0.25
    symptom_loss_weight: float = 0.35
    within_plant_loss_weight: float = 0.20

    disease_consistency_weight: float = 0.25
    plant_consistency_weight: float = 0.05
    symptom_consistency_weight: float = 0.10
    consistency_ramp_epochs: int = 5

    morphology_supcon_weight: float = 0.05
    symptom_supcon_weight: float = 0.05
    supcon_start_epoch: int = 5
    joint_queue_contrastive_weight: float = 0.05
    contrastive_queue_size: int = 2048
    contrastive_temperature: float = 0.10

    # EMA teacher-student regularisation is source-only: teacher predictions
    # are gated by known PlantVillage labels and supervise a paired strong view.
    ema_teacher_weight: float = 0.10
    ema_teacher_start_epoch: int = 5
    ema_teacher_ramp_epochs: int = 5
    ema_teacher_temperature: float = 1.5
    ema_teacher_confidence: float = 0.70
    attention_loss_weight: float = 0.03
    max_grad_norm: float = 1.0

    ema_decay: float = 0.999
    ema_warmup_updates: int = 100
    robust_eval_interval: int = 5
    # Diagnostics run on source-only validation data and never select a
    # checkpoint.  Keeping this separate from ``robust_eval_interval`` makes
    # the learning curves useful without silently changing model selection.
    diagnostic_eval_interval: int = 1
    diagnostic_top_k: int = 5
    diagnostic_min_support: int = 10

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

        source_csv = Path(self.train_csv)
        if source_csv.name.casefold().startswith("test"):
            raise ValueError("V5 source-only training refuses target/test CSV files")
        allowed_train_domains = {"D0", "D1", "D2"}
        unknown_train = set(self.train_domains) - allowed_train_domains
        if unknown_train:
            raise ValueError(
                "V5 leaf-only training accepts D0/D1/D2, never D3 or D4; "
                f"got {sorted(unknown_train)}"
            )
        if "D0" not in self.train_domains:
            raise ValueError("D0 must be present so every curriculum phase stays >=40% clean")
        if not self.selection_domains:
            raise ValueError("selection_domains cannot be empty")
        if set(self.selection_domains) - {"D1", "D2"}:
            raise ValueError("V5 selection domains must be a subset of D1/D2")
        if self.proxy_test_domain not in PROXY_DOMAINS:
            raise ValueError(f"Unknown source proxy domain: {self.proxy_test_domain!r}")
        if self.proxy_test_domain in set(self.train_domains) | set(
            self.selection_domains
        ):
            raise ValueError("The held-out source proxy must not select checkpoints")

        positive_ints = {
            "epochs": self.epochs,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "robust_eval_interval": self.robust_eval_interval,
            "diagnostic_eval_interval": self.diagnostic_eval_interval,
            "diagnostic_top_k": self.diagnostic_top_k,
            "diagnostic_min_support": self.diagnostic_min_support,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0.0 <= self.balance_fraction <= 1.0:
            raise ValueError("balance_fraction must lie in [0, 1]")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0, 1)")
        if not 0.0 <= self.leaf_crop_margin <= 0.5:
            raise ValueError("leaf_crop_margin must lie in [0, 0.5]")
        for name in (
            "plant_loss_weight",
            "symptom_loss_weight",
            "within_plant_loss_weight",
            "disease_consistency_weight",
            "plant_consistency_weight",
            "symptom_consistency_weight",
            "morphology_supcon_weight",
            "symptom_supcon_weight",
            "joint_queue_contrastive_weight",
            "ema_teacher_weight",
            "attention_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.contrastive_queue_size < 1:
            raise ValueError("contrastive_queue_size must be positive")
        if not math.isfinite(self.contrastive_temperature) or self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be finite and positive")
        if self.ema_teacher_start_epoch < 0 or self.ema_teacher_ramp_epochs < 0:
            raise ValueError("EMA teacher epochs cannot be negative")
        if not math.isfinite(self.ema_teacher_temperature) or self.ema_teacher_temperature <= 0:
            raise ValueError("ema_teacher_temperature must be finite and positive")
        if not 0.0 <= self.ema_teacher_confidence <= 1.0:
            raise ValueError("ema_teacher_confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    name: str
    start: int
    end: int
    unfrozen_blocks: int
    head_lr_factor: float
    backbone_lr_factor: float
    domain_mix: tuple[float, float, float]
    strong_ce_weight: float


PHASES: tuple[CurriculumPhase, ...] = (
    CurriculumPhase("head_warmup", 0, 5, 0, 1.00, 0.00, (0.70, 0.20, 0.10), 0.25),
    CurriculumPhase("leaf_invariance", 5, 15, 1, 0.70, 1.00, (0.50, 0.30, 0.20), 0.30),
    CurriculumPhase("domain_randomization", 15, 35, 2, 0.40, 0.50, (0.40, 0.30, 0.30), 0.30),
    CurriculumPhase("hard_generalization", 35, 45, 4, 0.20, 0.25, (0.40, 0.25, 0.35), 0.35),
    CurriculumPhase("consolidation", 45, 10**9, 2, 0.10, 0.10, (0.60, 0.20, 0.20), 0.25),
)


def phase_for_epoch(epoch: int) -> CurriculumPhase:
    for phase in PHASES:
        if phase.start <= epoch < phase.end:
            return phase
    raise RuntimeError(f"No V5 curriculum phase covers epoch {epoch}")


def _jsonable_config(config: TrainConfig) -> dict[str, Any]:
    result = asdict(config)
    result["train_domains"] = list(config.train_domains)
    result["selection_domains"] = list(config.selection_domains)
    return result


def _ordered_names(
    frame: pd.DataFrame, name_column: str, index_column: str
) -> list[str]:
    pairs = frame[[name_column, index_column]].drop_duplicates()
    if pairs[index_column].duplicated().any():
        raise ValueError(f"{index_column} maps to multiple {name_column} values")
    pairs = pairs.sort_values(index_column)
    actual = pairs[index_column].astype(int).tolist()
    if actual != list(range(len(pairs))):
        raise ValueError(f"{index_column} is not contiguous: {actual}")
    return pairs[name_column].astype(str).tolist()


def label_schema(
    manifest: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[int], list[int]]:
    """Return checkpoint-stable disease, plant, and symptom mappings."""

    class_names = _ordered_names(manifest, "plant_disease", "class_index")
    plant_names = _ordered_names(manifest, "plant_name", "plant_index")
    symptom_names = _ordered_names(
        manifest, "disease_type_name", "disease_type_index"
    )
    rows = (
        manifest[
            ["class_index", "plant_index", "disease_type_index"]
        ]
        .drop_duplicates()
        .sort_values("class_index")
    )
    if rows["class_index"].duplicated().any():
        raise ValueError("A disease class maps to multiple plant/symptom labels")
    if rows["class_index"].astype(int).tolist() != list(range(len(class_names))):
        raise ValueError("Disease class indices are not contiguous")
    class_to_plant = rows["plant_index"].astype(int).tolist()
    class_to_symptom = rows["disease_type_index"].astype(int).tolist()
    return (
        class_names,
        plant_names,
        symptom_names,
        class_to_plant,
        class_to_symptom,
    )


def _model_config(
    config: TrainConfig,
    *,
    num_disease_classes: int,
    num_plant_classes: int,
    num_symptom_classes: int,
) -> dict[str, Any]:
    """Persist every FinalModelV5 constructor argument for strict rebuild."""

    return {
        "num_disease_classes": int(num_disease_classes),
        "num_plant_classes": int(num_plant_classes),
        "num_symptom_classes": int(num_symptom_classes),
        "model_name": config.model_name,
        "num_intermediate_layers": 4,
        "trunk_dim": config.trunk_dim,
        "morphology_dim": config.morphology_dim,
        "symptom_dim": config.symptom_dim,
        "interaction_dim": config.interaction_dim,
        "projection_dim": config.projection_dim,
        "topk_ratio": 0.15,
        "dropout": 0.20,
        "shape_dim": config.shape_dim,
        "use_detail_branch": config.use_detail_branch,
        "detail_dim": config.detail_dim,
        "use_cosine_heads": config.use_cosine_heads,
        "leaf_background_value": config.leaf_background_value,
        "mask_erosion_kernel": config.mask_erosion_kernel,
        "hub_repo": PINNED_DINOV2_HUB_REPO,
    }


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_identity(
    config: TrainConfig,
    *,
    model_config: Mapping[str, Any],
    class_names: Sequence[str],
    plant_names: Sequence[str],
    symptom_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
) -> dict[str, Any]:
    """Fingerprint source data, code, mappings, architecture, and optimization."""

    critical = _jsonable_config(config)
    for key in (
        "project_root",
        "train_csv",
        "output_dir",
        "manifest_path",
        "epochs",
        "resume",
        "num_workers",
    ):
        critical.pop(key, None)
    dependency_hashes: dict[str, str] = {}
    for filename in (
        "curriculum_train_v5.py",
        "final_model_v5.py",
        "source_dg_data.py",
        "source_dg_losses.py",
        "curriculum_train_v4.py",
    ):
        path = Path(__file__).resolve().with_name(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Required V5 dependency is missing: {path}")
        dependency_hashes[filename] = _file_sha256(path)
    identity_payload = {
        "manifest_algorithm_version": MANIFEST_ALGORITHM_VERSION,
        "source_csv_sha256": _file_sha256(Path(config.train_csv).resolve()),
        "manifest_sha256": _file_sha256(Path(config.manifest_path).resolve()),
        "dependency_sha256": dependency_hashes,
        "critical_config": critical,
        "model_config": dict(model_config),
        "class_names": list(class_names),
        "plant_names": list(plant_names),
        "symptom_names": list(symptom_names),
        "class_to_plant": list(class_to_plant),
        "class_to_symptom": list(class_to_symptom),
        "inference_rule": INFERENCE_RULE,
    }
    serialized = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
        **identity_payload,
    }


def curriculum_domain_mix(phase: CurriculumPhase) -> dict[str, float]:
    mix = dict(zip(("D0", "D1", "D2"), phase.domain_mix))
    if not math.isclose(sum(mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Curriculum domain probabilities do not sum to one: {mix}")
    if mix["D0"] < 0.40:
        raise RuntimeError(f"V5 curriculum violated the D0 >=40% invariant: {mix}")
    if not 0.25 <= phase.strong_ce_weight <= 0.35:
        raise RuntimeError("Strong-view CE weight must stay in [0.25, 0.35]")
    return mix


def set_curriculum_sampler_mix(
    sampler: WeightedRandomSampler,
    train_dataset: ConcatDataset,
    config: TrainConfig,
    phase: CurriculumPhase,
) -> dict[str, float]:
    if len(train_dataset.datasets) != len(config.train_domains):
        raise RuntimeError("Train-domain datasets and config are out of sync")
    first_frame = train_dataset.datasets[0].frame
    base_weights = hybrid_sample_weights(
        first_frame["class_index"].astype(int).tolist(), config.balance_fraction
    )
    requested = curriculum_domain_mix(phase)
    raw = np.asarray(
        [requested.get(domain, 0.0) for domain in config.train_domains],
        dtype=np.float64,
    )
    if float(raw.sum()) <= 0.0:
        raise RuntimeError("Curriculum assigned no mass to configured domains")
    probabilities = raw / raw.sum()
    if float(probabilities[config.train_domains.index("D0")]) < 0.40 - 1e-9:
        raise RuntimeError("Normalized D0 sampling probability fell below 40%")
    sampler.weights = torch.cat(
        [base_weights * float(probability) for probability in probabilities]
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
    paired_ood_clean_validation: DataLoader
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
        small_mask_policy="keep",
        background_donor_probability=0.0,
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
    initial_mix = curriculum_domain_mix(PHASES[0])
    weights = torch.cat(
        [base_weights * initial_mix[domain] for domain in config.train_domains]
    )
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
    # This loader has the same rows as D1/D2 but uses a clean D0 view.  It is
    # diagnostic-only and lets us attribute D1/D2 changes to augmentation
    # shift rather than to a different validation partition.
    paired_ood_clean_dataset = SourceDGDataset(
        manifest,
        partition="ood_val",
        proxy_domain="D0",
        training=False,
        **common,
    )
    paired_ood_clean_loader = _loader(
        paired_ood_clean_dataset,
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
        paired_ood_clean_validation=paired_ood_clean_loader,
        ood_validation=ood_loaders,
        proxy_test=proxy_loader,
        train_dataset=train_dataset,
    )


def _split_weight_decay_parameters(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in named_parameters:
        if parameter.ndim <= 1 or name.endswith(".bias") or "logit_scale" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def build_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    backbone_named: list[tuple[str, nn.Parameter]] = []
    head_named: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        (backbone_named if name.startswith("backbone.") else head_named).append(
            (name, parameter)
        )
    head_decay, head_no_decay = _split_weight_decay_parameters(head_named)
    backbone_decay, backbone_no_decay = _split_weight_decay_parameters(backbone_named)
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
    model: FinalModelV5,
    optimizer: AdamW,
    epoch: int,
    config: TrainConfig,
) -> CurriculumPhase:
    phase = phase_for_epoch(epoch)
    if not hasattr(model.backbone, "freeze_all"):
        raise AttributeError("FinalModelV5.backbone must expose freeze_all()")
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
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _assert_output_contract(
    output: Mapping[str, Tensor],
    *,
    batch_size: int,
    num_disease_classes: int,
    num_plant_classes: int,
    num_symptom_classes: int,
) -> None:
    missing = REQUIRED_OUTPUT_KEYS.difference(output)
    if missing:
        raise KeyError(f"FinalModelV5 output lacks keys: {sorted(missing)}")
    expected_logits = {
        "disease_logits": num_disease_classes,
        "plant_logits": num_plant_classes,
        "symptom_logits": num_symptom_classes,
    }
    for key, classes in expected_logits.items():
        if output[key].shape != (batch_size, classes):
            raise ValueError(
                f"{key} shape must be {(batch_size, classes)}, "
                f"got {tuple(output[key].shape)}"
            )
    for key in ("morphology_features", "symptom_features", "features", "z"):
        if output[key].ndim != 2 or output[key].shape[0] != batch_size:
            raise ValueError(f"{key} must have shape [batch, features]")


@dataclass(slots=True)
class LossBundle:
    """Training-only objectives, including resume-safe contrastive queues."""

    consistency: WeakStrongConsistencyLoss
    morphology_supcon: SupervisedContrastiveLoss
    symptom_supcon: SupervisedContrastiveLoss
    joint_queue_contrastive: CrossBatchSupervisedContrastiveLoss
    ema_teacher: ConfidenceGatedTeacherLoss
    foreground_attention: ForegroundAttentionLoss

    @classmethod
    def from_config(cls, config: TrainConfig) -> "LossBundle":
        return cls(
            consistency=WeakStrongConsistencyLoss(
                divergence="js", teacher="weak", detach_teacher=True
            ),
            morphology_supcon=SupervisedContrastiveLoss(
                temperature=config.contrastive_temperature
            ),
            symptom_supcon=SupervisedContrastiveLoss(
                temperature=config.contrastive_temperature
            ),
            joint_queue_contrastive=CrossBatchSupervisedContrastiveLoss(
                feature_dim=config.projection_dim,
                queue_size=config.contrastive_queue_size,
                temperature=config.contrastive_temperature,
            ),
            ema_teacher=ConfidenceGatedTeacherLoss(
                temperature=config.ema_teacher_temperature,
                confidence_threshold=config.ema_teacher_confidence,
                require_label_agreement=True,
            ),
            foreground_attention=ForegroundAttentionLoss(
                from_logits=False, threshold=0.5, ignore_empty_masks=True
            ),
        )

    def to(self, device: torch.device) -> "LossBundle":
        for module in (
            self.consistency,
            self.morphology_supcon,
            self.symptom_supcon,
            self.joint_queue_contrastive,
            self.ema_teacher,
            self.foreground_attention,
        ):
            module.to(device)
        return self

    def state_dict(self) -> dict[str, Mapping[str, Tensor]]:
        """Serialize only trainer-side loss state required for exact resume."""

        return {
            "joint_queue_contrastive": self.joint_queue_contrastive.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        required = {"joint_queue_contrastive"}
        missing = required.difference(state_dict)
        if missing:
            raise KeyError(f"Loss state is missing keys: {sorted(missing)}")
        self.joint_queue_contrastive.load_state_dict(
            state_dict["joint_queue_contrastive"], strict=True
        )


def within_true_plant_ce(
    disease_logits: Tensor,
    disease_labels: Tensor,
    plant_labels: Tensor,
    class_to_plant: Tensor,
) -> Tensor:
    """Discriminate sibling diseases without changing full-space inference."""

    if class_to_plant.ndim != 1 or class_to_plant.numel() != disease_logits.shape[1]:
        raise ValueError("class_to_plant lookup does not match disease logits")
    allowed = class_to_plant.view(1, -1).eq(plant_labels.view(-1, 1))
    if not bool(allowed.gather(1, disease_labels.view(-1, 1)).all()):
        raise ValueError("Disease/plant labels disagree with class_to_plant")
    # finfo.min keeps AMP finite.  No label smoothing is used because smoothing
    # would intentionally allocate target mass to forbidden sibling groups.
    masked_logits = disease_logits.masked_fill(
        ~allowed, torch.finfo(disease_logits.dtype).min
    )
    return F.cross_entropy(masked_logits, disease_labels)


def _classification_loss(
    output: Mapping[str, Tensor],
    disease_labels: Tensor,
    plant_labels: Tensor,
    symptom_labels: Tensor,
    class_to_plant: Tensor,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    disease = F.cross_entropy(
        output["disease_logits"],
        disease_labels,
        label_smoothing=config.label_smoothing,
    )
    plant = F.cross_entropy(
        output["plant_logits"],
        plant_labels,
        label_smoothing=config.label_smoothing,
    )
    symptom = F.cross_entropy(
        output["symptom_logits"],
        symptom_labels,
        label_smoothing=config.label_smoothing,
    )
    within_plant = within_true_plant_ce(
        output["disease_logits"],
        disease_labels,
        plant_labels,
        class_to_plant,
    )
    total = (
        disease
        + config.plant_loss_weight * plant
        + config.symptom_loss_weight * symptom
        + config.within_plant_loss_weight * within_plant
    )
    return total, {
        "disease": disease,
        "plant": plant,
        "symptom": symptom,
        "within_plant": within_plant,
    }


def _consistency_ramp(config: TrainConfig, epoch: int, progress: float) -> float:
    if config.consistency_ramp_epochs <= 0:
        return 1.0
    completed = epoch + progress
    return min(1.0, max(0.0, completed / config.consistency_ramp_epochs))


def _ema_teacher_ramp(config: TrainConfig, epoch: int, progress: float) -> float:
    """Delay EMA distillation until the teacher has received real updates."""

    if epoch < config.ema_teacher_start_epoch:
        return 0.0
    if config.ema_teacher_ramp_epochs <= 0:
        return 1.0
    completed = epoch - config.ema_teacher_start_epoch + progress
    return min(1.0, max(0.0, completed / config.ema_teacher_ramp_epochs))


def train_one_epoch(
    model: FinalModelV5,
    ema: ModelEMA,
    loader: DataLoader,
    sampler: WeightedRandomSampler,
    optimizer: AdamW,
    scaler: GradScaler,
    losses: LossBundle,
    device: torch.device,
    config: TrainConfig,
    *,
    epoch: int,
    class_to_plant: Tensor,
    num_disease_classes: int,
    num_plant_classes: int,
    num_symptom_classes: int,
) -> dict[str, float]:
    model.train()
    _set_dataset_epoch(loader.dataset, epoch)
    if sampler.generator is not None:
        sampler.generator.manual_seed(config.seed + epoch)

    totals = {
        key: 0.0
        for key in (
            "loss",
            "classification",
            "weak_classification",
            "strong_classification",
            "disease",
            "plant",
            "symptom",
            "within_plant",
            "disease_consistency",
            "plant_consistency",
            "symptom_consistency",
            "morphology_supcon",
            "symptom_supcon",
            "joint_queue_contrastive",
            "ema_teacher",
            "ema_teacher_disease_coverage",
            "ema_teacher_plant_coverage",
            "ema_teacher_symptom_coverage",
            "attention",
        )
    }
    processed = 0
    optimizer_steps = 0
    total_batches = len(loader)
    if config.max_train_batches is not None:
        total_batches = min(total_batches, config.max_train_batches)
    if total_batches <= 0:
        raise RuntimeError("Training loader has no batches")

    phase = phase_for_epoch(epoch)
    strong_weight = phase.strong_ce_weight
    supcon_enabled = epoch >= config.supcon_start_epoch
    progress_bar = tqdm(
        loader,
        total=total_batches,
        desc=f"V5 train {epoch + 1}/{config.epochs}",
        leave=False,
    )
    optimizer.zero_grad(set_to_none=True)
    pending_queue_keys: list[tuple[Tensor, Tensor]] = []
    for batch_index, raw_batch in enumerate(progress_bar):
        if config.max_train_batches is not None and batch_index >= total_batches:
            break
        batch = _to_device(raw_batch, device)
        progress = batch_index / max(total_batches, 1)
        consistency_ramp = _consistency_ramp(config, epoch, progress)
        teacher_ramp = _ema_teacher_ramp(config, epoch, progress)
        teacher_required = (
            (config.ema_teacher_weight > 0.0 and teacher_ramp > 0.0)
            or (
                supcon_enabled
                and config.joint_queue_contrastive_weight > 0.0
            )
        )

        with _autocast_context(device, config.amp and device.type in {"cuda", "cpu"}):
            weak = model(
                batch["leaf_image_weak"],
                batch["leaf_mask_weak"],
                return_aux=True,
            )
            strong = model(
                batch["leaf_image_strong"],
                batch["leaf_mask_strong"],
                return_aux=True,
            )
            teacher_output: Mapping[str, Tensor] | None = None
            if teacher_required:
                # EMA stays in eval mode and is never differentiated through.
                # Its weak source view yields stable keys/soft targets for the
                # paired student strong view without reading any target image.
                with torch.no_grad():
                    teacher_output = ema.module(
                        batch["leaf_image_weak"],
                        batch["leaf_mask_weak"],
                        return_aux=False,
                    )
            batch_size = int(batch["label"].shape[0])
            _assert_output_contract(
                weak,
                batch_size=batch_size,
                num_disease_classes=num_disease_classes,
                num_plant_classes=num_plant_classes,
                num_symptom_classes=num_symptom_classes,
            )
            _assert_output_contract(
                strong,
                batch_size=batch_size,
                num_disease_classes=num_disease_classes,
                num_plant_classes=num_plant_classes,
                num_symptom_classes=num_symptom_classes,
            )
            weak_cls, weak_parts = _classification_loss(
                weak,
                batch["label"],
                batch["plant_label"],
                batch["disease_type_label"],
                class_to_plant,
                config,
            )
            strong_cls, strong_parts = _classification_loss(
                strong,
                batch["label"],
                batch["plant_label"],
                batch["disease_type_label"],
                class_to_plant,
                config,
            )
            # Weak supervision remains coefficient 1.0 throughout.  Strong-view
            # CE is additive and intentionally bounded to 0.25--0.35.
            classification = weak_cls + strong_weight * strong_cls

            disease_consistency = losses.consistency(
                weak["disease_logits"], strong["disease_logits"]
            )
            plant_consistency = losses.consistency(
                weak["plant_logits"], strong["plant_logits"]
            )
            symptom_consistency = losses.consistency(
                weak["symptom_logits"], strong["symptom_logits"]
            )
            consistency = consistency_ramp * (
                config.disease_consistency_weight * disease_consistency
                + config.plant_consistency_weight * plant_consistency
                + config.symptom_consistency_weight * symptom_consistency
            )

            zero = weak["features"].sum() * 0.0
            teacher_loss = zero
            teacher_disease_coverage = zero
            teacher_plant_coverage = zero
            teacher_symptom_coverage = zero
            if teacher_output is not None:
                teacher_disease, teacher_disease_coverage = losses.ema_teacher(
                    strong["disease_logits"],
                    teacher_output["disease_logits"],
                    batch["label"],
                )
                teacher_plant, teacher_plant_coverage = losses.ema_teacher(
                    strong["plant_logits"],
                    teacher_output["plant_logits"],
                    batch["plant_label"],
                )
                teacher_symptom, teacher_symptom_coverage = losses.ema_teacher(
                    strong["symptom_logits"],
                    teacher_output["symptom_logits"],
                    batch["disease_type_label"],
                )
                teacher_loss = (
                    teacher_disease
                    + config.plant_loss_weight * teacher_plant
                    + config.symptom_loss_weight * teacher_symptom
                )

            if supcon_enabled:
                morphology_supcon = losses.morphology_supcon(
                    torch.stack(
                        (weak["morphology_features"], strong["morphology_features"]),
                        dim=1,
                    ),
                    batch["plant_label"],
                )
                symptom_supcon = losses.symptom_supcon(
                    torch.stack(
                        (weak["symptom_features"], strong["symptom_features"]),
                        dim=1,
                    ),
                    batch["disease_type_label"],
                )
                if (
                    teacher_output is not None
                    and config.joint_queue_contrastive_weight > 0.0
                ):
                    joint_queue_contrastive = losses.joint_queue_contrastive(
                        strong["z"],
                        batch["label"],
                        key_features=teacher_output["z"],
                    )
                    pending_queue_keys.append(
                        (teacher_output["z"].detach(), batch["label"].detach())
                    )
                else:
                    joint_queue_contrastive = zero
            else:
                morphology_supcon = weak["morphology_features"].sum() * 0.0
                symptom_supcon = weak["symptom_features"].sum() * 0.0
                joint_queue_contrastive = zero

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
                + consistency
                + config.morphology_supcon_weight * morphology_supcon
                + config.symptom_supcon_weight * symptom_supcon
                + config.joint_queue_contrastive_weight * joint_queue_contrastive
                + config.ema_teacher_weight * teacher_ramp * teacher_loss
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
            step_was_skipped = scaler.get_scale() < scale_before_step
            if not step_was_skipped:
                ema.update(model)
                for queue_keys, queue_labels in pending_queue_keys:
                    losses.joint_queue_contrastive.enqueue(queue_keys, queue_labels)
                optimizer_steps += 1
            pending_queue_keys.clear()

        combined_parts = {
            key: weak_parts[key] + strong_weight * strong_parts[key]
            for key in weak_parts
        }
        values = {
            "loss": total_loss,
            "classification": classification,
            "weak_classification": weak_cls,
            "strong_classification": strong_cls,
            "disease": combined_parts["disease"],
            "plant": combined_parts["plant"],
            "symptom": combined_parts["symptom"],
            "within_plant": combined_parts["within_plant"],
            "disease_consistency": disease_consistency,
            "plant_consistency": plant_consistency,
            "symptom_consistency": symptom_consistency,
            "morphology_supcon": morphology_supcon,
            "symptom_supcon": symptom_supcon,
            "joint_queue_contrastive": joint_queue_contrastive,
            "ema_teacher": teacher_loss,
            "ema_teacher_disease_coverage": teacher_disease_coverage,
            "ema_teacher_plant_coverage": teacher_plant_coverage,
            "ema_teacher_symptom_coverage": teacher_symptom_coverage,
            "attention": attention,
        }
        processed += batch_size
        for key, value in values.items():
            totals[key] += float(value.detach().item()) * batch_size
        progress_bar.set_postfix(
            loss=f"{float(total_loss.detach().item()):.3f}",
            djs=f"{float(disease_consistency.detach().item()):.3f}",
        )

    if processed == 0 or optimizer_steps == 0:
        raise RuntimeError("V5 epoch produced no samples or optimizer updates")
    result = {key: value / processed for key, value in totals.items()}
    result["samples"] = float(processed)
    result["optimizer_steps"] = float(optimizer_steps)
    expected_optimizer_steps = math.ceil(total_batches / config.gradient_accumulation)
    result["expected_optimizer_steps"] = float(expected_optimizer_steps)
    result["skipped_optimizer_steps"] = float(
        expected_optimizer_steps - optimizer_steps
    )
    result["strong_ce_weight"] = float(strong_weight)
    result["consistency_ramp_end"] = float(
        _consistency_ramp(config, epoch, 1.0)
    )
    result["ema_teacher_ramp_end"] = float(_ema_teacher_ramp(config, epoch, 1.0))
    result["joint_queue_features"] = float(
        losses.joint_queue_contrastive.stored_features
    )
    return result


def _disease_class_diagnostics(
    metrics: ClassificationMetricsAccumulator,
    *,
    class_names: Sequence[str],
    class_to_plant: Sequence[int],
    train_class_counts: Sequence[int],
    top_k: int,
    min_support: int,
) -> dict[str, Any]:
    """Return compact, JSON-safe class errors without serialising a matrix."""

    num_classes = metrics.num_classes
    if len(class_names) != num_classes:
        raise ValueError("class_names do not match disease metric dimensions")
    if len(class_to_plant) != num_classes:
        raise ValueError("class_to_plant does not match disease metric dimensions")
    if len(train_class_counts) != num_classes:
        raise ValueError("train_class_counts do not match disease metric dimensions")

    per_class = metrics.per_class()
    matrix = metrics.confusion_matrix.detach().to(device="cpu", dtype=torch.int64)
    rows: list[dict[str, Any]] = []
    for index in range(num_classes):
        support = int(per_class["support"][index].item())
        if support <= 0:
            continue
        row = matrix[index].clone()
        row[index] = 0
        confused_count = int(row.max().item())
        dominant_confusion: dict[str, Any] | None = None
        if confused_count > 0:
            predicted_index = int(row.argmax().item())
            dominant_confusion = {
                "class_index": predicted_index,
                "class_name": str(class_names[predicted_index]),
                "count": confused_count,
                "rate_within_true_class": confused_count / support,
                "same_plant": bool(
                    int(class_to_plant[index]) == int(class_to_plant[predicted_index])
                ),
            }
        rows.append(
            {
                "class_index": index,
                "class_name": str(class_names[index]),
                "train_samples": int(train_class_counts[index]),
                "support": support,
                "precision": float(per_class["precision"][index].item()),
                "recall": float(per_class["recall"][index].item()),
                "f1": float(per_class["f1"][index].item()),
                "low_validation_support": support < min_support,
                "dominant_confusion": dominant_confusion,
            }
        )

    supported = [row for row in rows if row["support"] >= min_support]
    ranking_pool = supported if supported else rows
    ranked_rows = sorted(
        ranking_pool,
        key=lambda row: (row["f1"], row["recall"], -row["support"], row["class_name"]),
    )
    weakest = ranked_rows[:top_k]
    bottom_count = max(1, math.ceil(len(ranking_pool) / 4)) if ranking_pool else 0
    bottom_quartile_mean_f1 = (
        float(np.mean([row["f1"] for row in ranked_rows[:bottom_count]]))
        if bottom_count
        else 0.0
    )
    lowest_train_support = sorted(
        rows,
        key=lambda row: (row["train_samples"], row["class_name"]),
    )[:top_k]

    off_diagonal = matrix.clone()
    off_diagonal.fill_diagonal_(0)
    top_confusions: list[dict[str, Any]] = []
    flat = off_diagonal.reshape(-1)
    for flat_index in torch.argsort(flat, descending=True).tolist():
        count = int(flat[flat_index].item())
        if count <= 0 or len(top_confusions) >= top_k:
            break
        true_index = flat_index // num_classes
        predicted_index = flat_index % num_classes
        support = max(int(matrix[true_index].sum().item()), 1)
        top_confusions.append(
            {
                "true_class_index": true_index,
                "true_class_name": str(class_names[true_index]),
                "predicted_class_index": predicted_index,
                "predicted_class_name": str(class_names[predicted_index]),
                "count": count,
                "rate_within_true_class": count / support,
                "same_plant": bool(
                    int(class_to_plant[true_index])
                    == int(class_to_plant[predicted_index])
                ),
            }
        )

    return {
        "min_support_for_ranking": int(min_support),
        "bottom_quartile_mean_f1": bottom_quartile_mean_f1,
        "weakest_classes": weakest,
        "lowest_train_support_classes": lowest_train_support,
        "top_confusions": top_confusions,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    *,
    num_disease_classes: int,
    num_plant_classes: int,
    num_symptom_classes: int,
    class_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
    train_class_counts: Sequence[int],
    domain_name: str,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Evaluate source metrics and a source-label-only error decomposition.

    ``oracle_true_plant_disease_*`` is not an inference result.  It is an
    error probe that removes impossible plants with the known source label so
    we can separate crop/mask failures from within-plant lesion failures.
    """

    if len(class_names) != num_disease_classes:
        raise ValueError("class_names length differs from num_disease_classes")
    if len(class_to_plant) != num_disease_classes:
        raise ValueError("class_to_plant length differs from num_disease_classes")
    if len(class_to_symptom) != num_disease_classes:
        raise ValueError("class_to_symptom length differs from num_disease_classes")
    if len(train_class_counts) != num_disease_classes:
        raise ValueError("train_class_counts length differs from num_disease_classes")

    model.eval()
    disease_metrics = ClassificationMetricsAccumulator(num_disease_classes)
    oracle_disease_metrics = ClassificationMetricsAccumulator(num_disease_classes)
    plant_metrics = ClassificationMetricsAccumulator(num_plant_classes)
    mapped_disease_plant_metrics = ClassificationMetricsAccumulator(num_plant_classes)
    symptom_metrics = ClassificationMetricsAccumulator(num_symptom_classes)
    mapped_disease_symptom_metrics = ClassificationMetricsAccumulator(
        num_symptom_classes
    )
    class_to_plant_tensor = torch.as_tensor(
        class_to_plant, dtype=torch.long, device=device
    )
    class_to_symptom_tensor = torch.as_tensor(
        class_to_symptom, dtype=torch.long, device=device
    )
    disease_loss_sum = 0.0
    sample_count = 0
    disease_error_count = 0
    cross_plant_error_count = 0
    plant_head_agreement_count = 0
    symptom_head_agreement_count = 0
    plant_head_rescue_count = 0
    symptom_head_rescue_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if config.max_val_batches is not None and batch_index >= config.max_val_batches:
            break
        batch = _to_device(raw_batch, device)
        with _autocast_context(device, config.amp and device.type in {"cuda", "cpu"}):
            output = model(
                batch["leaf_image_strong"],
                batch["leaf_mask_strong"],
                return_aux=True,
            )
            batch_size = int(batch["label"].shape[0])
            _assert_output_contract(
                output,
                batch_size=batch_size,
                num_disease_classes=num_disease_classes,
                num_plant_classes=num_plant_classes,
                num_symptom_classes=num_symptom_classes,
            )
            disease_loss = F.cross_entropy(output["disease_logits"], batch["label"])
        disease_logits = output["disease_logits"].float()
        disease_predictions = disease_logits.argmax(dim=1)
        mapped_plant_predictions = class_to_plant_tensor[disease_predictions]
        mapped_symptom_predictions = class_to_symptom_tensor[disease_predictions]
        direct_plant_predictions = output["plant_logits"].argmax(dim=1)
        direct_symptom_predictions = output["symptom_logits"].argmax(dim=1)
        valid_true_plant = class_to_plant_tensor.unsqueeze(0).eq(
            batch["plant_label"].unsqueeze(1)
        )
        oracle_logits = disease_logits.masked_fill(~valid_true_plant, -1e9)
        oracle_disease_predictions = oracle_logits.argmax(dim=1)

        disease_metrics.update(disease_logits.cpu(), batch["label"].cpu())
        oracle_disease_metrics.update(
            oracle_disease_predictions.cpu(), batch["label"].cpu()
        )
        plant_metrics.update(
            output["plant_logits"].float().cpu(), batch["plant_label"].cpu()
        )
        mapped_disease_plant_metrics.update(
            mapped_plant_predictions.cpu(), batch["plant_label"].cpu()
        )
        symptom_metrics.update(
            output["symptom_logits"].float().cpu(),
            batch["disease_type_label"].cpu(),
        )
        mapped_disease_symptom_metrics.update(
            mapped_symptom_predictions.cpu(), batch["disease_type_label"].cpu()
        )
        disease_wrong = disease_predictions.ne(batch["label"])
        disease_error_count += int(disease_wrong.sum().item())
        cross_plant_error_count += int(
            (disease_wrong & mapped_plant_predictions.ne(batch["plant_label"]))
            .sum()
            .item()
        )
        plant_head_agreement_count += int(
            direct_plant_predictions.eq(mapped_plant_predictions).sum().item()
        )
        symptom_head_agreement_count += int(
            direct_symptom_predictions.eq(mapped_symptom_predictions).sum().item()
        )
        plant_head_rescue_count += int(
            (
                direct_plant_predictions.eq(batch["plant_label"])
                & mapped_plant_predictions.ne(batch["plant_label"])
            )
            .sum()
            .item()
        )
        symptom_head_rescue_count += int(
            (
                direct_symptom_predictions.eq(batch["disease_type_label"])
                & mapped_symptom_predictions.ne(batch["disease_type_label"])
            )
            .sum()
            .item()
        )
        disease_loss_sum += float(disease_loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(f"Evaluation domain {domain_name} had no samples")
    result: dict[str, Any] = {
        "loss": disease_loss_sum / sample_count,
        "samples": float(sample_count),
    }
    result.update(
        {f"disease_{key}": value for key, value in disease_metrics.compute().items()}
    )
    result.update(
        {
            f"oracle_true_plant_disease_{key}": value
            for key, value in oracle_disease_metrics.compute().items()
        }
    )
    result.update(
        {f"plant_{key}": value for key, value in plant_metrics.compute().items()}
    )
    result.update(
        {
            f"disease_mapped_plant_{key}": value
            for key, value in mapped_disease_plant_metrics.compute().items()
        }
    )
    result.update(
        {f"symptom_{key}": value for key, value in symptom_metrics.compute().items()}
    )
    result.update(
        {
            f"disease_mapped_symptom_{key}": value
            for key, value in mapped_disease_symptom_metrics.compute().items()
        }
    )
    plant_head_metrics = plant_metrics.compute()
    mapped_plant_metrics = mapped_disease_plant_metrics.compute()
    symptom_head_metrics = symptom_metrics.compute()
    mapped_symptom_metrics = mapped_disease_symptom_metrics.compute()
    result.update(
        {
            "cross_plant_error_rate": cross_plant_error_count / sample_count,
            "cross_plant_error_share": cross_plant_error_count
            / max(disease_error_count, 1),
            "plant_head_disease_mapping_agreement": plant_head_agreement_count
            / sample_count,
            "symptom_head_disease_mapping_agreement": symptom_head_agreement_count
            / sample_count,
            "plant_head_rescue_rate": plant_head_rescue_count / sample_count,
            "symptom_head_rescue_rate": symptom_head_rescue_count / sample_count,
            "plant_head_minus_disease_mapped_accuracy": (
                float(plant_head_metrics["accuracy"])
                - float(mapped_plant_metrics["accuracy"])
            ),
            "symptom_head_minus_disease_mapped_accuracy": (
                float(symptom_head_metrics["accuracy"])
                - float(mapped_symptom_metrics["accuracy"])
            ),
        }
    )
    if include_diagnostics:
        result["disease_class_diagnostics"] = _disease_class_diagnostics(
            disease_metrics,
            class_names=class_names,
            class_to_plant=class_to_plant,
            train_class_counts=train_class_counts,
            top_k=config.diagnostic_top_k,
            min_support=config.diagnostic_min_support,
        )
    return result


def robust_selection_score(
    id_metrics: Mapping[str, Any],
    ood_metrics: Mapping[str, Mapping[str, Any]],
) -> float:
    """Disease macro-F1 only; auxiliary heads never select checkpoints."""

    id_score = float(id_metrics["disease_macro_f1"])
    domain_scores = [
        float(metrics["disease_macro_f1"]) for metrics in ood_metrics.values()
    ]
    if not domain_scores:
        raise ValueError("At least one source OOD domain is required")
    return 0.20 * id_score + 0.50 * float(np.mean(domain_scores)) + 0.30 * min(
        domain_scores
    )


def _source_epoch_diagnostic(
    *,
    epoch: int,
    phase: CurriculumPhase,
    train_metrics: Mapping[str, float],
    id_metrics: Mapping[str, Any],
    paired_ood_clean_metrics: Mapping[str, Any],
    ood_metrics: Mapping[str, Mapping[str, Any]],
    robust_score: float,
    config: TrainConfig,
    train_class_counts: Sequence[int],
) -> dict[str, Any]:
    """Create reporting-only, source-only recommendations for one epoch.

    D1/D2 are compared with the paired clean D0 loader over the *same*
    ``ood_val`` rows.  This avoids mistaking a split difference for a domain
    robustness gap.  The result never changes a loss, sampler, learning rate,
    or checkpoint selection.
    """

    paired_domain_drops: dict[str, dict[str, float]] = {}
    for domain, metrics in ood_metrics.items():
        paired_domain_drops[domain] = {
            "paired_clean_d0_macro_f1": float(
                paired_ood_clean_metrics["disease_macro_f1"]
            ),
            "domain_macro_f1": float(metrics["disease_macro_f1"]),
            "macro_f1_drop": float(
                paired_ood_clean_metrics["disease_macro_f1"]
            )
            - float(metrics["disease_macro_f1"]),
            "accuracy_drop": float(paired_ood_clean_metrics["disease_accuracy"])
            - float(metrics["disease_accuracy"]),
        }

    disease_error_rate = 1.0 - float(id_metrics["disease_accuracy"])
    oracle_gain = float(id_metrics["oracle_true_plant_disease_macro_f1"]) - float(
        id_metrics["disease_macro_f1"]
    )
    augmentation_tax = float(train_metrics["strong_classification"]) - float(
        train_metrics["weak_classification"]
    )
    weak_by_domain = {
        "ID-D0": id_metrics.get("disease_class_diagnostics", {}),
        "OOD-D0-paired": paired_ood_clean_metrics.get(
            "disease_class_diagnostics", {}
        ),
        **{
            f"OOD-{domain}": metrics.get("disease_class_diagnostics", {})
            for domain, metrics in ood_metrics.items()
        },
    }

    suggestions: list[dict[str, str]] = []
    d1_drop = paired_domain_drops.get("D1", {}).get("macro_f1_drop", 0.0)
    if d1_drop >= 0.08:
        suggestions.append(
            {
                "code": "photometric_shift",
                "evidence": f"Paired D0→D1 macro-F1 drops {d1_drop:.3f}.",
                "action": (
                    "Tăng ablation colour/illumination robustness; không dùng "
                    "PlantDoc hoặc pseudo-label target."
                ),
            }
        )
    d2_drop = paired_domain_drops.get("D2", {}).get("macro_f1_drop", 0.0)
    if d2_drop >= 0.08:
        suggestions.append(
            {
                "code": "texture_resolution_shift",
                "evidence": f"Paired D0→D2 macro-F1 drops {d2_drop:.3f}.",
                "action": (
                    "Ưu tiên image_size=448 và --use-detail-branch; kiểm tra "
                    "compression/downscale augmentation trước khi đổi backbone."
                ),
            }
        )
    if (
        float(id_metrics["cross_plant_error_share"]) >= 0.50
        and oracle_gain >= 0.08
        and disease_error_rate >= 0.10
    ):
        suggestions.append(
            {
                "code": "cross_plant_error",
                "evidence": (
                    f"{float(id_metrics['cross_plant_error_share']):.1%} lỗi disease "
                    f"đi sang plant khác; oracle-plant macro-F1 gain={oracle_gain:.3f}."
                ),
                "action": (
                    "Ưu tiên kiểm tra SAM crop/mask và morphology/plant branch; "
                    "V6 dual-view mask là ablation hợp lý hơn DANN."
                ),
            }
        )
    if (
        float(id_metrics["plant_head_minus_disease_mapped_accuracy"]) >= 0.05
        and float(id_metrics["cross_plant_error_rate"]) >= 0.05
    ):
        suggestions.append(
            {
                "code": "taxonomy_head_gap",
                "evidence": (
                    "Plant auxiliary head vượt plant suy ra từ disease logits "
                    f"{float(id_metrics['plant_head_minus_disease_mapped_accuracy']):.3f}."
                ),
                "action": (
                    "Ablate soft taxonomy-compatibility residual hoặc simple-DINO "
                    "head; không hard-mask disease logits lúc inference."
                ),
            }
        )
    teacher_is_fully_ramped = epoch >= (
        config.ema_teacher_start_epoch + config.ema_teacher_ramp_epochs
    )
    teacher_coverage = float(train_metrics["ema_teacher_disease_coverage"])
    if (
        config.ema_teacher_weight > 0.0
        and teacher_is_fully_ramped
        and teacher_coverage < 0.05
    ):
        suggestions.append(
            {
                "code": "ema_teacher_inactive",
                "evidence": (
                    "EMA disease coverage is "
                    f"{teacher_coverage:.1%} after the ramp."
                ),
                "action": (
                    "Teacher gate gần như không đóng góp; thử confidence 0.55–0.65 "
                    "trong ablation riêng và giữ kiểm tra label-agreement."
                ),
            }
        )

    weak_d1 = {
        row["class_name"]
        for row in weak_by_domain.get("OOD-D1", {}).get("weakest_classes", [])
        if not row.get("low_validation_support", False)
    }
    weak_d2 = {
        row["class_name"]
        for row in weak_by_domain.get("OOD-D2", {}).get("weakest_classes", [])
        if not row.get("low_validation_support", False)
    }
    repeated_weak_classes = sorted(weak_d1 & weak_d2)
    if repeated_weak_classes:
        suggestions.append(
            {
                "code": "persistent_class_errors",
                "evidence": "Yếu ở cả D1 và D2: " + ", ".join(repeated_weak_classes[:3]),
                "action": (
                    "Kiểm tra nhãn/mask của lớp này và class sampling; nếu ít dữ liệu, "
                    "tune balance_fraction hoặc prototype/margin loss."
                ),
            }
        )

    rare_cutoff = float(np.quantile(np.asarray(train_class_counts), 0.25))
    id_weak = weak_by_domain["ID-D0"].get("weakest_classes", [])
    rare_weak = [
        row["class_name"]
        for row in id_weak
        if float(row.get("train_samples", math.inf)) <= rare_cutoff
    ]
    if rare_weak:
        suggestions.append(
            {
                "code": "class_imbalance",
                "evidence": "Lớp yếu có train support thấp: " + ", ".join(rare_weak[:3]),
                "action": (
                    "Thử balance_fraction 0.25/0.50/0.75 và đo macro-F1 qua nhiều seed; "
                    "đừng kết luận từ một lớp support nhỏ."
                ),
            }
        )
    if int(train_metrics["skipped_optimizer_steps"]) > 0:
        suggestions.append(
            {
                "code": "optimizer_steps_skipped",
                "evidence": (
                    f"Skipped {int(train_metrics['skipped_optimizer_steps'])} optimizer "
                    "step(s) this epoch."
                ),
                "action": "Kiểm tra AMP/LR trước khi diễn giải domain metrics.",
            }
        )
    if not suggestions:
        suggestions.append(
            {
                "code": "monitor",
                "evidence": "Chưa có gap vượt ngưỡng chẩn đoán trong epoch này.",
                "action": (
                    "Theo dõi xu hướng trong cùng curriculum phase; không đổi loss chỉ "
                    "vì một epoch dao động."
                ),
            }
        )

    return {
        "schema_version": 1,
        "source_only": True,
        "diagnostic_only": True,
        "epoch": int(epoch),
        "phase": phase.name,
        "robust_score": float(robust_score),
        "id": {
            "disease_macro_f1": float(id_metrics["disease_macro_f1"]),
            "disease_accuracy": float(id_metrics["disease_accuracy"]),
            "disease_balanced_accuracy": float(
                id_metrics["disease_balanced_accuracy"]
            ),
            "cross_plant_error_rate": float(id_metrics["cross_plant_error_rate"]),
            "cross_plant_error_share": float(id_metrics["cross_plant_error_share"]),
            "oracle_true_plant_macro_f1": float(
                id_metrics["oracle_true_plant_disease_macro_f1"]
            ),
            "oracle_true_plant_macro_f1_gain": oracle_gain,
            "plant_head_minus_disease_mapped_accuracy": float(
                id_metrics["plant_head_minus_disease_mapped_accuracy"]
            ),
            "symptom_head_minus_disease_mapped_accuracy": float(
                id_metrics["symptom_head_minus_disease_mapped_accuracy"]
            ),
        },
        "paired_domain_drops": paired_domain_drops,
        "training_health": {
            "loss": float(train_metrics["loss"]),
            "weak_classification": float(train_metrics["weak_classification"]),
            "strong_classification": float(train_metrics["strong_classification"]),
            "strong_augmentation_tax": augmentation_tax,
            "ema_teacher_disease_coverage": teacher_coverage,
            "joint_queue_features": int(train_metrics["joint_queue_features"]),
            "joint_queue_ready": bool(
                int(train_metrics["joint_queue_features"])
                >= config.contrastive_queue_size
            ),
            "optimizer_steps": int(train_metrics["optimizer_steps"]),
            "expected_optimizer_steps": int(
                train_metrics["expected_optimizer_steps"]
            ),
            "skipped_optimizer_steps": int(train_metrics["skipped_optimizer_steps"]),
        },
        "class_errors": weak_by_domain,
        "recommendations": suggestions,
        "held_out_data": {
            "D4_used": False,
            "PlantDoc_used": False,
            "checkpoint_selection_affected": False,
        },
    }


def _print_epoch_diagnostic(report: Mapping[str, Any]) -> None:
    """Print a concise Vietnamese operator view of the JSON diagnostic report."""

    id_metrics = report["id"]
    paired_drops = report["paired_domain_drops"]
    drop_text = ", ".join(
        f"{domain} F1={float(values['domain_macro_f1']):.4f} "
        f"(Δ={float(values['macro_f1_drop']):+.3f})"
        for domain, values in paired_drops.items()
    )
    health = report["training_health"]
    print(
        "V5 chẩn đoán "
        f"epoch {int(report['epoch']) + 1} [{report['phase']}] | "
        f"robust={float(report['robust_score']):.4f} | "
        f"ID F1={float(id_metrics['disease_macro_f1']):.4f} | {drop_text}"
    )
    print(
        "  lỗi: "
        f"cross-plant={float(id_metrics['cross_plant_error_rate']):.1%} "
        f"(share={float(id_metrics['cross_plant_error_share']):.1%}), "
        f"oracle-plant gain={float(id_metrics['oracle_true_plant_macro_f1_gain']):+.3f}; "
        f"EMA coverage={float(health['ema_teacher_disease_coverage']):.1%}, "
        f"queue={int(health['joint_queue_features'])}"
    )
    print(
        "  train: "
        f"loss={float(health['loss']):.3f}, "
        f"weak/strong CE={float(health['weak_classification']):.3f}/"
        f"{float(health['strong_classification']):.3f}, "
        f"augmentation-tax={float(health['strong_augmentation_tax']):+.3f}, "
        f"steps={int(health['optimizer_steps'])}/"
        f"{int(health['expected_optimizer_steps'])}"
    )
    weak = report["class_errors"].get("ID-D0", {}).get("weakest_classes", [])
    if weak:
        weak_text = ", ".join(
            f"{row['class_name']} F1={float(row['f1']):.2f}"
            for row in weak[:3]
        )
        print(f"  lớp yếu (ID): {weak_text}")
    for suggestion in report["recommendations"]:
        print(
            f"  → [{suggestion['code']}] {suggestion['action']} "
            f"({suggestion['evidence']})"
        )


def _checkpoint_metadata(
    *,
    epoch: int,
    config: TrainConfig,
    model_config: Mapping[str, Any],
    class_names: Sequence[str],
    plant_names: Sequence[str],
    symptom_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "protocol": PROTOCOL,
        "source_only": True,
        "epoch": int(epoch),
        "inference_rule": INFERENCE_RULE,
        "prediction_space": "all source classes; direct disease logits; no gating or masking",
        "selection_metric": "source robust disease macro-F1",
        "train_config": _jsonable_config(config),
        "model_config": dict(model_config),
        "class_names": list(class_names),
        "plant_names": list(plant_names),
        "symptom_names": list(symptom_names),
        "class_to_plant": list(class_to_plant),
        "class_to_symptom": list(class_to_symptom),
        "run_identity": dict(run_identity),
    }


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    best_score: float,
    model: FinalModelV5,
    ema: ModelEMA,
    losses: LossBundle,
    optimizer: AdamW,
    scaler: GradScaler,
    config: TrainConfig,
    model_config: Mapping[str, Any],
    class_names: Sequence[str],
    plant_names: Sequence[str],
    symptom_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
    run_identity: Mapping[str, Any],
    validation: Mapping[str, Any] | None,
) -> None:
    metadata = _checkpoint_metadata(
        epoch=epoch,
        config=config,
        model_config=model_config,
        class_names=class_names,
        plant_names=plant_names,
        symptom_names=symptom_names,
        class_to_plant=class_to_plant,
        class_to_symptom=class_to_symptom,
        run_identity=run_identity,
    )
    atomic_torch_save(
        {
            **metadata,
            "training_complete": False,
            "checkpoint_role": "resume_state",
            "source_acceptance_passed": False,
            "best_robust_score": float(best_score),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "loss_state_dict": losses.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _rng_state(),
            "validation": dict(validation) if validation is not None else None,
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
    model_config: Mapping[str, Any],
    class_names: Sequence[str],
    plant_names: Sequence[str],
    symptom_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
    run_identity: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    metadata = _checkpoint_metadata(
        epoch=epoch,
        config=config,
        model_config=model_config,
        class_names=class_names,
        plant_names=plant_names,
        symptom_names=symptom_names,
        class_to_plant=class_to_plant,
        class_to_symptom=class_to_symptom,
        run_identity=run_identity,
    )
    atomic_torch_save(
        {
            **metadata,
            "training_complete": False,
            "checkpoint_role": "source_selected_interim",
            "source_acceptance_passed": False,
            "robust_score": float(robust_score),
            "model_state_dict": ema.module.state_dict(),
            "validation": dict(validation),
        },
        path,
    )


def maybe_resume(
    path: Path,
    *,
    model: FinalModelV5,
    ema: ModelEMA,
    losses: LossBundle,
    optimizer: AdamW,
    scaler: GradScaler,
    model_config: Mapping[str, Any],
    class_names: Sequence[str],
    plant_names: Sequence[str],
    symptom_names: Sequence[str],
    class_to_plant: Sequence[int],
    class_to_symptom: Sequence[int],
    expected_run_identity: Mapping[str, Any],
) -> tuple[int, float, Mapping[str, Any] | None]:
    if not path.is_file():
        return 0, -math.inf, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Resume checkpoint is not format V5: {path}")
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("Resume checkpoint protocol differs from V5 source-only protocol")
    if checkpoint.get("inference_rule") != INFERENCE_RULE:
        raise ValueError("Resume checkpoint inference rule is not direct joint fusion")
    expected_sequences = {
        "class_names": class_names,
        "plant_names": plant_names,
        "symptom_names": symptom_names,
        "class_to_plant": class_to_plant,
        "class_to_symptom": class_to_symptom,
    }
    for key, expected in expected_sequences.items():
        if list(checkpoint.get(key, [])) != list(expected):
            raise ValueError(f"Resume checkpoint {key} differs from source manifest")
    if dict(checkpoint.get("model_config", {})) != dict(model_config):
        raise ValueError("Resume checkpoint model_config differs from requested model")
    checkpoint_identity = checkpoint.get("run_identity", {})
    if checkpoint_identity.get("fingerprint") != expected_run_identity.get(
        "fingerprint"
    ):
        raise ValueError(
            "Resume checkpoint fingerprint differs from source data, code, seed, "
            "mappings, model, domains, or loss config. Use a fresh output directory."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ema.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    loss_state = checkpoint.get("loss_state_dict")
    if not isinstance(loss_state, Mapping):
        raise ValueError("Resume checkpoint lacks source-only contrastive loss state")
    losses.load_state_dict(loss_state)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict", {})
    if scaler.is_enabled() and scaler_state:
        scaler.load_state_dict(scaler_state)
    _restore_rng_state(checkpoint.get("rng_state"))
    start_epoch = int(checkpoint["epoch"]) + 1
    best_score = float(checkpoint.get("best_robust_score", -math.inf))
    validation = checkpoint.get("validation")
    print(f"Resumed strict V5 state at epoch {start_epoch + 1}; best={best_score:.4f}")
    return start_epoch, best_score, validation


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _all_finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_metrics(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def train(config: TrainConfig) -> dict[str, Any]:
    set_global_seed(config.seed, config.deterministic)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # V4's manifest helper is reused because it is source-only, duplicate-group
    # aware, and refuses target CSV names.  Nothing below imports target data.
    manifest = load_or_create_manifest(config)
    (
        class_names,
        plant_names,
        symptom_names,
        class_to_plant,
        class_to_symptom,
    ) = label_schema(manifest)
    train_class_counts = (
        manifest.loc[manifest["split"].eq("train")]
        .groupby("class_index")
        .size()
        .reindex(range(len(class_names)), fill_value=0)
        .astype(int)
        .tolist()
    )
    model_config = _model_config(
        config,
        num_disease_classes=len(class_names),
        num_plant_classes=len(plant_names),
        num_symptom_classes=len(symptom_names),
    )
    run_identity = build_run_identity(
        config,
        model_config=model_config,
        class_names=class_names,
        plant_names=plant_names,
        symptom_names=symptom_names,
        class_to_plant=class_to_plant,
        class_to_symptom=class_to_symptom,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"V5 source-only | device={device} image={config.image_size} "
        f"classes={len(class_names)} plants={len(plant_names)} "
        f"symptoms={len(symptom_names)}"
    )
    model = FinalModelV5(**model_config).to(device)
    optimizer = build_optimizer(model, config)
    amp_enabled = config.amp and device.type == "cuda"
    # Cosine heads can overflow with GradScaler's default 2^16 scale on the
    # first CUDA update.  Start conservatively so a one-batch smoke run and the
    # first production epoch both perform a real optimizer/EMA step.
    scaler = GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=4096.0,
        growth_interval=1000,
    )
    ema = ModelEMA(
        model,
        decay=config.ema_decay,
        warmup_updates=config.ema_warmup_updates,
        device=device,
    )
    losses = LossBundle.from_config(config).to(device)
    loaders = build_loaders(config, manifest)
    class_to_plant_tensor = torch.as_tensor(
        class_to_plant, dtype=torch.long, device=device
    )

    resume_path = output_dir / RESUME_CHECKPOINT_NAME
    best_path = output_dir / BEST_CHECKPOINT_NAME
    history_path = output_dir / "history_v5.jsonl"
    diagnostic_history_path = output_dir / "diagnostics_v5.jsonl"
    latest_diagnostic_path = output_dir / "latest_diagnostics_v5.json"
    config_path = output_dir / "config_v5.json"
    if not config.resume and (resume_path.exists() or best_path.exists()):
        raise FileExistsError(
            "V5 checkpoint already exists. Use --resume or a fresh --output-dir."
        )

    start_epoch, best_score, last_validation = (0, -math.inf, None)
    if config.resume:
        start_epoch, best_score, last_validation = maybe_resume(
            resume_path,
            model=model,
            ema=ema,
            losses=losses,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            class_names=class_names,
            plant_names=plant_names,
            symptom_names=symptom_names,
            class_to_plant=class_to_plant,
            class_to_symptom=class_to_symptom,
            expected_run_identity=run_identity,
        )
    # Diagnostics can run before the first checkpoint-selection epoch, so a
    # non-empty last validation no longer implies that a best checkpoint exists.
    # Require it only after a finite selected score was actually recorded.
    if (
        start_epoch > 0
        and math.isfinite(best_score)
        and not best_path.is_file()
    ):
        raise FileNotFoundError("Resume state references an absent V5 best checkpoint")
    if start_epoch >= config.epochs:
        print("Configured V5 epochs are already complete; rebuilding source report.")

    _write_json_atomic(
        config_path,
        {
            "format_version": FORMAT_VERSION,
            "protocol": PROTOCOL,
            "inference_rule": INFERENCE_RULE,
            "run_fingerprint": run_identity["fingerprint"],
            "train_config": _jsonable_config(config),
            "model_config": model_config,
        },
    )

    for epoch in range(start_epoch, config.epochs):
        phase = apply_curriculum_phase(model, optimizer, epoch, config)
        domain_mix = set_curriculum_sampler_mix(
            loaders.train_sampler, loaders.train_dataset, config, phase
        )
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
            epoch=epoch,
            class_to_plant=class_to_plant_tensor,
            num_disease_classes=len(class_names),
            num_plant_classes=len(plant_names),
            num_symptom_classes=len(symptom_names),
        )

        should_select_checkpoint = (
            (epoch + 1) % config.robust_eval_interval == 0
            or epoch + 1 == config.epochs
        )
        should_diagnose = (
            (epoch + 1) % config.diagnostic_eval_interval == 0
            or should_select_checkpoint
        )
        validation: dict[str, Any] | None = None
        diagnostic_report: dict[str, Any] | None = None
        if should_diagnose:
            id_metrics = evaluate(
                ema.module,
                loaders.id_validation,
                device,
                config,
                num_disease_classes=len(class_names),
                num_plant_classes=len(plant_names),
                num_symptom_classes=len(symptom_names),
                class_names=class_names,
                class_to_plant=class_to_plant,
                class_to_symptom=class_to_symptom,
                train_class_counts=train_class_counts,
                domain_name="ID-D0",
                include_diagnostics=True,
            )
            paired_ood_clean_metrics = evaluate(
                ema.module,
                loaders.paired_ood_clean_validation,
                device,
                config,
                num_disease_classes=len(class_names),
                num_plant_classes=len(plant_names),
                num_symptom_classes=len(symptom_names),
                class_names=class_names,
                class_to_plant=class_to_plant,
                class_to_symptom=class_to_symptom,
                train_class_counts=train_class_counts,
                domain_name="OOD-D0-paired",
                include_diagnostics=True,
            )
            ood_metrics = {
                domain: evaluate(
                    ema.module,
                    loader,
                    device,
                    config,
                    num_disease_classes=len(class_names),
                    num_plant_classes=len(plant_names),
                    num_symptom_classes=len(symptom_names),
                    class_names=class_names,
                    class_to_plant=class_to_plant,
                    class_to_symptom=class_to_symptom,
                    train_class_counts=train_class_counts,
                    domain_name=f"OOD-{domain}",
                    include_diagnostics=True,
                )
                for domain, loader in loaders.ood_validation.items()
            }
            score = robust_selection_score(id_metrics, ood_metrics)
            validation = {
                "id": id_metrics,
                "paired_ood_clean": paired_ood_clean_metrics,
                "ood": ood_metrics,
                "robust_score": score,
                "selection_uses_auxiliary_heads": False,
                "selection_uses_proxy_test": False,
                "checkpoint_selection_epoch": should_select_checkpoint,
            }
            last_validation = validation
            diagnostic_report = _source_epoch_diagnostic(
                epoch=epoch,
                phase=phase,
                train_metrics=train_metrics,
                id_metrics=id_metrics,
                paired_ood_clean_metrics=paired_ood_clean_metrics,
                ood_metrics=ood_metrics,
                robust_score=score,
                config=config,
                train_class_counts=train_class_counts,
            )
            diagnostic_report["checkpoint_selection_epoch"] = should_select_checkpoint
            diagnostic_report["run_fingerprint"] = run_identity["fingerprint"]
            _print_epoch_diagnostic(diagnostic_report)
            append_history(diagnostic_history_path, diagnostic_report)
            _write_json_atomic(latest_diagnostic_path, diagnostic_report)
            if should_select_checkpoint and score > best_score:
                best_score = score
                save_best_checkpoint(
                    best_path,
                    epoch=epoch,
                    robust_score=score,
                    ema=ema,
                    config=config,
                    model_config=model_config,
                    class_names=class_names,
                    plant_names=plant_names,
                    symptom_names=symptom_names,
                    class_to_plant=class_to_plant,
                    class_to_symptom=class_to_symptom,
                    run_identity=run_identity,
                    validation=validation,
                )
                print(f"Saved new source-only V5 best: {best_path.name}")

        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            best_score=best_score,
            model=model,
            ema=ema,
            losses=losses,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            model_config=model_config,
            class_names=class_names,
            plant_names=plant_names,
            symptom_names=symptom_names,
            class_to_plant=class_to_plant,
            class_to_symptom=class_to_symptom,
            run_identity=run_identity,
            validation=validation if validation is not None else last_validation,
        )
        append_history(
            history_path,
            {
                "epoch": epoch,
                "phase": phase.name,
                "domain_mix": domain_mix,
                "strong_ce_weight": phase.strong_ce_weight,
                "train": train_metrics,
                "validation": validation,
                "diagnostic": diagnostic_report,
                "best_robust_score": best_score,
                "run_fingerprint": run_identity["fingerprint"],
            },
        )

    if not best_path.is_file():
        raise RuntimeError("V5 training completed without a source-selected best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if best_checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Best checkpoint format changed unexpectedly")
    if best_checkpoint.get("run_identity", {}).get("fingerprint") != run_identity[
        "fingerprint"
    ]:
        raise ValueError("Best checkpoint run fingerprint differs from this run")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    proxy_metrics = evaluate(
        model,
        loaders.proxy_test,
        device,
        config,
        num_disease_classes=len(class_names),
        num_plant_classes=len(plant_names),
        num_symptom_classes=len(symptom_names),
        class_names=class_names,
        class_to_plant=class_to_plant,
        class_to_symptom=class_to_symptom,
        train_class_counts=train_class_counts,
        domain_name=f"SOURCE-PROXY-{config.proxy_test_domain}",
    )

    source_checks = {
        "checkpoint_contract_valid": True,
        "metrics_are_finite": _all_finite_metrics(
            {"validation": best_checkpoint["validation"], "proxy": proxy_metrics}
        ),
        "all_classes_have_stable_mappings": (
            len(class_names) == len(class_to_plant) == len(class_to_symptom)
        ),
        "direct_full_space_inference": (
            best_checkpoint.get("inference_rule") == INFERENCE_RULE
        ),
        "d0_floor_respected_by_all_phases": all(
            curriculum_domain_mix(phase)["D0"] >= 0.40 for phase in PHASES
        ),
        "d3_not_used": "D3" not in config.train_domains,
        "proxy_not_used_for_selection": True,
        # Phrase every acceptance check positively so ``all(values)`` has the
        # intended meaning.  The previous ``target_data_used=False`` made an
        # otherwise valid source-only run fail its own protocol gate.
        "target_data_absent": True,
        "full_dataset_run": (
            config.max_train_batches is None and config.max_val_batches is None
        ),
        "full_curriculum_completed": config.epochs >= 55,
    }
    source_accepted = all(source_checks.values())
    final_path = output_dir / FINAL_CHECKPOINT_NAME
    source_report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "protocol": PROTOCOL,
        "inference_rule": INFERENCE_RULE,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _file_sha256(best_path),
        "best_epoch_zero_based": int(best_checkpoint["epoch"]),
        "best_robust_score": float(best_checkpoint["robust_score"]),
        "source_validation": best_checkpoint["validation"],
        "held_out_source_proxy_domain": config.proxy_test_domain,
        "held_out_source_proxy": proxy_metrics,
        "source_protocol_checks": source_checks,
        "accepted_for_one_way_final_evaluation": source_accepted,
        "sealed_final_checkpoint": str(final_path) if source_accepted else None,
        "target_accuracy_gate_applied": False,
        "warning": (
            "This source-only report does not estimate or guarantee PlantDoc "
            "accuracy. The frozen evaluator owns the target-domain gate."
        ),
    }
    source_report_path = output_dir / SOURCE_ACCEPTANCE_NAME
    _write_json_atomic(source_report_path, source_report)
    if source_accepted:
        sealed_checkpoint = {
            **best_checkpoint,
            "training_complete": True,
            "checkpoint_role": "source_selected_final",
            "source_acceptance_passed": True,
            "source_acceptance_sha256": _file_sha256(source_report_path),
        }
        atomic_torch_save(sealed_checkpoint, final_path)
        print(f"Sealed final source-selected checkpoint: {final_path.name}")
    elif final_path.exists():
        raise FileExistsError(
            "A sealed V5 final checkpoint exists but this run did not pass the "
            "full-source protocol checks; use a fresh output directory."
        )
    print(
        f"Source proxy {config.proxy_test_domain}: disease macro-F1="
        f"{proxy_metrics['disease_macro_f1']:.4f}. No target data were read."
    )
    return source_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train FinalModelV5 using PlantVillage source-only domain "
            "generalization; never reads PlantDoc."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--train-csv", default=str(PROJECT_ROOT / "train.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "new_model" / "checkpoints_curriculum_v5"),
    )
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--trunk-dim", type=int, default=512)
    parser.add_argument("--morphology-dim", type=int, default=512)
    parser.add_argument("--symptom-dim", type=int, default=512)
    parser.add_argument("--interaction-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--shape-dim", type=int, default=64)
    parser.add_argument("--use-detail-branch", action="store_true")
    parser.add_argument("--detail-dim", type=int, default=128)
    parser.add_argument("--leaf-crop-margin", type=float, default=0.10)
    parser.add_argument("--mask-erosion-kernel", type=int, default=7)

    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--plant-loss-weight", type=float, default=0.25)
    parser.add_argument("--symptom-loss-weight", type=float, default=0.35)
    parser.add_argument("--within-plant-loss-weight", type=float, default=0.20)
    parser.add_argument("--disease-consistency-weight", type=float, default=0.25)
    parser.add_argument("--plant-consistency-weight", type=float, default=0.05)
    parser.add_argument("--symptom-consistency-weight", type=float, default=0.10)
    parser.add_argument("--consistency-ramp-epochs", type=int, default=5)
    parser.add_argument("--morphology-supcon-weight", type=float, default=0.05)
    parser.add_argument("--symptom-supcon-weight", type=float, default=0.05)
    parser.add_argument("--supcon-start-epoch", type=int, default=5)
    parser.add_argument("--joint-queue-contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-queue-size", type=int, default=2048)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--ema-teacher-weight", type=float, default=0.10)
    parser.add_argument("--ema-teacher-start-epoch", type=int, default=5)
    parser.add_argument("--ema-teacher-ramp-epochs", type=int, default=5)
    parser.add_argument("--ema-teacher-temperature", type=float, default=1.5)
    parser.add_argument("--ema-teacher-confidence", type=float, default=0.70)
    parser.add_argument("--attention-loss-weight", type=float, default=0.03)
    parser.add_argument("--balance-fraction", type=float, default=0.50)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-updates", type=int, default=100)
    parser.add_argument("--robust-eval-interval", type=int, default=5)
    parser.add_argument(
        "--diagnostic-eval-interval",
        type=int,
        default=1,
        help="Run source-only ID/D1/D2 diagnostic reports every N epochs.",
    )
    parser.add_argument("--diagnostic-top-k", type=int, default=5)
    parser.add_argument("--diagnostic-min-support", type=int, default=10)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch, one train batch, one batch per source validation loader.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-hash-exact-duplicates", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    epochs = 1 if args.smoke else args.epochs
    num_workers = 0 if args.smoke else args.num_workers
    max_train_batches = (
        1 if args.smoke and args.max_train_batches is None else args.max_train_batches
    )
    max_val_batches = (
        1 if args.smoke and args.max_val_batches is None else args.max_val_batches
    )
    robust_eval_interval = 1 if args.smoke else args.robust_eval_interval
    return TrainConfig(
        project_root=args.project_root,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        seed=args.seed,
        epochs=epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_workers=num_workers,
        model_name=args.model_name,
        trunk_dim=args.trunk_dim,
        morphology_dim=args.morphology_dim,
        symptom_dim=args.symptom_dim,
        interaction_dim=args.interaction_dim,
        projection_dim=args.projection_dim,
        shape_dim=args.shape_dim,
        use_detail_branch=args.use_detail_branch,
        detail_dim=args.detail_dim,
        leaf_crop_margin=args.leaf_crop_margin,
        mask_erosion_kernel=args.mask_erosion_kernel,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        plant_loss_weight=args.plant_loss_weight,
        symptom_loss_weight=args.symptom_loss_weight,
        within_plant_loss_weight=args.within_plant_loss_weight,
        disease_consistency_weight=args.disease_consistency_weight,
        plant_consistency_weight=args.plant_consistency_weight,
        symptom_consistency_weight=args.symptom_consistency_weight,
        consistency_ramp_epochs=args.consistency_ramp_epochs,
        morphology_supcon_weight=args.morphology_supcon_weight,
        symptom_supcon_weight=args.symptom_supcon_weight,
        supcon_start_epoch=args.supcon_start_epoch,
        joint_queue_contrastive_weight=args.joint_queue_contrastive_weight,
        contrastive_queue_size=args.contrastive_queue_size,
        contrastive_temperature=args.contrastive_temperature,
        ema_teacher_weight=args.ema_teacher_weight,
        ema_teacher_start_epoch=args.ema_teacher_start_epoch,
        ema_teacher_ramp_epochs=args.ema_teacher_ramp_epochs,
        ema_teacher_temperature=args.ema_teacher_temperature,
        ema_teacher_confidence=args.ema_teacher_confidence,
        attention_loss_weight=args.attention_loss_weight,
        balance_fraction=args.balance_fraction,
        hash_exact_duplicates=not args.no_hash_exact_duplicates,
        ema_decay=args.ema_decay,
        ema_warmup_updates=args.ema_warmup_updates,
        robust_eval_interval=robust_eval_interval,
        diagnostic_eval_interval=args.diagnostic_eval_interval,
        diagnostic_top_k=args.diagnostic_top_k,
        diagnostic_min_support=args.diagnostic_min_support,
        amp=not args.no_amp,
        deterministic=args.deterministic,
        resume=not args.no_resume,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
