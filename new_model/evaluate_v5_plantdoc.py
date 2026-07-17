"""One-way final PlantDoc evaluation for a frozen leaf-only V5 checkpoint.

This module is intentionally isolated from training.  It loads one checkpoint
that was selected exclusively with PlantVillage source validation, performs no
adaptation, tuning, threshold search, class masking, or checkpoint selection,
and writes the same global final-test registry used by the V4 evaluator.

PlantDoc pixels can reach V5 only through this fixed path::

    SAM mask -> leaf bounding box -> neutralise non-leaf pixels
    -> aspect-preserving neutral letterbox -> leaf_image + leaf_mask

There is deliberately no full-image/context tensor in the dataset or model
call made by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .final_model_v5 import FinalModelV5
except ImportError:
    from final_model_v5 import FinalModelV5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGENET_MEAN_RGB = tuple(int(round(value * 255.0)) for value in IMAGENET_MEAN)
EXPECTED_JOINT_CLASSES = 38
DEFAULT_TARGET_ACCURACY = 0.70

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_data_fingerprint(
    row_indices: Sequence[int],
    targets: Sequence[int],
    image_hashes: Sequence[str],
    mask_hashes: Sequence[str],
) -> str:
    """Hash the ordered labels plus the exact image and mask contents."""

    rows = sorted(
        zip(row_indices, targets, image_hashes, mask_hashes),
        key=lambda item: item[0],
    )
    digest = hashlib.sha256()
    for row_index, target, image_hash, mask_hash in rows:
        digest.update(
            f"{int(row_index)}\x1f{int(target)}\x1f{image_hash}\x1f{mask_hash}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _image_parts(raw_path: str) -> tuple[str, str]:
    path = PureWindowsPath(str(raw_path).strip())
    class_folder, filename = path.parent.name, path.name
    if class_folder in {"", ".", ".."} or filename in {"", ".", ".."}:
        raise ValueError(f"Unsafe PlantDoc image path: {raw_path!r}")
    return class_folder, filename


def _safe_child(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        inside = os.path.commonpath((str(resolved_root), str(candidate))) == str(
            resolved_root
        )
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"Resolved path escapes frozen root {resolved_root}: {candidate}")
    return candidate


def resolve_plantdoc_image(raw_path: str, plantdoc_root: Path | None) -> Path:
    if plantdoc_root is not None:
        class_folder, filename = _image_parts(raw_path)
        candidate = _safe_child(plantdoc_root, class_folder, filename)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"PlantDoc image is absent below the explicit frozen root: {candidate}"
            )
        return candidate

    direct = Path(str(raw_path)).expanduser()
    if direct.is_file():
        return direct.resolve()
    raise FileNotFoundError(
        f"PlantDoc image path from CSV does not exist: {raw_path}. "
        "Pass --plantdoc-root if the dataset was moved."
    )


def resolve_plantdoc_mask(raw_path: str, mask_root: Path) -> Path:
    class_folder, filename = _image_parts(raw_path)
    mask_name = Path(filename).with_suffix(".png").name
    candidate = _safe_child(mask_root, class_folder, mask_name)
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing PlantDoc SAM mask: {candidate}")
    return candidate


def _neutral_letterbox(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize without distortion and pad with the ImageNet-neutral colour."""

    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or image_size <= 0:
        raise ValueError(f"Invalid crop/image size: {image.shape}, {image_size}")
    scale = image_size / max(height, width)
    resized_height = max(1, min(image_size, int(round(height * scale))))
    resized_width = max(1, min(image_size, int(round(width * scale))))
    resized_image = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    resized_mask = cv2.resize(
        mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas = np.full(
        (image_size, image_size, 3), IMAGENET_MEAN_RGB, dtype=np.uint8
    )
    mask_canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized_image
    mask_canvas[top : top + resized_height, left : left + resized_width] = resized_mask
    return canvas, np.ascontiguousarray(mask_canvas > 0, dtype=np.uint8)


def preprocess_leaf_only(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the sole V5 input using a SAM bbox and neutral letterbox.

    Pixels outside the binary mask are replaced before cropping.  Thus even
    the bbox margin cannot carry PlantDoc background/context into the model.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 image, got {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Image/mask shapes differ: {image.shape[:2]} vs {mask.shape}")
    if not 0.0 <= margin <= 0.5:
        raise ValueError("leaf crop margin must be in [0, 0.5]")

    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("Empty PlantDoc mask is forbidden in final evaluation")
    x, y, width, height = cv2.boundingRect(points)
    margin_x = max(2, int(round(width * margin)))
    margin_y = max(2, int(round(height * margin)))
    x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
    x1 = min(image.shape[1], x + width + margin_x)
    y1 = min(image.shape[0], y + height + margin_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("SAM bbox produced an empty PlantDoc crop")

    neutralised = np.full_like(image, IMAGENET_MEAN_RGB)
    neutralised[binary > 0] = image[binary > 0]
    crop = neutralised[y0:y1, x0:x1]
    mask_crop = binary[y0:y1, x0:x1]
    return _neutral_letterbox(crop, mask_crop, image_size)


def _normalised_image_tensor(image: np.ndarray) -> Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


class PlantDocLeafOnlyDataset(Dataset[dict[str, Any]]):
    """Fail-fast frozen PlantDoc dataset that exposes no context tensor."""

    def __init__(
        self,
        csv_path: Path,
        *,
        class_names: Sequence[str],
        image_size: int,
        mask_root: Path,
        plantdoc_root: Path | None = None,
        leaf_crop_margin: float = 0.10,
    ) -> None:
        frame = pd.read_csv(csv_path)
        required = {"image", "plant_disease"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"PlantDoc CSV lacks columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("PlantDoc CSV is empty")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= leaf_crop_margin <= 0.5:
            raise ValueError("leaf_crop_margin must be in [0, 0.5]")

        class_to_index = {name: index for index, name in enumerate(class_names)}
        unknown = sorted(set(frame["plant_disease"].astype(str)) - set(class_to_index))
        if unknown:
            raise ValueError(
                "PlantDoc contains labels absent from the frozen checkpoint: "
                f"{unknown}"
            )

        self.frame = frame.reset_index(drop=True).copy()
        self.frame["label"] = self.frame["plant_disease"].map(class_to_index).astype(int)
        self.image_size = int(image_size)
        self.leaf_crop_margin = float(leaf_crop_margin)
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        for raw_path in self.frame["image"].astype(str):
            self.image_paths.append(resolve_plantdoc_image(raw_path, plantdoc_root))
            self.mask_paths.append(resolve_plantdoc_mask(raw_path, mask_root))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image_payload = image_path.read_bytes()
        image_hash = hashlib.sha256(image_payload).hexdigest()
        image_bgr = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image_bgr is None:
            raise OSError(f"OpenCV could not decode PlantDoc image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_payload = mask_path.read_bytes()
        mask_hash = hashlib.sha256(mask_payload).hexdigest()
        mask = cv2.imdecode(
            np.frombuffer(mask_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise OSError(f"OpenCV could not decode PlantDoc mask: {mask_path}")
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        if not np.any(mask):
            raise ValueError(f"Empty PlantDoc SAM mask: {mask_path}")
        mask_area = float(mask.mean())

        leaf_image, leaf_mask = preprocess_leaf_only(
            image,
            mask,
            self.image_size,
            self.leaf_crop_margin,
        )
        return {
            # Intentionally no `image` or context field.
            "leaf_image": _normalised_image_tensor(leaf_image),
            "leaf_mask": torch.from_numpy(
                np.ascontiguousarray(leaf_mask[None, ...], dtype=np.float32)
            ),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_sha256": image_hash,
            "mask_sha256": mask_hash,
            "mask_area": torch.tensor(mask_area, dtype=torch.float32),
        }


def _confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    encoded = targets.astype(np.int64) * num_classes + predictions.astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def classification_summary(confusion: np.ndarray) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    present = support > 0
    total = int(confusion.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1_present": float(f1[present].mean()) if np.any(present) else 0.0,
        "macro_f1_all_classes": float(f1.mean()),
        "balanced_accuracy_present": (
            float(recall[present].mean()) if np.any(present) else 0.0
        ),
        "classes_with_support": int(present.sum()),
        "support": support.astype(int).tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def _topk_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    order = np.argsort(-probabilities, axis=1)
    result: dict[str, float] = {}
    for requested in ks:
        k = min(int(requested), probabilities.shape[1])
        result[f"top{k}_accuracy"] = float(
            np.any(order[:, :k] == targets[:, None], axis=1).mean()
        )
    return result


def grouped_bootstrap_ci(
    targets: np.ndarray,
    predictions: np.ndarray,
    exact_hashes: Sequence[str],
    *,
    num_classes: int,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    if repeats <= 0:
        return {}
    group_to_indices: dict[str, list[int]] = {}
    for index, digest in enumerate(exact_hashes):
        group_to_indices.setdefault(str(digest), []).append(index)
    groups = list(group_to_indices.values())
    fixed_present = np.bincount(targets, minlength=num_classes) > 0
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    macro_f1: list[float] = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate(
            [np.asarray(groups[group_index], dtype=np.int64) for group_index in sampled]
        )
        summary = classification_summary(
            _confusion_matrix(targets[indices], predictions[indices], num_classes)
        )
        accuracies.append(float(summary["accuracy"]))
        values = np.asarray(summary["f1"], dtype=np.float64)
        macro_f1.append(float(values[fixed_present].mean()))

    def interval(values: Sequence[float]) -> list[float]:
        return [float(value) for value in np.percentile(values, (2.5, 97.5))]

    return {
        "accuracy_95ci": interval(accuracies),
        "macro_f1_present_95ci": interval(macro_f1),
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_lookup(
    lookup: Sequence[int],
    *,
    class_names: Sequence[str],
    target_names: Sequence[str],
    name: str,
) -> list[int]:
    values = [int(value) for value in lookup]
    if len(values) != len(class_names):
        raise ValueError(f"Checkpoint {name} length differs from class_names")
    if not target_names or min(values) < 0 or max(values) >= len(target_names):
        raise ValueError(f"Checkpoint {name} contains an out-of-range index")
    return values


def _optional_auxiliary(
    checkpoint: Mapping[str, Any],
    *,
    names_key: str,
    lookup_key: str,
    class_names: Sequence[str],
) -> tuple[list[str], np.ndarray] | None:
    names_present = names_key in checkpoint
    lookup_present = lookup_key in checkpoint
    if names_present != lookup_present:
        raise ValueError(
            f"Checkpoint must contain both {names_key!r} and {lookup_key!r}, or neither"
        )
    if not names_present:
        return None
    names = [str(value) for value in checkpoint[names_key]]
    lookup = _validate_lookup(
        checkpoint[lookup_key],
        class_names=class_names,
        target_names=names,
        name=lookup_key,
    )
    return names, np.asarray(lookup, dtype=np.int64)


def _load_frozen_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FinalModelV5, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 5:
        raise ValueError("Only format_version=5 checkpoints are accepted")
    protocol = str(checkpoint.get("protocol", ""))
    if "plantvillage-only" not in protocol.casefold():
        raise ValueError(f"Checkpoint violates source-only protocol: {protocol!r}")
    required = (
        "model_state_dict",
        "model_config",
        "class_names",
        "plant_names",
        "symptom_names",
        "class_to_plant",
        "class_to_symptom",
        "robust_score",
        "validation",
        "epoch",
        "train_config",
        "inference_rule",
        "run_identity",
        "source_only",
        "training_complete",
        "checkpoint_role",
        "source_acceptance_passed",
        "source_acceptance_sha256",
    )
    for key in required:
        if key not in checkpoint:
            raise ValueError(f"V5 checkpoint lacks required field: {key}")
    if checkpoint["inference_rule"] != "direct_joint_feature_fusion":
        raise ValueError(
            "V5 final evaluation accepts only direct joint feature-fusion inference"
        )
    if checkpoint["source_only"] is not True:
        raise ValueError("V5 final checkpoint is not marked source-only")
    if checkpoint["training_complete"] is not True:
        raise ValueError("Interim/resume/smoke checkpoints cannot consume PlantDoc")
    if checkpoint["checkpoint_role"] != "source_selected_final":
        raise ValueError("PlantDoc requires the sealed source-selected final checkpoint")
    if checkpoint["source_acceptance_passed"] is not True:
        raise ValueError("Checkpoint did not pass the full-source acceptance protocol")
    source_report_path = checkpoint_path.parent / "source_acceptance_v5.json"
    if not source_report_path.is_file():
        raise FileNotFoundError(
            f"Sealed checkpoint lacks its source acceptance report: {source_report_path}"
        )
    if sha256_file(source_report_path) != checkpoint["source_acceptance_sha256"]:
        raise ValueError("Source acceptance report digest differs from sealed checkpoint")

    train_config = checkpoint["train_config"]
    if not isinstance(train_config, Mapping):
        raise ValueError("V5 train_config must be a mapping")
    if train_config.get("max_train_batches") is not None or train_config.get(
        "max_val_batches"
    ) is not None:
        raise ValueError("Truncated/smoke source runs cannot consume PlantDoc")
    if int(train_config.get("epochs", 0)) < 55:
        raise ValueError("The frozen 55-epoch source curriculum was not completed")

    class_names = [str(value) for value in checkpoint["class_names"]]
    if len(class_names) != EXPECTED_JOINT_CLASSES or len(set(class_names)) != len(
        class_names
    ):
        raise ValueError(
            f"Expected {EXPECTED_JOINT_CLASSES} unique joint classes, "
            f"got {len(class_names)}"
        )
    model_config = dict(checkpoint["model_config"])
    configured_classes = model_config.get(
        "num_disease_classes", model_config.get("num_joint_classes")
    )
    if configured_classes is not None and int(configured_classes) != len(class_names):
        raise ValueError("model_config joint class count differs from class_names")

    plant = _optional_auxiliary(
        checkpoint,
        names_key="plant_names",
        lookup_key="class_to_plant",
        class_names=class_names,
    )
    symptom = _optional_auxiliary(
        checkpoint,
        names_key="symptom_names",
        lookup_key="class_to_symptom",
        class_names=class_names,
    )
    run_identity = checkpoint["run_identity"]
    if not isinstance(run_identity, Mapping) or not run_identity.get("fingerprint"):
        raise ValueError("V5 checkpoint has no valid frozen run identity")
    dependency_hashes = run_identity.get("dependency_sha256")
    if not isinstance(dependency_hashes, Mapping):
        raise ValueError("V5 run identity lacks dependency code hashes")
    expected_model_hash = dependency_hashes.get("final_model_v5.py")
    active_model_path = Path(__file__).resolve().with_name("final_model_v5.py")
    if not expected_model_hash or sha256_file(active_model_path) != expected_model_hash:
        raise ValueError(
            "Active final_model_v5.py differs from the code frozen by training"
        )
    identity_expectations = {
        "model_config": model_config,
        "class_names": class_names,
        "plant_names": plant[0] if plant is not None else None,
        "symptom_names": symptom[0] if symptom is not None else None,
        "class_to_plant": plant[1].tolist() if plant is not None else None,
        "class_to_symptom": symptom[1].tolist() if symptom is not None else None,
        "inference_rule": checkpoint["inference_rule"],
    }
    for key, expected in identity_expectations.items():
        if run_identity.get(key) != expected:
            raise ValueError(f"Top-level checkpoint and run_identity {key} differ")
    if plant is not None and "class_to_plant" in model_config:
        if [int(value) for value in model_config["class_to_plant"]] != plant[1].tolist():
            raise ValueError("Top-level and model_config class_to_plant differ")
    if symptom is not None and "class_to_symptom" in model_config:
        if [int(value) for value in model_config["class_to_symptom"]] != symptom[1].tolist():
            raise ValueError("Top-level and model_config class_to_symptom differ")

    model = FinalModelV5(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for attribute, auxiliary in (
        ("class_to_plant", plant),
        ("class_to_symptom", symptom),
    ):
        if auxiliary is not None and hasattr(model, attribute):
            loaded = getattr(model, attribute).detach().cpu().tolist()
            if loaded != auxiliary[1].tolist():
                raise ValueError(f"Loaded model {attribute} differs from checkpoint")
    model.to(device).eval()
    return model, checkpoint


def _auxiliary_summary(
    *,
    names: Sequence[str],
    lookup: np.ndarray,
    joint_targets: np.ndarray,
    joint_predictions: np.ndarray,
    head_predictions: np.ndarray,
) -> dict[str, Any]:
    targets = lookup[joint_targets]
    joint_derived = lookup[joint_predictions]
    summary = classification_summary(
        _confusion_matrix(targets, head_predictions, len(names))
    )
    summary.update(
        {
            "available": True,
            "names": list(names),
            "joint_derived_accuracy": float((joint_derived == targets).mean()),
            "head_vs_joint_agreement": float((head_predictions == joint_derived).mean()),
        }
    )
    return summary


@torch.inference_mode()
def evaluate_final(
    *,
    checkpoint_path: Path,
    test_csv: Path,
    mask_root: Path,
    plantdoc_root: Path | None,
    output_dir: Path,
    batch_size: int,
    workers: int,
    amp: bool,
    bootstrap_repeats: int,
    allow_repeat: bool,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
) -> dict[str, Any]:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats cannot be negative")
    if not DEFAULT_TARGET_ACCURACY <= target_accuracy <= 1.0:
        raise ValueError(
            f"target_accuracy is frozen at >= {DEFAULT_TARGET_ACCURACY:.2f}; "
            "the final gate may be raised but never weakened"
        )

    checkpoint_path = checkpoint_path.resolve()
    test_csv = test_csv.resolve()
    mask_root = mask_root.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not test_csv.is_file():
        raise FileNotFoundError(test_csv)
    if not mask_root.is_dir():
        raise NotADirectoryError(mask_root)

    checkpoint_digest = sha256_file(checkpoint_path)
    csv_digest = sha256_file(test_csv)
    lock_path = output_dir / "FINAL_TEST_LOCK.json"
    # Intentionally shared with V4: a prior final inspection is globally visible.
    registry_path = PROJECT_ROOT / "new_model" / "PLANTDOC_FINAL_TEST_REGISTRY.json"
    existing_lock = next(
        (path for path in (registry_path, lock_path) if path.is_file()), None
    )
    if existing_lock is not None and not allow_repeat:
        raise RuntimeError(
            "PlantDoc final evaluation was already registered. Repeated inspection "
            "must not drive model selection. Existing lock: "
            f"{existing_lock}. --allow-repeat is only for a documented audit rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reserve the global final-test slot atomically before any PlantDoc batch is
    # loaded.  If evaluation crashes, this sentinel intentionally remains as a
    # preserved failed attempt and requires manual protocol audit; it is removed
    # only after the completed registry/result/lock have all been committed.
    attempt_path = PROJECT_ROOT / "new_model" / "PLANTDOC_FINAL_TEST_IN_PROGRESS.json"
    attempt_started_at = datetime.now(timezone.utc).isoformat()
    attempt_record = {
        "status": "in_progress",
        "started_at_utc": attempt_started_at,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "plantdoc_csv": str(test_csv),
        "plantdoc_csv_sha256": csv_digest,
        "audit_repeat": bool(existing_lock is not None),
        "warning": "Do not remove without auditing a failed/interrupted final attempt",
    }
    try:
        descriptor = os.open(
            attempt_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "A PlantDoc final evaluation is already running or a prior attempt "
            f"failed before completion: {attempt_path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(attempt_record, stream, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_frozen_checkpoint(checkpoint_path, device)
    class_names = [str(value) for value in checkpoint["class_names"]]
    train_config = dict(checkpoint["train_config"])
    image_size = int(train_config.get("image_size", 0))
    if image_size <= 0:
        raise ValueError("Checkpoint train_config lacks a positive image_size")
    leaf_crop_margin = float(
        checkpoint["model_config"].get(
            "leaf_crop_margin", train_config.get("leaf_crop_margin", 0.10)
        )
    )
    plant_spec = _optional_auxiliary(
        checkpoint,
        names_key="plant_names",
        lookup_key="class_to_plant",
        class_names=class_names,
    )
    symptom_spec = _optional_auxiliary(
        checkpoint,
        names_key="symptom_names",
        lookup_key="class_to_symptom",
        class_names=class_names,
    )

    dataset = PlantDocLeafOnlyDataset(
        test_csv,
        class_names=class_names,
        image_size=image_size,
        mask_root=mask_root,
        plantdoc_root=plantdoc_root.resolve() if plantdoc_root is not None else None,
        leaf_crop_margin=leaf_crop_margin,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    joint_targets_parts: list[np.ndarray] = []
    joint_probability_parts: list[np.ndarray] = []
    plant_prediction_parts: list[np.ndarray] = []
    symptom_prediction_parts: list[np.ndarray] = []
    row_indices: list[int] = []
    image_paths: list[str] = []
    mask_paths: list[str] = []
    image_hashes: list[str] = []
    mask_hashes: list[str] = []
    mask_areas: list[float] = []

    for batch in tqdm(loader, desc="FINAL V5 leaf-only PlantDoc evaluation"):
        leaf_image = batch["leaf_image"].to(device, non_blocking=True)
        leaf_mask = batch["leaf_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            # This is the only model invocation: no context/full-image argument.
            output = model(leaf_image=leaf_image, leaf_mask=leaf_mask)
        if "disease_logits" not in output:
            raise KeyError("FinalModelV5 output lacks direct disease_logits")
        joint_logits = output["disease_logits"].float()
        if joint_logits.ndim != 2 or joint_logits.shape[1] != len(class_names):
            raise ValueError("V5 disease_logits do not span all 38 frozen classes")
        joint_probability_parts.append(joint_logits.softmax(dim=1).cpu().numpy())
        joint_targets_parts.append(batch["label"].numpy())

        if plant_spec is not None:
            if "plant_logits" not in output:
                raise KeyError("Checkpoint has plant mapping but model lacks plant_logits")
            logits = output["plant_logits"].float()
            if logits.ndim != 2 or logits.shape[1] != len(plant_spec[0]):
                raise ValueError("plant_logits dimension differs from plant_names")
            plant_prediction_parts.append(logits.argmax(dim=1).cpu().numpy())
        if symptom_spec is not None:
            if "symptom_logits" not in output:
                raise KeyError("Checkpoint has symptom mapping but model lacks symptom_logits")
            logits = output["symptom_logits"].float()
            if logits.ndim != 2 or logits.shape[1] != len(symptom_spec[0]):
                raise ValueError("symptom_logits dimension differs from symptom_names")
            symptom_prediction_parts.append(logits.argmax(dim=1).cpu().numpy())

        row_indices.extend(int(value) for value in batch["row_index"].tolist())
        image_paths.extend(str(value) for value in batch["image_path"])
        mask_paths.extend(str(value) for value in batch["mask_path"])
        image_hashes.extend(str(value) for value in batch["image_sha256"])
        mask_hashes.extend(str(value) for value in batch["mask_sha256"])
        mask_areas.extend(float(value) for value in batch["mask_area"].tolist())

    if not joint_targets_parts:
        raise RuntimeError("PlantDoc evaluation produced no batches")
    targets = np.concatenate(joint_targets_parts)
    probabilities = np.concatenate(joint_probability_parts)
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(-probabilities, axis=1)
    confidence = probabilities[np.arange(len(targets)), predictions]

    joint_confusion = _confusion_matrix(
        targets, predictions, EXPECTED_JOINT_CLASSES
    )
    joint_metrics = classification_summary(joint_confusion)
    joint_metrics.update(_topk_metrics(probabilities, targets))
    true_probability = probabilities[np.arange(len(targets)), targets]
    joint_metrics["negative_log_likelihood"] = float(
        -np.log(np.clip(true_probability, 1e-12, 1.0)).mean()
    )

    plant_metrics: dict[str, Any] = {"available": False}
    plant_predictions: np.ndarray | None = None
    if plant_spec is not None:
        plant_predictions = np.concatenate(plant_prediction_parts)
        plant_metrics = _auxiliary_summary(
            names=plant_spec[0],
            lookup=plant_spec[1],
            joint_targets=targets,
            joint_predictions=predictions,
            head_predictions=plant_predictions,
        )
    symptom_metrics: dict[str, Any] = {"available": False}
    symptom_predictions: np.ndarray | None = None
    if symptom_spec is not None:
        symptom_predictions = np.concatenate(symptom_prediction_parts)
        symptom_metrics = _auxiliary_summary(
            names=symptom_spec[0],
            lookup=symptom_spec[1],
            joint_targets=targets,
            joint_predictions=predictions,
            head_predictions=symptom_predictions,
        )

    plantdoc_content_digest = ordered_data_fingerprint(
        row_indices, targets.tolist(), image_hashes, mask_hashes
    )
    bootstrap = grouped_bootstrap_ci(
        targets,
        predictions,
        image_hashes,
        num_classes=EXPECTED_JOINT_CLASSES,
        repeats=bootstrap_repeats,
        seed=2026,
    )
    gate_passed = bool(joint_metrics["accuracy"] >= target_accuracy)
    gate = {
        "metric": "joint_38_accuracy",
        "target": float(target_accuracy),
        "actual": float(joint_metrics["accuracy"]),
        "passed": gate_passed,
        "status": "PASS" if gate_passed else "FAIL",
        "failure_exit_code": 0 if gate_passed else 2,
    }

    predictions_payload: dict[str, Any] = {
        "row_index": row_indices,
        "image_path": image_paths,
        "mask_path": mask_paths,
        "image_sha256": image_hashes,
        "mask_sha256": mask_hashes,
        "mask_area_fraction": mask_areas,
        "true_index": targets,
        "true_class": [class_names[index] for index in targets],
        "pred_index": predictions,
        "pred_class": [class_names[index] for index in predictions],
        "confidence": confidence,
        "correct": predictions == targets,
        "top3_indices": ["|".join(map(str, row[:3])) for row in order],
        "top3_classes": [
            "|".join(class_names[index] for index in row[:3]) for row in order
        ],
        "top5_indices": ["|".join(map(str, row[:5])) for row in order],
        "top5_classes": [
            "|".join(class_names[index] for index in row[:5]) for row in order
        ],
    }
    if plant_spec is not None and plant_predictions is not None:
        plant_targets = plant_spec[1][targets]
        predictions_payload.update(
            {
                "plant_true": [plant_spec[0][index] for index in plant_targets],
                "plant_head_pred": [
                    plant_spec[0][index] for index in plant_predictions
                ],
            }
        )
    if symptom_spec is not None and symptom_predictions is not None:
        symptom_targets = symptom_spec[1][targets]
        predictions_payload.update(
            {
                "symptom_true": [
                    symptom_spec[0][index] for index in symptom_targets
                ],
                "symptom_head_pred": [
                    symptom_spec[0][index] for index in symptom_predictions
                ],
            }
        )
    pd.DataFrame(predictions_payload).sort_values("row_index").to_csv(
        output_dir / "predictions.csv", index=False
    )

    per_class = pd.DataFrame(
        {
            "class_index": np.arange(EXPECTED_JOINT_CLASSES),
            "class_name": class_names,
            "present_in_plantdoc": np.asarray(joint_metrics["support"]) > 0,
            "support": joint_metrics["support"],
            "precision": joint_metrics["precision"],
            "recall": joint_metrics["recall"],
            "f1": joint_metrics["f1"],
        }
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        joint_confusion,
        index=[f"true::{name}" for name in class_names],
        columns=[f"pred::{name}" for name in class_names],
    ).to_csv(output_dir / "confusion_matrix.csv", index_label="class")

    result: dict[str, Any] = {
        "evaluation_protocol": "V5 frozen one-way leaf-only PlantDoc final test",
        "model_selection_used": False,
        "test_time_adaptation_used": False,
        "tta_used": False,
        "class_masking_used": False,
        "context_input_used": False,
        "inference_rule": checkpoint["inference_rule"],
        "preprocessing": (
            "SAM bbox -> neutralise outside-mask pixels -> "
            "aspect-preserving ImageNet-neutral letterbox"
        ),
        "prediction_space": "all 38 frozen checkpoint classes",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "source_validation_robust_score": float(checkpoint["robust_score"]),
        "plantdoc_csv": str(test_csv),
        "plantdoc_csv_sha256": csv_digest,
        "plantdoc_ordered_image_mask_sha256": plantdoc_content_digest,
        "samples": int(len(targets)),
        "joint_classes": EXPECTED_JOINT_CLASSES,
        "plantdoc_classes_with_support": int(
            np.count_nonzero(np.bincount(targets, minlength=EXPECTED_JOINT_CLASSES))
        ),
        "absent_checkpoint_classes": [
            class_names[index]
            for index in range(EXPECTED_JOINT_CLASSES)
            if not np.any(targets == index)
        ],
        "image_size": image_size,
        "leaf_crop_margin": leaf_crop_margin,
        "device": str(device),
        "joint_38": joint_metrics,
        "plant_auxiliary": plant_metrics,
        "symptom_auxiliary": symptom_metrics,
        "grouped_bootstrap": bootstrap,
        "accuracy_gate": gate,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
    }
    lock_record = {
        "warning": "Final-test lock: do not use PlantDoc results for model selection",
        "evaluation_protocol": result["evaluation_protocol"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_started_at_utc": attempt_started_at,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "plantdoc_csv": str(test_csv),
        "plantdoc_csv_sha256": csv_digest,
        "plantdoc_ordered_image_mask_sha256": plantdoc_content_digest,
        "samples": int(len(targets)),
        "audit_repeat": bool(existing_lock is not None),
        "model_selection_used": False,
        "context_input_used": False,
        "accuracy_gate": gate,
    }
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            raise ValueError(f"Invalid final-test registry: {registry_path}")
    else:
        registry = {
            "warning": "One-way PlantDoc final-test registry",
            "evaluations": [],
        }
    evaluations = registry.setdefault("evaluations", [])
    if not isinstance(evaluations, list):
        raise ValueError(f"Invalid evaluations list in registry: {registry_path}")
    evaluations.append(lock_record)
    _atomic_json(registry, registry_path)
    _atomic_json(lock_record, lock_path)
    _atomic_json(result, output_dir / "plantdoc_evaluation.json")
    attempt_path.unlink()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen leaf-only V5 checkpoint on PlantDoc once"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "new_model"
            / "checkpoints_curriculum_v5"
            / "final_source_dg_v5.pt"
        ),
    )
    parser.add_argument("--test-csv", default=str(PROJECT_ROOT / "test.csv"))
    parser.add_argument(
        "--mask-root", default=str(PROJECT_ROOT / "SAM2_Masks" / "Test")
    )
    parser.add_argument("--plantdoc-root", default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=DEFAULT_TARGET_ACCURACY,
        help="Frozen final target; may be raised but never below 0.70",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Audit rerun only; never use a repeat for model selection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.resolve().parent / "plantdoc_final_eval"
    )
    print(
        "FINAL TEST ONLY: PlantDoc will not be used for training, adaptation, "
        "checkpoint selection, class masking, threshold tuning, or context input."
    )
    result = evaluate_final(
        checkpoint_path=checkpoint_path,
        test_csv=Path(args.test_csv),
        mask_root=Path(args.mask_root),
        plantdoc_root=Path(args.plantdoc_root) if args.plantdoc_root else None,
        output_dir=output_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=not args.no_amp,
        bootstrap_repeats=args.bootstrap_repeats,
        allow_repeat=args.allow_repeat,
        target_accuracy=args.target_accuracy,
    )
    joint = result["joint_38"]
    gate = result["accuracy_gate"]
    print(
        f"PlantDoc joint-38: accuracy={joint['accuracy']:.4f}, "
        f"macro-F1(all)={joint['macro_f1_all_classes']:.4f}, "
        f"macro-F1(present)={joint['macro_f1_present']:.4f}, "
        f"top-3={joint['top3_accuracy']:.4f}, top-5={joint['top5_accuracy']:.4f}"
    )
    print(
        f"Accuracy gate: {gate['status']} "
        f"({gate['actual']:.4f} vs target {gate['target']:.4f})"
    )
    print(f"Audit artifacts: {output_dir.resolve()}")
    return int(gate["failure_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
