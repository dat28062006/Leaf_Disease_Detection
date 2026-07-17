"""PlantVillage-only data utilities for source-only domain generalization.

The module deliberately never reads ``test.csv`` or PlantDoc.  It provides:

* deterministic, grouped and stratified source splits;
* grouping of repeated capture basenames and optional exact file duplicates;
* robust workspace-relative image and SAM2-mask resolution;
* weak/strong views with fixed proxy domains D0--D4; and
* a multiprocessing-safe :class:`SourceDGDataset` returning labels/metadata.

Typical usage::

    manifest = build_source_splits(seed=2026)
    train_ds = SourceDGDataset(
        manifest, partition="train", proxy_domain="D1", training=True,
    )
    ood_val_ds = SourceDGDataset(
        manifest, partition="ood_val", proxy_domain="D3", training=False,
    )

Split original images first and construct each dataset from exactly one
partition.  Synthetic views and background donors are then guaranteed to stay
inside that partition.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote

# Prevent Albumentations from attempting a network version check on workers.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "train.csv"
DEFAULT_MASK_ROOT = PROJECT_ROOT / "SAM2_Masks"
MANIFEST_ALGORITHM_VERSION = 4

DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "id_val": 0.10,
    "ood_val": 0.10,
    "proxy_test": 0.10,
}

PROXY_DOMAINS: dict[str, int] = {
    "D0": 0,  # clean / ordinary source view
    "D1": 1,  # illumination and colour shift
    "D2": 2,  # camera/compression degradation
    "D3": 3,  # geometry and mask-aware background randomisation
    "D4": 4,  # segmentation-mask corruption
}

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGENET_MEAN_RGB = tuple(int(round(value * 255.0)) for value in IMAGENET_MEAN)

_UUID_PREFIX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___",
    flags=re.IGNORECASE,
)
_COPY_SUFFIX = re.compile(
    r"(?:[\s._-]+(?:copy|duplicate|dup)(?:[\s._-]*\d+)?|\s*\(\d+\))$",
    flags=re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^0-9a-z]+", flags=re.IGNORECASE)
_KNOWN_VIEW_SERIES = re.compile(
    r"^(GH_HL|GHLB2|GHLB2ES)\s+Leaf\s+(\d+)\.\d+$",
    flags=re.IGNORECASE,
)
_KNOWN_DAY_SERIES = re.compile(
    r"^(GHLB|GHLB_PS|GHLB2|GHLB2ES)\s+Leaf\s+"
    r"(\d+(?:\.\d+)?)\s+Day\s+\d+$",
    flags=re.IGNORECASE,
)


def _stable_int(*parts: object, bits: int = 32) -> int:
    """Return a process-independent integer hash (unlike Python ``hash``)."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest_size = max(4, bits // 8)
    value = int.from_bytes(
        hashlib.blake2b(payload, digest_size=digest_size).digest(), "little"
    )
    return value & ((1 << bits) - 1)


def _normalise_relpath(raw_path: str | os.PathLike[str]) -> str:
    """Validate and normalise a CSV path without allowing absolute paths."""

    raw = unquote(str(raw_path).strip().strip('"').strip("'"))
    if not raw:
        raise ValueError("Empty image path in source CSV")
    if PureWindowsPath(raw).is_absolute() or Path(raw).is_absolute():
        raise ValueError(f"Source image path must be relative, got: {raw!r}")
    raw = raw.replace("\\", "/")
    pure = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe relative image path: {raw!r}")
    if not pure.parts or pure.parts[0].casefold() != "train":
        raise ValueError(
            "Source-only utilities accept only paths below Train/; "
            f"got {raw!r}"
        )
    return pure.as_posix()


def resolve_source_path(
    relative_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    require_exists: bool = True,
) -> Path:
    """Resolve a safe PlantVillage path and reject traversal outside the root."""

    rel = _normalise_relpath(relative_path)
    root = Path(project_root).resolve()
    candidate = (root / Path(*PurePosixPath(rel).parts)).resolve()
    try:
        inside_root = os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Resolved path escapes project root: {relative_path!r}")
    if require_exists and not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def sam_mask_relative_path(image_relative_path: str | os.PathLike[str]) -> str:
    """Map ``Train/class/image.ext`` to ``SAM2_Masks/Train/class/image.png``."""

    rel = PurePosixPath(_normalise_relpath(image_relative_path))
    # with_suffix removes only the final extension, matching generate_sam2_masks.py.
    return (PurePosixPath("SAM2_Masks") / rel.with_suffix(".png")).as_posix()


def resolve_sam_mask_path(
    image_relative_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    require_exists: bool = False,
) -> Path:
    """Resolve the local SAM2 mask corresponding to a PlantVillage image."""

    root = Path(project_root).resolve()
    rel = PurePosixPath(sam_mask_relative_path(image_relative_path))
    candidate = (root / Path(*rel.parts)).resolve()
    try:
        inside_root = os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Resolved mask path escapes project root: {rel}")
    if require_exists and not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def canonical_capture_key(image_relative_path: str | os.PathLike[str]) -> str:
    """Return a conservative capture key for UUID-prefixed PlantVillage names.

    The parent class folder is included to avoid joining unrelated plants that
    happen to share acquisition numbers.  A leading UUID and obvious copy
    suffixes are removed.  Only known PlantVillage tomato acquisition grammars
    additionally collapse view suffixes or ``Day N`` for the same physical leaf.
    """

    rel = PurePosixPath(_normalise_relpath(image_relative_path))
    stem = unquote(rel.stem).strip()
    stem = _UUID_PREFIX.sub("", stem)
    stem = _COPY_SUFFIX.sub("", stem)
    # PlantVillage contains explicitly named views/days of the same physical
    # tomato leaf.  Canonicalise only the known acquisition grammar; a generic
    # trailing-number rule would incorrectly merge unrelated captures.
    view_match = _KNOWN_VIEW_SERIES.fullmatch(stem)
    day_match = _KNOWN_DAY_SERIES.fullmatch(stem)
    if view_match:
        stem = f"{view_match.group(1)} Leaf {view_match.group(2)}"
    elif day_match:
        # Preserve the full leaf ID (for example 23.2), remove only Day N.
        stem = f"{day_match.group(1)} Leaf {day_match.group(2)}"
    stem = _NON_ALNUM.sub("_", stem).strip("_").casefold()
    parent = _NON_ALNUM.sub("_", rel.parent.name).strip("_").casefold()
    if not stem:
        stem = hashlib.blake2b(rel.name.encode("utf-8"), digest_size=8).hexdigest()
    return f"{parent}::{stem}"


def infer_acquisition_tag(image_relative_path: str | os.PathLike[str]) -> str:
    """Infer a coarse acquisition/source prefix from a PlantVillage filename."""

    rel = PurePosixPath(_normalise_relpath(image_relative_path))
    stem = _UUID_PREFIX.sub("", unquote(rel.stem))
    prefix = re.split(r"\d", stem, maxsplit=1)[0]
    prefix = _NON_ALNUM.sub("_", prefix).strip("_").upper()
    return prefix or "UNKNOWN"


def infer_plant_and_health(
    plant_disease: str, image_relative_path: str | os.PathLike[str]
) -> tuple[str, str, bool]:
    """Infer plant, disease text and healthy/diseased status.

    Folder metadata is preferred because labels such as ``Pepper,_bell`` and
    ``Cherry_(including_sour)`` are ambiguous to split on whitespace.
    """

    parent = PurePosixPath(_normalise_relpath(image_relative_path)).parent.name
    if "___" in parent:
        plant_raw, disease_raw = parent.split("___", maxsplit=1)
        plant = re.sub(r"_+", " ", plant_raw).strip()
        disease = re.sub(r"_+", " ", disease_raw).strip()
    else:
        label = re.sub(r"\s+", " ", str(plant_disease)).strip()
        plant = label.split(" ", maxsplit=1)[0]
        disease = label[len(plant) :].strip()
    healthy = disease.casefold() == "healthy" or str(plant_disease).casefold().endswith(
        " healthy"
    )
    return plant, disease, healthy


def canonical_disease_type(disease_name: str) -> str:
    """Return the stable phenotype-family label used by V5's auxiliary head.

    These labels are deliberately *auxiliary*: similarly named diseases on
    different plants are useful visual supervision but need not share a causal
    pathogen.  The primary target therefore remains the 38-way joint label.
    """

    value = _NON_ALNUM.sub("_", str(disease_name).casefold()).strip("_")
    if not value:
        raise ValueError("Disease type cannot be empty")
    aliases = {
        "cercospora_leaf_spot_gray_leaf_spot": "cercospora_gray_leaf_spot",
        "esca_black_measles": "esca_black_measles",
        "haunglongbing_citrus_greening": "huanglongbing_citrus_greening",
        "leaf_blight_isariopsis_leaf_spot": "isariopsis_leaf_spot",
        "spider_mites_two_spotted_spider_mite": "two_spotted_spider_mite",
    }
    return aliases.get(value, value)


def _exact_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _assign_group_ids(
    frame: pd.DataFrame,
    *,
    project_root: Path,
    hash_exact_duplicates: bool,
    hash_workers: int,
) -> pd.DataFrame:
    """Union rows sharing capture keys or exact bytes into stable groups."""

    result = frame.copy()
    paths = [
        resolve_source_path(path, project_root=project_root)
        for path in result["image_rel"]
    ]
    sizes = np.asarray([path.stat().st_size for path in paths], dtype=np.int64)
    captures = [canonical_capture_key(path) for path in result["image_rel"]]
    digests = [""] * len(result)

    if hash_exact_duplicates:
        size_counts = pd.Series(sizes).value_counts().to_dict()
        candidate_indices = [
            index for index, size in enumerate(sizes) if size_counts[int(size)] > 1
        ]

        def hash_index(index: int) -> tuple[int, str]:
            return index, _exact_digest(paths[index])

        if hash_workers > 1 and candidate_indices:
            with ThreadPoolExecutor(max_workers=hash_workers) as executor:
                for index, digest in executor.map(hash_index, candidate_indices):
                    digests[index] = digest
        else:
            for index in candidate_indices:
                digests[index] = _exact_digest(paths[index])

    union_find = _UnionFind(len(result))
    capture_owner: dict[str, int] = {}
    digest_owner: dict[str, int] = {}
    for index, (capture, digest) in enumerate(zip(captures, digests)):
        if capture in capture_owner:
            union_find.union(index, capture_owner[capture])
        else:
            capture_owner[capture] = index
        if digest:
            if digest in digest_owner:
                union_find.union(index, digest_owner[digest])
            else:
                digest_owner[digest] = index

    component_tokens: dict[int, list[str]] = {}
    for index, (capture, digest) in enumerate(zip(captures, digests)):
        root = union_find.find(index)
        tokens = component_tokens.setdefault(root, [])
        tokens.append(f"capture:{capture}")
        if digest:
            tokens.append(f"exact:{digest}")

    component_id = {
        root: "g_"
        + hashlib.blake2b(min(tokens).encode("utf-8"), digest_size=10).hexdigest()
        for root, tokens in component_tokens.items()
    }
    result["capture_key"] = captures
    result["exact_hash"] = digests
    result["file_size"] = sizes
    result["group_id"] = [component_id[union_find.find(i)] for i in range(len(result))]
    return result


def load_source_dataframe(
    csv_path: str | os.PathLike[str] = DEFAULT_TRAIN_CSV,
    root: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    project_root: str | os.PathLike[str] | None = None,
    hash_exact_duplicates: bool = True,
    hash_workers: int | None = None,
) -> pd.DataFrame:
    """Load and annotate PlantVillage ``train.csv`` without touching test data."""

    # ``project_root`` is a backwards-compatible keyword alias; the compact
    # trainer-facing API is load_source_dataframe(csv_path, root).
    source_root = Path(project_root if project_root is not None else root).resolve()
    csv_path = Path(csv_path).resolve()
    if csv_path.name.casefold().startswith("test"):
        raise ValueError("Source-only utilities refuse test CSV files")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    required = {"image", "plant_disease"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Source CSV is empty: {csv_path}")

    frame = frame.loc[:, ["image", "plant_disease"]].copy()
    frame["image_rel"] = frame["image"].map(_normalise_relpath)
    if frame["image_rel"].duplicated().any():
        # Identical rows belong to one group, but retaining duplicate samples
        # would still bias training.  Keep the first deterministic occurrence.
        frame = frame.drop_duplicates(subset="image_rel", keep="first").copy()
    frame = frame.sort_values("image_rel", kind="mergesort").reset_index(drop=True)

    inferred = [
        infer_plant_and_health(label, path)
        for label, path in zip(frame["plant_disease"], frame["image_rel"])
    ]
    frame["plant_name"] = [item[0] for item in inferred]
    frame["disease_name"] = [item[1] for item in inferred]
    frame["disease_type_name"] = frame["disease_name"].map(canonical_disease_type)
    frame["is_healthy"] = [item[2] for item in inferred]
    frame["health_index"] = frame["is_healthy"].astype(np.int64)
    frame["acquisition_tag"] = frame["image_rel"].map(infer_acquisition_tag)
    frame["mask_rel"] = frame["image_rel"].map(sam_mask_relative_path)

    class_names = sorted(frame["plant_disease"].astype(str).unique())
    plant_names = sorted(frame["plant_name"].astype(str).unique())
    disease_type_names = sorted(frame["disease_type_name"].astype(str).unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    plant_to_index = {name: index for index, name in enumerate(plant_names)}
    disease_type_to_index = {
        name: index for index, name in enumerate(disease_type_names)
    }
    frame["class_index"] = frame["plant_disease"].map(class_to_index).astype(np.int64)
    frame["plant_index"] = frame["plant_name"].map(plant_to_index).astype(np.int64)
    frame["disease_type_index"] = (
        frame["disease_type_name"].map(disease_type_to_index).astype(np.int64)
    )

    workers = hash_workers
    if workers is None:
        workers = min(8, max(1, os.cpu_count() or 1))
    frame = _assign_group_ids(
        frame,
        project_root=source_root,
        hash_exact_duplicates=hash_exact_duplicates,
        hash_workers=max(0, int(workers)),
    )
    frame.attrs["class_to_index"] = class_to_index
    frame.attrs["plant_to_index"] = plant_to_index
    frame.attrs["disease_type_to_index"] = disease_type_to_index
    frame.attrs["source_csv"] = str(csv_path)
    return frame


def _fold_counts(ratios: Mapping[str, float], n_folds: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    names = list(ratios)
    values = np.asarray([float(ratios[name]) for name in names], dtype=np.float64)
    if not names or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("Split ratios must be non-negative and sum to 1")
    positive = values > 0
    if int(positive.sum()) > n_folds:
        raise ValueError("n_folds is smaller than the number of non-empty splits")

    raw = values * n_folds
    counts = np.floor(raw).astype(int)
    counts[(positive) & (counts == 0)] = 1
    while counts.sum() > n_folds:
        candidates = np.where(counts > 1)[0]
        if not len(candidates):
            raise ValueError("Unable to allocate folds for the requested ratios")
        index = min(candidates, key=lambda i: (raw[i] - counts[i], -counts[i]))
        counts[index] -= 1
    while counts.sum() < n_folds:
        remainders = raw - counts
        index = max(range(len(counts)), key=lambda i: (remainders[i], values[i], -i))
        counts[index] += 1
    return dict(zip(names, counts.tolist()))


def grouped_stratified_split(
    frame: pd.DataFrame,
    *,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 2026,
    n_folds: int = 10,
) -> pd.DataFrame:
    """Assign deterministic stratified folds without splitting ``group_id``.

    ``StratifiedGroupKFold`` first creates balanced atomic folds.  Whole folds
    are then allocated to named partitions according to ``ratios``.  The
    default 10 folds exactly implement a 70/10/10/10 split.
    """

    required = {"plant_disease", "class_index", "group_id", "image_rel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataframe lacks split columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe")

    result = frame.sort_values("image_rel", kind="mergesort").reset_index(drop=True).copy()
    groups_per_class = result.groupby("class_index")["group_id"].nunique()
    effective_folds = min(int(n_folds), int(groups_per_class.min()))
    positive_splits = sum(float(value) > 0 for value in ratios.values())
    if effective_folds < positive_splits:
        raise ValueError(
            "Not enough independent capture groups per class for all splits: "
            f"minimum={groups_per_class.min()}, required={positive_splits}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=effective_folds, shuffle=True, random_state=int(seed)
    )
    y = result["class_index"].to_numpy()
    groups = result["group_id"].astype(str).to_numpy()
    fold_ids = np.full(len(result), -1, dtype=np.int64)
    for fold, (_, held_indices) in enumerate(
        splitter.split(np.zeros(len(result), dtype=np.uint8), y, groups)
    ):
        fold_ids[held_indices] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("Some samples were not assigned to a fold")

    counts = _fold_counts(ratios, effective_folds)
    fold_order = np.random.default_rng(int(seed)).permutation(effective_folds)
    fold_to_split: dict[int, str] = {}
    cursor = 0
    for split_name, count in counts.items():
        for fold in fold_order[cursor : cursor + count]:
            fold_to_split[int(fold)] = split_name
        cursor += count

    result["fold"] = fold_ids
    result["split"] = [fold_to_split[int(fold)] for fold in fold_ids]
    if result.groupby("group_id")["split"].nunique().max() != 1:
        raise RuntimeError("Group leakage detected after split assignment")

    expected_classes = set(result["class_index"].unique())
    for split_name, count in counts.items():
        if count <= 0:
            continue
        present = set(result.loc[result["split"] == split_name, "class_index"].unique())
        if present != expected_classes:
            missing_classes = sorted(expected_classes.difference(present))
            raise RuntimeError(
                f"Split {split_name!r} is missing class indices {missing_classes}"
            )
    result.attrs.update(frame.attrs)
    result.attrs["split_seed"] = int(seed)
    result.attrs["split_ratios"] = dict(ratios)
    result.attrs["n_folds"] = effective_folds
    return result


def make_grouped_splits(
    dataframe: pd.DataFrame,
    seed: int = 2026,
    fractions: Mapping[str, float] | Sequence[float] = DEFAULT_SPLIT_RATIOS,
    *,
    n_folds: int = 10,
) -> dict[str, pd.DataFrame]:
    """Trainer-facing wrapper returning one dataframe per leakage-safe split.

    ``fractions`` may be a mapping of split names to fractions or a sequence
    of four values ordered as ``train, id_val, ood_val, proxy_test``.
    """

    if isinstance(fractions, Mapping):
        ratios = dict(fractions)
    else:
        values = tuple(float(value) for value in fractions)
        if len(values) != 4:
            raise ValueError(
                "Sequence fractions must contain train/id_val/ood_val/proxy_test"
            )
        ratios = dict(zip(DEFAULT_SPLIT_RATIOS, values))
    manifest = grouped_stratified_split(
        dataframe, ratios=ratios, seed=seed, n_folds=n_folds
    )
    output: dict[str, pd.DataFrame] = {}
    for split_name in ratios:
        split_frame = manifest.loc[manifest["split"] == split_name].copy()
        split_frame.attrs.update(manifest.attrs)
        output[split_name] = split_frame.reset_index(drop=True)
    return output


def build_source_splits(
    train_csv: str | os.PathLike[str] = DEFAULT_TRAIN_CSV,
    *,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 2026,
    n_folds: int = 10,
    hash_exact_duplicates: bool = True,
    hash_workers: int | None = None,
) -> pd.DataFrame:
    """Load PlantVillage metadata and return a complete split manifest."""

    source = load_source_dataframe(
        train_csv,
        project_root,
        hash_exact_duplicates=hash_exact_duplicates,
        hash_workers=hash_workers,
    )
    return grouped_stratified_split(source, ratios=ratios, seed=seed, n_folds=n_folds)


def seed_worker(worker_id: int) -> None:
    """Optional deterministic ``DataLoader(worker_init_fn=seed_worker)`` hook."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _letterbox_transforms(image_size: int) -> list[A.BasicTransform]:
    return [
        A.LongestMaxSize(
            max_size=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            # Neutral after ImageNet normalization; avoids teaching a black
            # letterbox cue that is common only in non-square field images.
            fill=IMAGENET_MEAN_RGB,
            fill_mask=0,
            p=1.0,
        ),
    ]


def _make_transforms(image_size: int, training: bool) -> dict[str, A.Compose]:
    weak_prefix: list[A.BasicTransform] = []
    if training:
        weak_prefix = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=IMAGENET_MEAN_RGB,
                fill_mask=0,
                p=0.5,
            ),
        ]

    strong_base: list[A.BasicTransform] = []
    if training:
        strong_base = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.Affine(
                scale=(0.85, 1.10),
                translate_percent=(-0.08, 0.08),
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                fill_mask=0,
                p=0.8,
            ),
        ]

    transforms: dict[str, A.Compose] = {
        "weak": A.Compose(weak_prefix + _letterbox_transforms(image_size)),
        "D0": A.Compose(
            strong_base
            + ([A.RandomBrightnessContrast(0.12, 0.12, p=0.5)] if training else [])
            + _letterbox_transforms(image_size)
        ),
        "D1": A.Compose(
            strong_base
            + [
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(75, 135), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.22, contrast_limit=0.20, p=1.0
                        ),
                        A.RandomShadow(
                            shadow_roi=(0.0, 0.0, 1.0, 1.0),
                            num_shadows_limit=(1, 3),
                            shadow_dimension=6,
                            shadow_intensity_range=(0.20, 0.48),
                            p=1.0,
                        ),
                    ],
                    p=1.0,
                ),
                A.HueSaturationValue(
                    # Keep hue changes deliberately small: colour is itself a
                    # disease cue, while lighting/value may vary more strongly.
                    hue_shift_limit=4,
                    sat_shift_limit=18,
                    val_shift_limit=15,
                    p=0.65,
                ),
                A.RGBShift(
                    r_shift_limit=9,
                    g_shift_limit=9,
                    b_shift_limit=9,
                    p=0.35,
                ),
            ]
            + _letterbox_transforms(image_size)
        ),
        "D2": A.Compose(
            strong_base
            + [
                A.OneOf(
                    [
                        A.Downscale(
                            scale_range=(0.65, 0.85),
                            interpolation_pair={
                                "downscale": cv2.INTER_AREA,
                                "upscale": cv2.INTER_LINEAR,
                            },
                            p=1.0,
                        ),
                        A.ImageCompression(quality_range=(45, 82), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.3, 1.0), p=1.0),
                        A.MotionBlur(blur_limit=(3, 3), p=1.0),
                    ],
                    p=1.0,
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.04),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    p=0.55,
                ),
            ]
            + _letterbox_transforms(image_size)
        ),
        "D3": A.Compose(
            [
                A.HorizontalFlip(p=0.5 if training else 0.0),
                A.Affine(
                    scale=(0.62, 1.05),
                    translate_percent=(-0.18, 0.18),
                    rotate=(-28, 28),
                    shear=(-8, 8),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=IMAGENET_MEAN_RGB,
                    fill_mask=0,
                    p=1.0,
                ),
                A.Perspective(
                    scale=(0.03, 0.10),
                    keep_size=True,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=IMAGENET_MEAN_RGB,
                    fill_mask=0,
                    p=0.70,
                ),
            ]
            + _letterbox_transforms(image_size)
        ),
        "D4": A.Compose(strong_base + _letterbox_transforms(image_size)),
    }
    return transforms


def _apply_seeded(
    transform: A.Compose, image: np.ndarray, mask: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    # Albumentations 2 owns isolated Python/NumPy RNGs; reseeding the Compose
    # makes output independent of worker scheduling and process start method.
    transform.set_random_seed(int(seed))
    transformed = transform(image=image, mask=mask)
    out_image = np.ascontiguousarray(transformed["image"], dtype=np.uint8)
    out_mask = np.ascontiguousarray(transformed["mask"] > 0, dtype=np.uint8)
    return out_image, out_mask


def _procedural_background(
    height: int,
    width: int,
    rng: np.random.Generator,
    reference_pixels: np.ndarray | None = None,
) -> np.ndarray:
    if reference_pixels is not None and len(reference_pixels) >= 32:
        pixels = reference_pixels.astype(np.float32)
        mean = np.median(pixels, axis=0)
        spread = np.maximum(np.std(pixels, axis=0), 8.0)
    else:
        mean = rng.uniform(45, 210, size=3).astype(np.float32)
        spread = rng.uniform(10, 45, size=3).astype(np.float32)

    small_h = max(3, min(12, math.ceil(height / 32)))
    small_w = max(3, min(12, math.ceil(width / 32)))
    noise = rng.normal(0.0, 1.0, size=(small_h, small_w, 3)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    x_gradient = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    gradient_gain = rng.uniform(-0.45, 0.45, size=(1, 1, 3)).astype(np.float32)
    background = mean + spread * (
        0.45 * noise + gradient_gain * (x_gradient + y_gradient)
    )
    return np.clip(background, 0, 255).astype(np.uint8)


def _corrupt_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    corrupted = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    mode = int(rng.integers(0, 5))
    if mode in (0, 1):
        kernel_size = int(rng.choice((3, 5, 7, 9, 11, 13, 15)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        operation = cv2.MORPH_ERODE if mode == 0 else cv2.MORPH_DILATE
        iterations = int(rng.integers(1, 3))
        corrupted = cv2.morphologyEx(
            corrupted, operation, kernel, iterations=iterations
        )
    elif mode == 2:
        shift_x = int(rng.integers(-16, 17))
        shift_y = int(rng.integers(-16, 17))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        corrupted = cv2.warpAffine(
            corrupted,
            matrix,
            (corrupted.shape[1], corrupted.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif mode == 3:
        height, width = corrupted.shape
        holes = int(rng.integers(1, 5))
        for _ in range(holes):
            centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            axes = (
                int(rng.integers(max(2, width // 24), max(3, width // 8))),
                int(rng.integers(max(2, height // 24), max(3, height // 8))),
            )
            cv2.ellipse(
                corrupted,
                centre,
                axes,
                float(rng.uniform(0, 180)),
                0,
                360,
                color=0,
                thickness=-1,
            )
    else:
        # Occasionally emulate a failed segmenter exactly as the existing
        # training code does: use an all-foreground fallback.
        corrupted.fill(1)
    if not np.any(corrupted):
        # Keep D4 difficult without silently erasing every foreground token.
        corrupted = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    return corrupted


def _image_tensor(image: np.ndarray, normalize: bool) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    if normalize:
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _mask_tensor(mask: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(mask > 0, dtype=np.float32)[None, ...]
    return torch.from_numpy(array)


def crop_leaf_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int,
    margin: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a leaf mask box on CPU and resize it back to a square canvas."""

    if not 0.0 <= margin <= 0.5:
        raise ValueError("leaf crop margin must be in [0, 0.5]")
    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        return image.copy(), np.ones(image.shape[:2], dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(points)
    margin_x = max(2, int(round(width * margin)))
    margin_y = max(2, int(round(height * margin)))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(image.shape[1], x + width + margin_x)
    y1 = min(image.shape[0], y + height + margin_y)
    # Remove background *before* taking the margin crop.  This makes the leaf
    # crop itself context-free (rather than relying solely on the model to
    # ignore non-leaf pixels) and matches the frozen PlantDoc V5 evaluator.
    neutralised = np.full_like(image, IMAGENET_MEAN_RGB)
    neutralised[binary > 0] = image[binary > 0]
    image_crop = neutralised[y0:y1, x0:x1]
    mask_crop = binary[y0:y1, x0:x1]
    crop_height, crop_width = image_crop.shape[:2]
    scale = image_size / max(crop_height, crop_width)
    resized_height = max(1, min(image_size, int(round(crop_height * scale))))
    resized_width = max(1, min(image_size, int(round(crop_width * scale))))
    resized_image = cv2.resize(
        image_crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_mask = cv2.resize(
        mask_crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas_image = np.full(
        (image_size, image_size, 3), IMAGENET_MEAN_RGB, dtype=np.uint8
    )
    canvas_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas_image[top : top + resized_height, left : left + resized_width] = (
        resized_image
    )
    canvas_mask[top : top + resized_height, left : left + resized_width] = resized_mask
    return canvas_image, canvas_mask


class SourceDGDataset(Dataset[dict[str, Any]]):
    """PlantVillage weak/strong dataset with fixed proxy domain recipes.

    ``partition`` should select exactly one manifest split.  If a dataframe
    containing several splits is passed without ``partition``, construction
    fails so that D3 background donors cannot leak across partitions.

    ``image_weak`` is a minimally augmented source view.  ``image_strong`` is
    the deterministic domain view selected by ``proxy_domain``.  In evaluation
    mode (``training=False``), augmentations do not vary by epoch; in training
    call :meth:`set_epoch` before each epoch to obtain a new deterministic view.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        project_root: str | os.PathLike[str] = PROJECT_ROOT,
        partition: str | None = None,
        proxy_domain: str = "D0",
        image_size: int = 224,
        training: bool | None = None,
        seed: int = 2026,
        normalize: bool = True,
        missing_mask_policy: str = "full",
        small_mask_policy: str = "full",
        background_donor_probability: float = 0.75,
        leaf_crop_margin: float = 0.10,
    ) -> None:
        if proxy_domain not in PROXY_DOMAINS:
            raise ValueError(
                f"Unknown proxy domain {proxy_domain!r}; choose {tuple(PROXY_DOMAINS)}"
            )
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if missing_mask_policy not in {"full", "empty", "error"}:
            raise ValueError("missing_mask_policy must be 'full', 'empty', or 'error'")
        if small_mask_policy not in {"full", "keep", "error"}:
            raise ValueError("small_mask_policy must be 'full', 'keep', or 'error'")
        if not 0.0 <= background_donor_probability <= 1.0:
            raise ValueError("background_donor_probability must be in [0, 1]")
        if not 0.0 <= leaf_crop_margin <= 0.5:
            raise ValueError("leaf_crop_margin must be in [0, 0.5]")

        required = {
            "image_rel",
            "plant_disease",
            "plant_name",
            "is_healthy",
            "class_index",
            "plant_index",
            "health_index",
            "disease_type_name",
            "disease_type_index",
            "group_id",
        }
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Dataframe lacks dataset columns: {sorted(missing)}")

        selected = dataframe
        if partition is not None:
            if "split" not in dataframe.columns:
                raise ValueError("partition was provided but dataframe has no split column")
            selected = dataframe.loc[dataframe["split"].astype(str) == str(partition)]
            if selected.empty:
                raise ValueError(f"No rows found for partition {partition!r}")
        elif "split" in dataframe.columns and dataframe["split"].nunique() > 1:
            raise ValueError(
                "Pass partition=... or pre-filter dataframe to one split; "
                "cross-partition background donors are forbidden"
            )

        self.frame = selected.sort_values("image_rel", kind="mergesort").reset_index(
            drop=True
        )
        self.project_root = Path(project_root).resolve()
        self.partition = (
            str(partition)
            if partition is not None
            else (
                str(self.frame["split"].iloc[0])
                if "split" in self.frame.columns
                else "unspecified"
            )
        )
        self.proxy_domain = proxy_domain
        self.domain_index = PROXY_DOMAINS[proxy_domain]
        self.image_size = int(image_size)
        self.training = self.partition == "train" if training is None else bool(training)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.missing_mask_policy = missing_mask_policy
        self.small_mask_policy = small_mask_policy
        self.background_donor_probability = float(background_donor_probability)
        self.leaf_crop_margin = float(leaf_crop_margin)
        self.epoch = 0
        self.transforms = _make_transforms(self.image_size, self.training)

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """Set the augmentation epoch; ignored in deterministic evaluation."""

        self.epoch = int(epoch)

    def _sample_seed(self, index: int, stream: str) -> int:
        epoch = self.epoch if self.training else 0
        group_id = str(self.frame.iloc[index]["group_id"])
        return _stable_int(
            self.seed,
            epoch,
            self.partition,
            self.proxy_domain,
            group_id,
            stream,
        )

    def _read_image_mask(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, bool, Path, Path]:
        row = self.frame.iloc[index]
        image_path = resolve_source_path(
            str(row["image_rel"]), project_root=self.project_root
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"OpenCV could not decode source image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = resolve_sam_mask_path(
            str(row["image_rel"]), project_root=self.project_root
        )
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_found = mask_raw is not None and bool(np.any(mask_raw))
        if not mask_found:
            if self.missing_mask_policy == "error":
                raise FileNotFoundError(f"Missing or empty SAM mask: {mask_path}")
            fill = 255 if self.missing_mask_policy == "full" else 0
            mask_raw = np.full(image.shape[:2], fill, dtype=np.uint8)
        elif mask_raw.shape != image.shape[:2]:
            mask_raw = cv2.resize(
                mask_raw,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = np.ascontiguousarray(mask_raw > 0, dtype=np.uint8)
        return image, mask, mask_found, image_path, mask_path

    def _background_from_partition(
        self,
        index: int,
        height: int,
        width: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str, str]:
        use_donor = (
            len(self.frame) > 1
            and rng.random() < self.background_donor_probability
        )
        if use_donor:
            donor_index = int(rng.integers(0, len(self.frame) - 1))
            if donor_index >= index:
                donor_index += 1
            donor_image, donor_mask, _, _, _ = self._read_image_mask(donor_index)
            if donor_image.shape[:2] != (height, width):
                donor_image = cv2.resize(
                    donor_image, (width, height), interpolation=cv2.INTER_AREA
                )
                donor_mask = cv2.resize(
                    donor_mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            background_pixels = donor_image[donor_mask == 0]
            if len(background_pixels) >= max(64, height * width // 100):
                background = _procedural_background(
                    height, width, rng, background_pixels
                )
                donor_rel = str(self.frame.iloc[donor_index]["image_rel"])
                return background, "partition_donor", donor_rel

        return _procedural_background(height, width, rng), "procedural", ""

    def _randomise_background(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        index: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str, str]:
        height, width = mask.shape
        coverage = float(np.mean(mask > 0))
        if not 0.08 <= coverage <= 0.95:
            return image, "skipped_unreliable_mask", ""
        background, source, donor_rel = self._background_from_partition(
            index, height, width, rng
        )
        # Protect the leaf boundary from SAM under-segmentation before blending.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        composite_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        alpha = cv2.GaussianBlur(
            composite_mask.astype(np.float32), (0, 0), sigmaX=1.1
        )
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        composite = image.astype(np.float32) * alpha + background.astype(
            np.float32
        ) * (1.0 - alpha)
        return np.clip(composite, 0, 255).astype(np.uint8), source, donor_rel

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        row = self.frame.iloc[index]
        image, mask, mask_found, image_path, mask_path = self._read_image_mask(index)
        weak_seed = self._sample_seed(index, "weak")
        strong_seed = self._sample_seed(index, "strong")
        background_seed = self._sample_seed(index, "background")
        mask_seed = self._sample_seed(index, "mask")

        weak_image, weak_mask = _apply_seeded(
            self.transforms["weak"], image, mask, weak_seed
        )
        strong_image, strong_mask = _apply_seeded(
            self.transforms[self.proxy_domain], image, mask, strong_seed
        )

        background_source = "none"
        donor_image_rel = ""
        if self.proxy_domain == "D3":
            strong_image, background_source, donor_image_rel = self._randomise_background(
                strong_image,
                strong_mask,
                index,
                np.random.default_rng(background_seed),
            )
        elif self.proxy_domain == "D4":
            strong_mask = _corrupt_mask(
                strong_mask, np.random.default_rng(mask_seed)
            )

        # V4 keeps its historical full-view fallback by default.  Leaf-only V5
        # explicitly requests ``keep`` so even a small non-empty SAM component
        # is cropped/neutralised and laboratory context can never re-enter.
        def focus_mask(candidate: np.ndarray, view: str) -> np.ndarray:
            area = float(np.mean(candidate > 0))
            if area >= 0.05 or self.proxy_domain == "D4":
                return candidate
            if self.small_mask_policy == "keep":
                return candidate
            if self.small_mask_policy == "error":
                raise ValueError(
                    f"SAM mask area {area:.5f} is too small for {view} view: "
                    f"{row['image_rel']}"
                )
            return np.ones_like(candidate, dtype=np.uint8)

        weak_focus_mask = focus_mask(weak_mask, "weak")
        strong_focus_mask = focus_mask(strong_mask, "strong")
        leaf_image_weak, leaf_mask_weak = crop_leaf_to_canvas(
            weak_image,
            weak_focus_mask,
            self.image_size,
            self.leaf_crop_margin,
        )
        leaf_image_strong, leaf_mask_strong = crop_leaf_to_canvas(
            strong_image,
            strong_focus_mask,
            self.image_size,
            self.leaf_crop_margin,
        )

        metadata: dict[str, Any] = {
            "dataset_index": int(index),
            "image_rel": str(row["image_rel"]),
            "mask_rel": sam_mask_relative_path(str(row["image_rel"])),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "mask_found": bool(mask_found),
            "weak_mask_area": float(np.mean(weak_mask > 0)),
            "strong_mask_area": float(np.mean(strong_mask > 0)),
            "weak_mask_fallback": bool(weak_focus_mask is not weak_mask),
            "strong_mask_fallback": bool(strong_focus_mask is not strong_mask),
            "group_id": str(row["group_id"]),
            "capture_key": str(row.get("capture_key", "")),
            "acquisition_tag": str(row.get("acquisition_tag", "UNKNOWN")),
            "class_name": str(row["plant_disease"]),
            "plant_name": str(row["plant_name"]),
            "is_healthy": bool(row["is_healthy"]),
            "disease_type_name": str(row["disease_type_name"]),
            "partition": self.partition,
            "proxy_domain": self.proxy_domain,
            "domain_index": int(self.domain_index),
            "background_source": background_source,
            "donor_image_rel": donor_image_rel,
            "weak_seed": int(weak_seed),
            "strong_seed": int(strong_seed),
        }

        return {
            "image_weak": _image_tensor(weak_image, self.normalize),
            "image_strong": _image_tensor(strong_image, self.normalize),
            "mask_weak": _mask_tensor(weak_mask),
            "mask_strong": _mask_tensor(strong_mask),
            "leaf_image_weak": _image_tensor(leaf_image_weak, self.normalize),
            "leaf_image_strong": _image_tensor(leaf_image_strong, self.normalize),
            "leaf_mask_weak": _mask_tensor(leaf_mask_weak),
            "leaf_mask_strong": _mask_tensor(leaf_mask_strong),
            "label": torch.tensor(int(row["class_index"]), dtype=torch.long),
            "plant_label": torch.tensor(int(row["plant_index"]), dtype=torch.long),
            "health_label": torch.tensor(int(row["health_index"]), dtype=torch.long),
            "disease_type_label": torch.tensor(
                int(row["disease_type_index"]), dtype=torch.long
            ),
            "domain_label": torch.tensor(self.domain_index, dtype=torch.long),
            "metadata": metadata,
        }


__all__ = [
    "DEFAULT_MASK_ROOT",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TRAIN_CSV",
    "MANIFEST_ALGORITHM_VERSION",
    "PROJECT_ROOT",
    "PROXY_DOMAINS",
    "SourceDGDataset",
    "build_source_splits",
    "canonical_capture_key",
    "canonical_disease_type",
    "crop_leaf_to_canvas",
    "grouped_stratified_split",
    "infer_acquisition_tag",
    "infer_plant_and_health",
    "load_source_dataframe",
    "resolve_sam_mask_path",
    "resolve_source_path",
    "sam_mask_relative_path",
    "seed_worker",
]
