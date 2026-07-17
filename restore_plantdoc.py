"""Restore the PlantDoc image layout expected by test.csv and the V5 evaluator.

The upstream PlantDoc repository stores images below human-readable folders
such as ``Apple Scab Leaf`` and preserves its original train/test split.  The
local frozen CSV instead refers to normalized folders such as
``apple_apple_scab``.  This utility copies the exact upstream image named by
each CSV row into that normalized layout without changing the CSV or masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path, PureWindowsPath


LABEL_TO_UPSTREAM_FOLDER = {
    "Apple Apple Scab": "Apple Scab Leaf",
    "Apple Cedar Apple Rust": "Apple rust leaf",
    "Apple Healthy": "Apple leaf",
    "Blueberry Healthy": "Blueberry leaf",
    "Cherry_(including_sour) Healthy": "Cherry leaf",
    "Corn_(maize) Cercospora Leaf Spot Gray Leaf Spot": "Corn Gray leaf spot",
    "Corn_(maize) Common Rust": "Corn rust leaf",
    "Corn_(maize) Northern Leaf Blight": "Corn leaf blight",
    "Grape Black Rot": "grape leaf black rot",
    "Grape Healthy": "grape leaf",
    "Peach Healthy": "Peach leaf",
    "Pepper,_bell Bacterial Spot": "Bell_pepper leaf spot",
    "Pepper,_bell Healthy": "Bell_pepper leaf",
    "Potato Early Blight": "Potato leaf early blight",
    "Potato Late Blight": "Potato leaf late blight",
    "Raspberry Healthy": "Raspberry leaf",
    "Soybean Healthy": "Soyabean leaf",
    "Squash Powdery Mildew": "Squash Powdery mildew leaf",
    "Strawberry Healthy": "Strawberry leaf",
    "Tomato Bacterial Spot": "Tomato leaf bacterial spot",
    "Tomato Early Blight": "Tomato Early blight leaf",
    "Tomato Healthy": "Tomato leaf",
    "Tomato Late Blight": "Tomato leaf late blight",
    "Tomato Leaf Mold": "Tomato mold leaf",
    "Tomato Septoria Leaf Spot": "Tomato Septoria leaf spot",
    "Tomato Spider Mites Two-Spotted Spider Mite": (
        "Tomato two spotted spider mites leaf"
    ),
    "Tomato Tomato Mosaic Virus": "Tomato leaf mosaic virus",
    "Tomato Tomato Yellow Leaf Curl Virus": "Tomato leaf yellow virus",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Restore the PlantDoc folder layout referenced by test.csv"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=project_root / "test.csv")
    parser.add_argument(
        "--output-root", type=Path, default=project_root / "PlantDoc_restored"
    )
    parser.add_argument(
        "--mask-root", type=Path, default=project_root / "SAM2_Masks" / "Test"
    )
    return parser.parse_args()


def select_source(
    source_root: Path,
    upstream_folder: str,
    filename: str,
) -> tuple[Path, bool]:
    candidates = [
        source_root / split / upstream_folder / filename
        for split in ("train", "test")
    ]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Image is absent from both upstream splits: {upstream_folder}/{filename}"
        )
    if len(matches) == 1:
        return matches[0], False
    if sha256_file(matches[0]) != sha256_file(matches[1]):
        raise RuntimeError(
            "Same class/filename exists in upstream train and test with different "
            f"contents: {upstream_folder}/{filename}"
        )
    return matches[0], True


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    test_csv = args.test_csv.resolve()
    output_root = args.output_root.resolve()
    mask_root = args.mask_root.resolve()

    for split in ("train", "test"):
        if not (source_root / split).is_dir():
            raise NotADirectoryError(source_root / split)
    if not test_csv.is_file():
        raise FileNotFoundError(test_csv)
    if not mask_root.is_dir():
        raise NotADirectoryError(mask_root)

    with test_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image", "plant_disease"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain {sorted(required)} and at least one row")

    output_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    duplicate_sources = 0
    targets: set[Path] = set()

    for row_index, row in enumerate(rows):
        label = row["plant_disease"].strip()
        upstream_folder = LABEL_TO_UPSTREAM_FOLDER.get(label)
        if upstream_folder is None:
            raise KeyError(f"No upstream mapping for row {row_index}: {label!r}")

        csv_path = PureWindowsPath(row["image"].strip())
        normalized_folder = csv_path.parent.name
        filename = csv_path.name
        if normalized_folder in {"", ".", ".."} or filename in {"", ".", ".."}:
            raise ValueError(f"Unsafe CSV image path at row {row_index}: {row['image']!r}")

        source, was_duplicate = select_source(
            source_root, upstream_folder, filename
        )
        duplicate_sources += int(was_duplicate)
        destination = output_root / normalized_folder / filename
        if destination in targets:
            raise ValueError(f"Duplicate output target in CSV: {destination}")
        targets.add(destination)

        mask = mask_root / normalized_folder / Path(filename).with_suffix(".png").name
        if not mask.is_file():
            raise FileNotFoundError(f"Missing aligned SAM2 mask at row {row_index}: {mask}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise FileExistsError(
                    f"Refusing to overwrite a different restored image: {destination}"
                )
            reused += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    restored_files = sum(1 for path in output_root.rglob("*") if path.is_file())
    if restored_files != len(rows):
        raise RuntimeError(
            f"Restored folder contains {restored_files} files; expected {len(rows)}"
        )
    print(
        f"PlantDoc restore complete: rows={len(rows)}, copied={copied}, "
        f"reused={reused}, identical_upstream_duplicates={duplicate_sources}, "
        f"output={output_root}"
    )


if __name__ == "__main__":
    main()
