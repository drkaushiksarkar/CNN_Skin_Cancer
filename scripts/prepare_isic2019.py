#!/usr/bin/env python3
"""
Utility to project the ISIC 2019 metadata into DermAssist's class taxonomy.

The raw download from https://www.isic-archive.com/ exposes hierarchical
diagnosis columns (diagnosis_1 … diagnosis_5).  This helper collapses those
labels into the nine classes expected by `config/default.yaml`, performs a
stratified train/validation split, and materialises class folders via
symlinks (default) or copies for compatibility.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

CLASSES: Sequence[str] = (
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented_benign_keratosis",
    "seborrheic_keratosis",
    "squamous_cell_carcinoma",
    "vascular_lesion",
)

# diagnosis_3 carries the most granular labels; fall back to diagnosis_2 for vascular lesions
DIAG3_TO_CLASS: dict[str, str] = {
    "Solar or actinic keratosis": "actinic_keratosis",
    "Basal cell carcinoma": "basal_cell_carcinoma",
    "Dermatofibroma": "dermatofibroma",
    "Melanoma, NOS": "melanoma",
    "Melanoma in situ": "melanoma",
    "Melanoma Invasive": "melanoma",
    "Atypical melanocytic neoplasm": "melanoma",
    "Nevus": "nevus",
    "Epidermal nevus": "nevus",
    "Pigmented benign keratosis": "pigmented_benign_keratosis",
    "Seborrheic keratosis": "seborrheic_keratosis",
    "Solar lentigo": "pigmented_benign_keratosis",
    "Squamous cell carcinoma, NOS": "squamous_cell_carcinoma",
}

DIAG2_TO_CLASS: dict[str, str] = {
    "Benign soft tissue proliferations - Vascular": "vascular_lesion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/isic2019/images/metadata.csv"),
        help="CSV exported from the ISIC archive.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/raw/isic2019/images"),
        help="Directory containing the ISIC_*.jpg files.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/train"),
        help="Output directory for the training split.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/val"),
        help="Output directory for the validation split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the stratified split.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy images instead of symlinking (useful on network filesystems).",
    )
    return parser.parse_args()


def assign_class(row: dict[str, str]) -> str | None:
    diag3 = (row.get("diagnosis_3") or "").strip()
    diag2 = (row.get("diagnosis_2") or "").strip()
    return DIAG3_TO_CLASS.get(diag3) or DIAG2_TO_CLASS.get(diag2)


def collect_samples(metadata_path: Path, images_dir: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = defaultdict(list)
    with metadata_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = assign_class(row)
            if not label:
                continue
            image_id = row["isic_id"].strip()
            image_path = images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                continue
            buckets[label].append(image_path)
    return buckets


def reset_dirs(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for label in CLASSES:
        (root / label).mkdir(parents=True, exist_ok=True)


def stratified_split(
    samples: dict[str, list[Path]], val_ratio: float, seed: int
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    rng = random.Random(seed)  # nosec B311 - deterministic split only
    train: dict[str, list[Path]] = {}
    val: dict[str, list[Path]] = {}
    for label in CLASSES:
        paths = samples.get(label, [])
        if not paths:
            train[label] = []
            val[label] = []
            continue
        paths = list(paths)
        rng.shuffle(paths)
        split_idx = max(1, int(len(paths) * (1 - val_ratio)))
        train[label] = paths[:split_idx]
        val[label] = paths[split_idx:] or paths[-1:]
        if not val[label]:
            val[label] = [train[label].pop()]
    return train, val


def materialise_samples(
    allocations: dict[str, list[Path]],
    target_root: Path,
    copy_files: bool,
) -> None:
    for label, paths in allocations.items():
        dest_dir = target_root / label
        for src in paths:
            dest = dest_dir / src.name
            if copy_files:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.metadata, args.images)
    missing = [label for label in CLASSES if not samples[label]]
    if missing:
        print(f"[warn] classes without samples: {', '.join(missing)}")
    print("Sample counts:")
    for label in CLASSES:
        print(f"  {label:<28} {len(samples[label])}")

    reset_dirs(args.train_dir)
    reset_dirs(args.val_dir)

    train_data, val_data = stratified_split(samples, args.val_ratio, args.seed)
    materialise_samples(train_data, args.train_dir, args.copy_files)
    materialise_samples(val_data, args.val_dir, args.copy_files)
    print(f"Train directory populated under {args.train_dir}")
    print(f"Validation directory populated under {args.val_dir}")


if __name__ == "__main__":
    main()
