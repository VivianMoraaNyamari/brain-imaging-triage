from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import DATA_DIR, IMAGE_EXTENSIONS, LABELS, TARGET_SIZE


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "energy",
    "edge_density",
    "mean_gradient",
    "contrast",
]


def _image_to_array(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def _compute_features(image_path: str | Path) -> dict[str, float]:
    arr = _image_to_array(image_path)
    flat = arr.ravel()
    diff = np.diff(flat) if flat.size > 1 else np.array([0.0])
    grad = np.abs(diff)

    return {
        "mean_intensity": float(np.mean(flat)),
        "std_intensity": float(np.std(flat)),
        "min_intensity": float(np.min(flat)),
        "max_intensity": float(np.max(flat)),
        "energy": float(np.mean(flat ** 2)),
        "edge_density": float(np.mean(grad)) if grad.size else 0.0,
        "mean_gradient": float(np.mean(np.abs(np.diff(arr)))) if arr.size > 1 else 0.0,
        "contrast": float(np.var(flat)),
    }


def build_dataset_rows(data_root: Path | str = DATA_DIR) -> list[dict[str, Any]]:
    """Build a dataset list from train/val/test folders and label directories."""
    root = Path(data_root)
    rows: list[dict[str, Any]] = []

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for label in LABELS:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for file_path in sorted(label_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    features = _compute_features(file_path)
                    row = {
                        "split": split,
                        "label": label,
                        "filename": file_path.name,
                        "path": str(file_path),
                        "label_index": LABELS.index(label),
                        **features,
                    }
                    rows.append(row)

    return rows


def build_dataset_dataframe(data_root: Path | str = DATA_DIR) -> list[dict[str, Any]]:
    return build_dataset_rows(data_root)


def get_split_data(data_root: Path | str = DATA_DIR, split: str = "train") -> tuple[list[list[float]], list[int]]:
    rows = build_dataset_rows(data_root)
    filtered = [row for row in rows if row["split"] == split]
    X = [[row[column] for column in FEATURE_COLUMNS] for row in filtered]
    y = [int(row["label_index"]) for row in filtered]
    return X, y
