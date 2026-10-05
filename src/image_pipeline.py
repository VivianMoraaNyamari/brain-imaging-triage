from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from config import TARGET_SIZE


def load_image_as_array(image_path: str | Path, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Load an image and resize to a fixed array size."""
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def generate_image_dataset(image_paths: Iterable[str | Path], target_size: tuple[int, int] = TARGET_SIZE) -> List[np.ndarray]:
    """Generate a list of normalized arrays for image paths."""
    arrays = []
    for path in image_paths:
        arrays.append(load_image_as_array(path, target_size=target_size))
    return arrays
