from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from config import DATA_DIR, IMAGE_EXTENSIONS, LABELS


def get_image_files(data_root: Path | str = DATA_DIR) -> Dict[str, List[Path]]:
    """Return image files grouped by label and split."""
    root = Path(data_root)
    grouped: Dict[str, List[Path]] = {label: [] for label in LABELS}

    for label in LABELS:
        label_dir = root / label
        if label_dir.exists():
            for file_path in sorted(label_dir.glob('*')):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    grouped[label].append(file_path)
    return grouped


def build_dataset_rows(data_root: Path | str = DATA_DIR) -> list[dict]:
    """Create rows of image paths and label assignments for all dataset folders."""
    rows = []
    root = Path(data_root)

    for split in ['train', 'val', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for label in LABELS:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for image_path in sorted(label_dir.glob('*')):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        'split': split,
                        'label': label,
                        'path': str(image_path),
                        'filename': image_path.name,
                    })

    return rows


def summarize_dataset(data_root: Path | str = DATA_DIR) -> list[dict]:
    """Summarize image counts by split and class without using pandas."""
    rows = build_dataset_rows(data_root)
    if not rows:
        return []

    summary: Dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row['split'], row['label'])
        if key not in summary:
            summary[key] = {'split': row['split'], 'label': row['label'], 'count': 0}
        summary[key]['count'] += 1

    return sorted(summary.values(), key=lambda item: (item['split'], item['label']))
