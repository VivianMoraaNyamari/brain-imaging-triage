from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from config import TARGET_SIZE


MODALITY_LABELS = ["CT", "MRI"]
CLASS_NAMES = ["stroke", "hemorrhage", "mass_effect", "normal"]
PRIORITY_ORDER = ["Routine", "Moderate", "Urgent", "Critical"]


@dataclass
class TriageResult:
    modality: str
    predicted_label: str
    priority: str
    confidence: float
    features: Dict[str, float]
    rationale: str
    preprocessed_path: str | None = None


def preprocess_image(image_path: str | Path, modality: str = "CT") -> np.ndarray:
    """Resize, grayscale, normalize, and center crop a single medical image."""
    img = Image.open(image_path).convert("L")
    img = img.resize(TARGET_SIZE)
    array = np.asarray(img, dtype=np.float32)

    if modality.upper() == "CT":
        array = np.clip(array, 0, 255)
    else:
        array = np.clip(array, 0, 255)

    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    return array


def extract_features(image_array: np.ndarray) -> Dict[str, float]:
    """Compute summary radiomic-like features for the triage prototype."""
    flat = image_array.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    energy = float(np.mean(flat ** 2))
    edge_strength = float(np.mean(np.abs(np.diff(flat))))

    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "min_intensity": min_val,
        "max_intensity": max_val,
        "energy": energy,
        "edge_density": edge_strength,
    }


def compute_class_scores(features: Dict[str, float], modality: str = "CT") -> Dict[str, float]:
    """Approximate evidence scores for each label using intensity and texture features."""
    mean = features["mean_intensity"]
    std = features["std_intensity"]
    energy = features["energy"]
    edge = features["edge_density"]

    stroke_score = 0.35 * mean + 0.30 * std + 0.25 * edge + 0.10 * energy
    hemorrhage_score = 0.40 * std + 0.35 * edge + 0.15 * mean + 0.10 * energy
    mass_effect_score = 0.30 * mean + 0.40 * edge + 0.20 * std + 0.10 * energy
    normal_score = 1.0 - (0.55 * std + 0.30 * edge + 0.15 * mean)

    if modality.upper() == "MRI":
        stroke_score *= 1.08
        hemorrhage_score *= 1.10
        mass_effect_score *= 1.04

    return {
        "stroke": float(stroke_score),
        "hemorrhage": float(hemorrhage_score),
        "mass_effect": float(mass_effect_score),
        "normal": float(normal_score),
    }


def triage_decision(input_image: str | Path, modality: str = "CT") -> TriageResult:
    """Run the end-to-end triage prototype for a single image."""
    processed = preprocess_image(input_image, modality)
    features = extract_features(processed)
    scores = compute_class_scores(features, modality)
    predicted_label = max(scores, key=scores.get)

    if predicted_label == "hemorrhage":
        priority = "Critical"
        confidence = min(0.98, 0.68 + (scores[predicted_label] / max(scores.values())))
        rationale = "High signal heterogeneity and edge intensity suggest acute hemorrhagic change requiring immediate review."
    elif predicted_label == "mass_effect":
        priority = "Urgent"
        confidence = min(0.96, 0.62 + (scores[predicted_label] / max(scores.values())))
        rationale = "Structural distortion and parenchymal compression are consistent with mass effect and elevated clinical concern."
    elif predicted_label == "stroke":
        priority = "Urgent"
        confidence = min(0.94, 0.60 + (scores[predicted_label] / max(scores.values())))
        rationale = "The image profile indicates ischemic or acute stroke-related change, requiring timely evaluation."
    else:
        priority = "Routine"
        confidence = min(0.90, 0.58 + (scores[predicted_label] / max(scores.values())))
        rationale = "The feature profile is closest to a normal study, though this remains a screening decision."

    return TriageResult(
        modality=modality.upper(),
        predicted_label=predicted_label,
        priority=priority,
        confidence=float(confidence),
        features=features,
        rationale=rationale,
    )


def explain_prediction(result: TriageResult) -> List[Tuple[str, float]]:
    """Return feature contributions ranked by relevance."""
    ranking = sorted(result.features.items(), key=lambda item: abs(item[1]), reverse=True)
    return ranking[:5]
