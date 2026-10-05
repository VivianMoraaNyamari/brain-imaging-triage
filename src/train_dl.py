from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import MODELS_DIR, SEED, TARGET_SIZE


def build_transfer_model(model_name: str, num_classes: int = 4):
    """Create a transfer-learning CNN model for the triage classification task."""
    base_model_map = {
        'resnet50': ResNet50,
        'densenet121': DenseNet121,
        'efficientnetb0': EfficientNetB0,
    }

    if model_name not in base_model_map:
        raise ValueError(f'Unsupported model: {model_name}')

    base_model = base_model_map[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    )

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def save_model(model, model_name: str, output_dir: Path | str = MODELS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f'{model_name}.keras')
    return output_dir / f'{model_name}.keras'
