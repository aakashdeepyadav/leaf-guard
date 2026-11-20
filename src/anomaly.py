"""Optional autoencoder-based out-of-distribution rejection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .config import AE_IMG_SIZE, MODELS_DIR
except ImportError:
    from config import AE_IMG_SIZE, MODELS_DIR


def is_leaf(
    image: np.ndarray,
    model_path: Path = MODELS_DIR / "autoencoder.keras",
    threshold_path: Path = MODELS_DIR / "ae_threshold.json",
) -> tuple[bool, float]:
    """Return whether reconstruction error is below the learned threshold."""
    if not model_path.exists() or not threshold_path.exists():
        return True, 0.0
    model = tf.keras.models.load_model(model_path)
    threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    tensor = tf.image.resize(tf.convert_to_tensor(image[..., :3]), (AE_IMG_SIZE, AE_IMG_SIZE))
    tensor = tf.cast(tensor, tf.float32) / 255.0
    reconstruction = model(tensor[None, ...], training=False)
    error = float(tf.reduce_mean(tf.square(tensor[None, ...] - reconstruction)))
    return error <= threshold, error
