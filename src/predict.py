"""Single-image prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from .config import CLASS_NAMES_PATH, IMG_SIZE
except ImportError:
    from config import CLASS_NAMES_PATH, IMG_SIZE


def load_class_names(path: Path = CLASS_NAMES_PATH) -> list[str]:
    """Load model output labels."""
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an image to a single float32 batch in [0, 1]."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = tf.image.resize(tf.convert_to_tensor(image[..., :3]), (IMG_SIZE, IMG_SIZE))
    return (tf.cast(image, tf.float32) / 255.0).numpy()[None, ...]


def predict_top3(
    model: tf.keras.Model,
    image: Image.Image | np.ndarray,
    class_names: list[str],
) -> list[tuple[str, float]]:
    """Return the three most likely classes and probabilities."""
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:3]
    return [(class_names[int(index)], float(probabilities[index])) for index in indices]
