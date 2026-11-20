"""Evaluate images named with their exact PlantVillage class prefix."""

from __future__ import annotations

import csv
from pathlib import Path

import tensorflow as tf
from PIL import Image

try:
    from .config import CLASS_NAMES_PATH, MODELS_DIR, REAL_WORLD_DIR, RESULTS_DIR
    from .predict import load_class_names, predict_top3
except ImportError:
    from config import CLASS_NAMES_PATH, MODELS_DIR, REAL_WORLD_DIR, RESULTS_DIR
    from predict import load_class_names, predict_top3


def main() -> None:
    """Write predictions and accuracy for files in real_world_samples."""
    model_path = MODELS_DIR / "mobilenetv2_leafguard.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(CLASS_NAMES_PATH)
    rows = []
    for path in sorted(REAL_WORLD_DIR.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        prediction = predict_top3(model, Image.open(path), class_names)[0]
        expected = next((name for name in class_names if path.name.startswith(name)), "")
        rows.append(
            {
                "file": path.name,
                "expected": expected,
                "predicted": prediction[0],
                "confidence": prediction[1],
            }
        )
    accuracy = sum(row["expected"] == row["predicted"] for row in rows) / len(rows) if rows else 0.0
    with (RESULTS_DIR / "real_world.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["file", "expected", "predicted", "confidence", "accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "accuracy": accuracy})


if __name__ == "__main__":
    main()
