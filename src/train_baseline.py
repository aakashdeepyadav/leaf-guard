"""Train the custom LeafGuard CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from .config import BASELINE_EPOCHS, BATCH_SIZE, MODELS_DIR, RESULTS_DIR, ensure_directories
    from .data import load_data
    from .models import build_baseline_cnn
except ImportError:
    from config import BASELINE_EPOCHS, BATCH_SIZE, MODELS_DIR, RESULTS_DIR, ensure_directories
    from data import load_data
    from models import build_baseline_cnn


def main() -> None:
    """Train and save the baseline model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=BASELINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    ensure_directories()
    train, validation, _, class_names = load_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
        quick=args.quick,
        batch_size=args.batch_size,
    )
    model = build_baseline_cnn(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3_accuracy")],
    )
    epochs = 1 if args.quick else args.epochs
    history = model.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2),
            tf.keras.callbacks.ModelCheckpoint(MODELS_DIR / "baseline_cnn.keras", save_best_only=True),
        ],
    )
    model.save(MODELS_DIR / "baseline_cnn.keras")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "baseline_training_curves.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
