"""Train the MobileNetV2 transfer-learning model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from .config import (
        BATCH_SIZE,
        RESULTS_DIR,
        MODELS_DIR,
        TRANSFER_FINE_TUNE_EPOCHS,
        TRANSFER_HEAD_EPOCHS,
        ensure_directories,
    )
    from .data import load_data
    from .models import build_transfer_model
except ImportError:
    from config import (
        BATCH_SIZE,
        MODELS_DIR,
        RESULTS_DIR,
        TRANSFER_FINE_TUNE_EPOCHS,
        TRANSFER_HEAD_EPOCHS,
        ensure_directories,
    )
    from data import load_data
    from models import build_transfer_model


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3_accuracy")],
    )


def main() -> None:
    """Train the frozen head, then fine-tune MobileNetV2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=TRANSFER_HEAD_EPOCHS)
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
    model = build_transfer_model(len(class_names))
    _compile(model, 1e-3)
    head_epochs = 1 if args.quick else args.epochs
    first = model.fit(train, validation_data=validation, epochs=head_epochs)
    feature_layer_index = next(
        index for index, layer in enumerate(model.layers) if layer.name == "out_relu"
    )
    fine_tune_layers = model.layers[max(0, feature_layer_index - 30) : feature_layer_index]
    for layer in fine_tune_layers:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    _compile(model, 1e-5)
    fine_tune_epochs = 1 if args.quick else TRANSFER_FINE_TUNE_EPOCHS
    second = model.fit(train, validation_data=validation, epochs=fine_tune_epochs)
    model.save(MODELS_DIR / "mobilenetv2_leafguard.keras")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    accuracy = first.history["accuracy"] + second.history["accuracy"]
    val_accuracy = first.history["val_accuracy"] + second.history["val_accuracy"]
    loss = first.history["loss"] + second.history["loss"]
    val_loss = first.history["val_loss"] + second.history["val_loss"]
    axes[0].plot(accuracy, label="train")
    axes[0].plot(val_accuracy, label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "transfer_training_curves.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
