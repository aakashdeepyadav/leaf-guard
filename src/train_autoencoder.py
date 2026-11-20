"""Train the optional convolutional leaf autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from .config import AE_BATCH_SIZE, AE_EPOCHS, AE_IMG_SIZE, MODELS_DIR, RESULTS_DIR
    from .data import load_data
except ImportError:
    from config import AE_BATCH_SIZE, AE_EPOCHS, AE_IMG_SIZE, MODELS_DIR, RESULTS_DIR
    from data import load_data


def main() -> None:
    """Train the autoencoder and derive a validation rejection threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    _, validation, _, _ = load_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
        quick=args.quick,
    )
    images = np.concatenate([x.numpy() for x, _ in validation], axis=0)
    images = tf.image.resize(images, (AE_IMG_SIZE, AE_IMG_SIZE))
    inputs = tf.keras.Input((AE_IMG_SIZE, AE_IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        images,
        images,
        epochs=1 if args.quick else AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_split=0.1,
    )
    model.save(MODELS_DIR / "autoencoder.keras")
    errors = tf.reduce_mean(tf.square(images - model(images, training=False)), axis=(1, 2, 3)).numpy()
    threshold = float(np.percentile(errors, 95))
    (MODELS_DIR / "ae_threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")
    (_, _), (cifar_images, _) = tf.keras.datasets.cifar10.load_data()
    if args.quick:
        cifar_images = cifar_images[: min(128, len(cifar_images))]
    cifar_images = tf.image.resize(cifar_images, (AE_IMG_SIZE, AE_IMG_SIZE))
    cifar_images = tf.cast(cifar_images, tf.float32) / 255.0
    negative_errors = tf.reduce_mean(
        tf.square(cifar_images - model(cifar_images, training=False)), axis=(1, 2, 3)
    ).numpy()
    leaf_accepted = float(np.mean(errors <= threshold))
    non_leaf_rejected = float(np.mean(negative_errors > threshold))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(errors, bins=30, alpha=0.7, label="PlantVillage leaves")
    axis.hist(negative_errors, bins=30, alpha=0.7, label="CIFAR-10 non-leaves")
    axis.axvline(threshold, color="black", linestyle="--", label="95th percentile threshold")
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Images")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "ood_reconstruction_errors.png", dpi=150)
    plt.close(figure)
    (RESULTS_DIR / "ood_results.md").write_text(
        "\n".join(
            [
                "# OOD results",
                "",
                f"- Validation leaf acceptance threshold: `{threshold:.6f}`",
                f"- Leaf acceptance fraction: `{leaf_accepted:.2%}`",
                f"- CIFAR-10 non-leaf rejection fraction: `{non_leaf_rejected:.2%}`",
                "- Histogram: `ood_reconstruction_errors.png`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
