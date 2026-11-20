"""Grad-CAM generation and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from .config import BATCH_SIZE, MODELS_DIR, RESULTS_DIR
    from .data import load_data
except ImportError:
    from config import BATCH_SIZE, MODELS_DIR, RESULTS_DIR
    from data import load_data


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: tf.Tensor | np.ndarray,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for one preprocessed image."""
    image = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if image.shape.rank == 3:
        image = image[None, ...]
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        convolution_output, predictions = grad_model(image, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        score = predictions[:, pred_index]
    gradients = tape.gradient(score, convolution_output)
    if gradients is None:
        raise RuntimeError("Unable to compute Grad-CAM gradients")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(convolution_output * weights[:, None, None, :], axis=-1)[0]
    cam = tf.maximum(cam, 0)
    maximum = tf.reduce_max(cam)
    cam = tf.where(maximum > 0, cam / maximum, tf.zeros_like(cam))
    return cam.numpy().astype(np.float32)


def overlay_heatmap(image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a normalized heatmap on an RGB uint8 image."""
    image = np.asarray(image_uint8, dtype=np.uint8)
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    color = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0)


def main() -> None:
    """Generate a comparison grid of Grad-CAM examples for both models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "gradcam_examples.png")
    args = parser.parse_args()
    _, _, test, class_names = load_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
        quick=args.quick,
        batch_size=BATCH_SIZE,
    )
    images = np.concatenate([batch.numpy() for batch, _ in test], axis=0)
    labels = np.concatenate([label.numpy() for _, label in test], axis=0)
    models = [
        ("Baseline CNN", "baseline_cnn.keras", "last_conv"),
        ("MobileNetV2", "mobilenetv2_leafguard.keras", "out_relu"),
    ]
    loaded_models = [
        (name, tf.keras.models.load_model(MODELS_DIR / filename), layer_name)
        for name, filename, layer_name in models
        if (MODELS_DIR / filename).exists()
    ]
    if not loaded_models:
        raise FileNotFoundError("No trained model was found in the models directory")
    candidate_indices: list[int] = []
    for _, model, _ in loaded_models:
        predictions = model.predict(images, verbose=0).argmax(axis=1)
        candidate_indices.extend(np.flatnonzero(predictions != labels).tolist())
    candidate_indices.extend(range(len(images)))
    selected = list(dict.fromkeys(candidate_indices))[:8]
    figure, axes = plt.subplots(
        len(selected) * len(loaded_models),
        3,
        figsize=(12, max(8, len(selected) * len(loaded_models) * 2.4)),
        squeeze=False,
    )
    row = 0
    for model_name, model, layer_name in loaded_models:
        predictions = model.predict(images[selected], verbose=0)
        for item, image_index in enumerate(selected):
            image_uint8 = np.uint8(np.clip(images[image_index] * 255.0, 0, 255))
            prediction_index = int(predictions[item].argmax())
            heatmap = make_gradcam_heatmap(
                model,
                images[image_index : image_index + 1],
                layer_name,
                pred_index=prediction_index,
            )
            axes[row, 0].imshow(image_uint8)
            axes[row, 0].set_title(f"{model_name} | original")
            axes[row, 1].imshow(overlay_heatmap(image_uint8, heatmap))
            axes[row, 1].set_title("Grad-CAM")
            axes[row, 2].text(
                0.05,
                0.5,
                f"Predicted: {class_names[prediction_index]}\n"
                f"Confidence: {predictions[item, prediction_index]:.1%}\n"
                f"Actual: {class_names[int(labels[image_index])]}",
                va="center",
                wrap=True,
            )
            axes[row, 2].axis("off")
            for column in axes[row, :2]:
                column.axis("off")
            row += 1
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
