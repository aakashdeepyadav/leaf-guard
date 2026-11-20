"""Evaluate saved models and write report-ready metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from .config import BATCH_SIZE, MODELS_DIR, RESULTS_DIR, ensure_directories
    from .data import load_data
except ImportError:
    from config import BATCH_SIZE, MODELS_DIR, RESULTS_DIR, ensure_directories
    from data import load_data


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, labels: np.ndarray) -> dict[str, float]:
    """Compute classification and latency metrics."""
    images = np.concatenate([batch.numpy() for batch, _ in dataset], axis=0)
    start = time.perf_counter()
    probabilities = model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "top3_accuracy": float(np.mean([label in np.argsort(row)[-3:] for row, label in zip(probabilities, labels)])),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "inference_ms_per_image": elapsed * 1000 / len(images),
        "model_size_mb": 0.0,
    }


def main() -> None:
    """Evaluate baseline and transfer models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    ensure_directories()
    _, _, test, class_names = load_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
        batch_size=BATCH_SIZE,
    )
    labels = np.concatenate([label.numpy() for _, label in test], axis=0)
    rows = []
    confused_pairs: list[tuple[int, int, int]] = []
    for name, filename in (("Baseline CNN", "baseline_cnn.keras"), ("MobileNetV2", "mobilenetv2_leafguard.keras")):
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        model = tf.keras.models.load_model(path)
        metrics = evaluate_model(model, test, labels)
        metrics["model"] = name
        metrics["model_size_mb"] = path.stat().st_size / 1_000_000
        rows.append(metrics)
        probabilities = model.predict(
            np.concatenate([batch.numpy() for batch, _ in test], axis=0), verbose=0
        )
        predictions = probabilities.argmax(axis=1)
        report = pd.DataFrame(classification_report(
            labels, predictions, target_names=class_names, output_dict=True, zero_division=0
        )).T
        report.to_csv(RESULTS_DIR / f"{filename.removesuffix('.keras')}_classification_report.csv")
        matrix = confusion_matrix(labels, predictions, normalize="true")
        raw_matrix = confusion_matrix(labels, predictions)
        for actual in range(raw_matrix.shape[0]):
            for predicted in range(raw_matrix.shape[1]):
                if actual != predicted and raw_matrix[actual, predicted] > 0:
                    confused_pairs.append((int(raw_matrix[actual, predicted]), actual, predicted))
        plt.figure(figsize=(16, 14))
        plt.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"{name} normalized confusion matrix")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{filename.removesuffix('.keras')}_confusion_matrix.png", dpi=150)
        plt.close()
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    top_pairs = sorted(confused_pairs, reverse=True)[:10]
    pair_lines = "\n".join(
        f"- {count}: `{class_names[actual]}` predicted as `{class_names[predicted]}`"
        for count, actual, predicted in top_pairs
    )
    report = f"{comparison.to_markdown(index=False)}\n\n### Most confused class pairs\n\n{pair_lines}\n"
    (RESULTS_DIR / "model_comparison.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
