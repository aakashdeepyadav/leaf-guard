"""Evaluate FGSM robustness for the saved classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .config import ATTACK_EPSILONS, ATTACK_LIMIT, BATCH_SIZE, MODELS_DIR, RESULTS_DIR
    from .data import load_data
    from .fgsm import fgsm_attack
except ImportError:
    from config import ATTACK_EPSILONS, ATTACK_LIMIT, BATCH_SIZE, MODELS_DIR, RESULTS_DIR
    from data import load_data
    from fgsm import fgsm_attack


def main() -> None:
    """Write accuracy-versus-epsilon results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    _, _, test, _ = load_data(data_source=args.data_source, data_dir=args.data_dir, quick=args.quick)
    images = np.concatenate([x.numpy() for x, _ in test], axis=0)
    labels = np.concatenate([y.numpy() for _, y in test], axis=0)
    limit = min(len(images), ATTACK_LIMIT if not args.quick else 128)
    images, labels = images[:limit], labels[:limit]
    rows = []
    example_saved = False
    for model_name, filename in (("Baseline CNN", "baseline_cnn.keras"), ("MobileNetV2", "mobilenetv2_leafguard.keras")):
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        model = tf.keras.models.load_model(path)
        for epsilon in ATTACK_EPSILONS:
            adversarial = images if epsilon == 0 else fgsm_attack(model, images, labels, epsilon).numpy()
            accuracy = np.mean(model.predict(adversarial, verbose=0).argmax(axis=1) == labels)
            rows.append({"model": model_name, "epsilon": epsilon, "accuracy": float(accuracy)})
            if not example_saved and epsilon == 0.05:
                clean = images[0]
                adversarial_image = fgsm_attack(
                    model, images[:1], labels[:1], epsilon
                ).numpy()[0]
                perturbation = np.clip(
                    (adversarial_image - clean) * 5.0 + 0.5, 0.0, 1.0
                )
                clean_label = int(model.predict(images[:1], verbose=0).argmax(axis=1)[0])
                adversarial_label = int(
                    model.predict(adversarial_image[None, ...], verbose=0).argmax(axis=1)[0]
                )
                figure, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(clean)
                axes[0].set_title(f"Clean (class {clean_label})")
                axes[1].imshow(perturbation)
                axes[1].set_title("Amplified perturbation")
                axes[2].imshow(adversarial_image)
                axes[2].set_title(f"FGSM (class {adversarial_label})")
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout()
                figure.savefig(RESULTS_DIR / "fgsm_examples.png", dpi=150)
                plt.close(figure)
                example_saved = True
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "fgsm_accuracy.csv", index=False)
    if not frame.empty:
        for model_name, group in frame.groupby("model"):
            plt.plot(group["epsilon"], group["accuracy"], marker="o", label=model_name)
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fgsm_accuracy.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
