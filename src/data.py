"""PlantVillage loading, preprocessing, augmentation, and exploratory plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from .config import (
        BATCH_SIZE,
        CLASS_NAMES_PATH,
        IMG_SIZE,
        QUICK_LIMIT,
        RESULTS_DIR,
        SEED,
        TEST_SPLIT,
        VAL_SPLIT,
        ensure_directories,
    )
except ImportError:
    from config import (
        BATCH_SIZE,
        CLASS_NAMES_PATH,
        IMG_SIZE,
        QUICK_LIMIT,
        RESULTS_DIR,
        SEED,
        TEST_SPLIT,
        VAL_SPLIT,
        ensure_directories,
    )

AUTOTUNE = tf.data.AUTOTUNE
_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.1, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED + 1),
        tf.keras.layers.RandomContrast(0.1, seed=SEED + 2),
    ],
    name="training_augmentation",
)


def _set_seed() -> None:
    """Set deterministic seeds used by TensorFlow and NumPy."""
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)


def _preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Resize an image and convert it to float32 values in [0, 1]."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply training-only augmentation to one already-normalized image."""
    image = _AUGMENTATION(image, training=True)
    return tf.clip_by_value(image, 0.0, 1.0), label


def _finalize(
    dataset: tf.data.Dataset,
    *,
    training: bool,
    batch_size: int,
) -> tf.data.Dataset:
    """Apply preprocessing, optional augmentation, batching, and prefetching."""
    dataset = dataset.map(_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.map(_augment, num_parallel_calls=AUTOTUNE)
    return dataset.cache().batch(batch_size).prefetch(AUTOTUNE)


def _split_dataset(
    dataset: tf.data.Dataset,
    size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split an unbatched dataset into deterministic train/validation/test sets."""
    shuffled = dataset.shuffle(size, seed=SEED, reshuffle_each_iteration=False)
    train_count = int(size * (1.0 - VAL_SPLIT - TEST_SPLIT))
    val_count = int(size * VAL_SPLIT)
    train = shuffled.take(train_count)
    remainder = shuffled.skip(train_count)
    validation = remainder.take(val_count)
    test = remainder.skip(val_count)
    return train, validation, test


def _load_tfds(quick: bool) -> tuple[tf.data.Dataset, int, list[str]]:
    """Load PlantVillage from TensorFlow Datasets."""
    import tensorflow_datasets as tfds

    dataset, metadata = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )
    size = int(metadata.splits["train"].num_examples)
    class_names = list(metadata.features["label"].names)
    if quick:
        size = min(size, QUICK_LIMIT)
        dataset = dataset.take(size)
    return dataset, size, class_names


def _load_folder(data_dir: Path, quick: bool) -> tuple[tf.data.Dataset, int, list[str]]:
    """Load a Kaggle-style class-folder dataset without applying augmentation."""
    class_names = sorted(
        directory.name
        for directory in data_dir.iterdir()
        if directory.is_dir() and directory.name != "Background_without_leaves"
    )
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=False,
        class_names=class_names,
    )
    size = sum(1 for _ in dataset)
    if quick:
        size = min(size, QUICK_LIMIT)
        dataset = dataset.take(size)
    return dataset.unbatch() if dataset.element_spec[0].shape.rank == 4 else dataset, size, class_names


def save_class_names(class_names: Iterable[str]) -> None:
    """Persist the class order used by labels and model outputs."""
    ensure_directories()
    CLASS_NAMES_PATH.write_text(
        json.dumps(list(class_names), indent=2),
        encoding="utf-8",
    )


def save_eda_plots(
    dataset: tf.data.Dataset,
    class_names: list[str],
    *,
    max_samples_per_class: int = 2,
) -> None:
    """Save a representative image grid and class distribution chart."""
    ensure_directories()
    samples: dict[int, list[np.ndarray]] = {index: [] for index in range(len(class_names))}
    counts = np.zeros(len(class_names), dtype=np.int64)
    for image, label in dataset:
        class_index = int(label.numpy())
        counts[class_index] += 1
        if len(samples[class_index]) < max_samples_per_class:
            samples[class_index].append(image.numpy().astype(np.uint8))

    plt.figure(figsize=(16, max(6, len(class_names) * 0.35)))
    plt.barh(class_names, counts)
    plt.xlabel("Images")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()

    populated = [(index, image) for index, values in samples.items() for image in values]
    columns = 4
    rows = max(1, (len(populated) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, rows * 3))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (class_index, image) in zip(axes_array, populated):
        axis.imshow(image)
        axis.set_title(class_names[class_index], fontsize=8)
        axis.axis("off")
    for axis in axes_array[len(populated) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "sample_grid.png", dpi=150)
    plt.close(figure)


def load_data(
    *,
    data_source: str = "tfds",
    data_dir: Path | None = None,
    quick: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load and split PlantVillage into augmented train, validation, and test datasets."""
    _set_seed()
    ensure_directories()
    if data_source == "tfds":
        raw_dataset, size, class_names = _load_tfds(quick)
    elif data_source == "folder":
        if data_dir is None:
            raise ValueError("--data-dir is required when --data-source folder is used")
        raw_dataset, size, class_names = _load_folder(data_dir, quick)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    train_raw, validation_raw, test_raw = _split_dataset(raw_dataset, size)
    save_class_names(class_names)
    if not (RESULTS_DIR / "class_distribution.png").exists():
        save_eda_plots(raw_dataset, class_names)
    return (
        _finalize(train_raw, training=True, batch_size=batch_size),
        _finalize(validation_raw, training=False, batch_size=batch_size),
        _finalize(test_raw, training=False, batch_size=batch_size),
        class_names,
    )


def main() -> None:
    """Run a data-loading smoke check and generate EDA artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-source", choices=("tfds", "folder"), default="tfds")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train, validation, test, class_names = load_data(
        data_source=args.data_source,
        data_dir=args.data_dir,
        quick=args.quick,
        batch_size=args.batch_size,
    )
    print(
        f"Loaded {len(class_names)} classes; "
        f"train={train.cardinality().numpy()}, "
        f"validation={validation.cardinality().numpy()}, "
        f"test={test.cardinality().numpy()}"
    )


if __name__ == "__main__":
    main()
