"""Model architectures used by LeafGuard."""

from __future__ import annotations

import tensorflow as tf

try:
    from .config import DROPOUT_RATE, IMG_SIZE
except ImportError:
    from config import DROPOUT_RATE, IMG_SIZE


def build_baseline_cnn(num_classes: int) -> tf.keras.Model:
    """Build the custom convolutional baseline."""
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="image")
    x = inputs
    for block, filters in enumerate((32, 64, 128, 128)):
        layer_name = "last_conv" if block == 3 else None
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=layer_name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(num_classes: int) -> tf.keras.Model:
    """Build a flat MobileNetV2 transfer-learning classifier."""
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="image")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_rescaling")(inputs)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        input_tensor=x,
        name="mobilenetv2_base",
    )
    base.trainable = False
    features = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    features = tf.keras.layers.Dropout(DROPOUT_RATE)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_leafguard")
