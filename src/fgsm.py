"""Fast Gradient Sign Method adversarial attack."""

from __future__ import annotations

import tensorflow as tf


def fgsm_attack(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    epsilon: float,
) -> tf.Tensor:
    """Perturb normalized images in the direction that increases loss."""
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int64)
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    gradient = tape.gradient(loss, images)
    if gradient is None:
        raise RuntimeError("Unable to compute image gradients for FGSM")
    return tf.clip_by_value(images + epsilon * tf.sign(gradient), 0.0, 1.0)
