"""Fast architecture and numerical smoke tests."""

import numpy as np
import tensorflow as tf

from src.fgsm import fgsm_attack
from src.gradcam import make_gradcam_heatmap
from src.models import build_baseline_cnn, build_transfer_model
from src.predict import preprocess_image


def test_models_build_and_predict():
    for builder in (build_baseline_cnn, build_transfer_model):
        model = builder(3)
        output = model(tf.random.uniform((2, 160, 160, 3)))
        assert output.shape == (2, 3)


def test_gradcam_and_fgsm():
    model = build_baseline_cnn(3)
    images = tf.random.uniform((2, 160, 160, 3))
    labels = tf.constant([0, 1])
    heatmap = make_gradcam_heatmap(model, images[:1], "last_conv")
    assert heatmap.ndim == 2
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
    attacked = fgsm_attack(model, images, labels, 0.05)
    assert float(tf.reduce_min(attacked)) >= 0.0
    assert float(tf.reduce_max(attacked)) <= 1.0
    assert float(tf.reduce_max(tf.abs(attacked - images))) <= 0.05 + 1e-6


def test_preprocess_shape_and_range():
    image = np.full((220, 180, 3), 127, dtype=np.uint8)
    processed = preprocess_image(image)
    assert processed.shape == (1, 160, 160, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min()) <= float(processed.max()) <= 1.0
