"""Streamlit interface for LeafGuard."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from anomaly import is_leaf
from config import CLASS_NAMES_PATH, CONFIDENCE_WARNING_THRESHOLD, MODELS_DIR
from fgsm import fgsm_attack
from gradcam import make_gradcam_heatmap, overlay_heatmap
from predict import load_class_names, preprocess_image, predict_top3


@st.cache_resource
def load_selected_model(filename: str) -> tf.keras.Model:
    """Cache model loading across Streamlit reruns."""
    return tf.keras.models.load_model(MODELS_DIR / filename)


def pretty(name: str) -> str:
    """Make PlantVillage identifiers readable."""
    return name.replace("___", " - ").replace("_", " ")


st.set_page_config(page_title="LeafGuard", page_icon="🌿", layout="wide")
st.title("LeafGuard")
st.caption("Explainable plant disease detection with adversarial robustness.")
model_label = st.sidebar.selectbox("Model", ["MobileNetV2", "Baseline CNN"])
threshold = st.sidebar.slider(
    "Confidence warning threshold",
    0.0,
    1.0,
    CONFIDENCE_WARNING_THRESHOLD,
    0.05,
)
filename = "mobilenetv2_leafguard.keras" if model_label == "MobileNetV2" else "baseline_cnn.keras"
uploaded = st.file_uploader("Upload a leaf photo", type=("jpg", "jpeg", "png"))

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        array = np.asarray(image)
        accepted, error = is_leaf(array)
        if not accepted:
            st.error(f"This does not look like a leaf (reconstruction error {error:.4f}).")
        else:
            model = load_selected_model(filename)
            classes = load_class_names(CLASS_NAMES_PATH)
            top3 = predict_top3(model, image, classes)
            predicted, confidence = top3[0]
            st.subheader(pretty(predicted))
            st.metric("Confidence", f"{confidence:.1%}")
            if confidence < threshold:
                st.warning("The model is uncertain. Consider a clearer image or expert review.")
            left, right = st.columns(2)
            with left:
                st.image(image, caption="Original", use_container_width=True)
            with right:
                layer = "out_relu" if model_label == "MobileNetV2" else "last_conv"
                heatmap = make_gradcam_heatmap(preprocess_image(image), layer)
                st.image(overlay_heatmap(array, heatmap), caption="Grad-CAM", use_container_width=True)
            st.bar_chart({pretty(name): probability for name, probability in top3})
            st.caption("How to read the heatmap: warmer colors show regions contributing more to the prediction.")
            with st.expander("Adversarial attack demo"):
                epsilon = st.slider("Epsilon", 0.0, 0.1, 0.02, 0.005)
                clean = preprocess_image(image)
                target = np.array([classes.index(predicted)])
                adversarial = fgsm_attack(model, clean, target, epsilon).numpy()[0]
                st.image([array, np.uint8(adversarial * 255)], caption=["Original", "FGSM image"])
                st.write("Adversarial prediction:", pretty(predict_top3(model, adversarial, classes)[0][0]))
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not process this image: {error}")
