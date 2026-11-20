"""FastAPI inference service for the LeafGuard web frontend."""

from __future__ import annotations

import base64
import io
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly import is_leaf
from config import ATTACK_DEFAULT_EPSILON, CLASS_NAMES_PATH, CONFIDENCE_WARNING_THRESHOLD, MODELS_DIR
from fgsm import fgsm_attack
from gradcam import make_gradcam_heatmap, overlay_heatmap
from predict import load_class_names, preprocess_image, predict_top3

app = FastAPI(title="LeafGuard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_filename(model_name: str) -> str:
    """Map the public model choice to a safe artifact filename."""
    choices = {"mobilenet": "mobilenetv2_leafguard.keras", "baseline": "baseline_cnn.keras"}
    try:
        return choices[model_name]
    except KeyError as error:
        raise HTTPException(status_code=400, detail="model must be 'mobilenet' or 'baseline'") from error


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> tf.keras.Model:
    """Load a trained model once per process."""
    path = MODELS_DIR / _model_filename(model_name)
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Trained model is missing: {path.name}")
    return tf.keras.models.load_model(path)


def _read_image(content: bytes) -> Image.Image:
    """Decode uploaded bytes as a safe RGB image."""
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from error


def _data_url(image: np.ndarray, image_format: str = "PNG") -> str:
    """Encode an RGB image as a browser-ready data URL."""
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _prediction_response(model_name: str, image: Image.Image, confidence_threshold: float) -> dict:
    """Run OOD rejection, classification, and Grad-CAM."""
    model = _load_model(model_name)
    class_names = load_class_names(CLASS_NAMES_PATH)
    original = np.asarray(image)
    accepted, reconstruction_error = is_leaf(original)
    if not accepted:
        return {
            "accepted": False,
            "reconstruction_error": reconstruction_error,
            "message": "This image does not look like a leaf.",
        }
    top3 = predict_top3(model, image, class_names)
    layer = "out_relu" if model_name == "mobilenet" else "last_conv"
    heatmap = make_gradcam_heatmap(model, preprocess_image(image), layer)
    overlay = overlay_heatmap(original, heatmap)
    predicted, confidence = top3[0]
    return {
        "accepted": True,
        "predicted": predicted,
        "confidence": confidence,
        "low_confidence": confidence < confidence_threshold,
        "top3": [{"label": label, "probability": probability} for label, probability in top3],
        "original": _data_url(original),
        "gradcam": _data_url(overlay),
        "reconstruction_error": reconstruction_error,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = "mobilenet",
    confidence_threshold: float = Query(
        CONFIDENCE_WARNING_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence below this value is marked as low confidence.",
    ),
) -> dict:
    """Classify an uploaded image and return explanation artifacts."""
    image = _read_image(await file.read())
    return _prediction_response(model, image, confidence_threshold)


@app.post("/attack")
async def attack(
    file: UploadFile = File(...),
    model: str = "mobilenet",
    epsilon: float = Query(
        ATTACK_DEFAULT_EPSILON,
        ge=0.0,
        le=0.1,
        description="Maximum normalized FGSM perturbation.",
    ),
) -> dict:
    """Apply FGSM using the clean prediction as the target label."""
    image = _read_image(await file.read())
    model_instance = _load_model(model)
    class_names = load_class_names(CLASS_NAMES_PATH)
    clean = preprocess_image(image)
    clean_top = predict_top3(model_instance, image, class_names)[0]
    target = tf.constant([class_names.index(clean_top[0])])
    adversarial = fgsm_attack(model_instance, clean, target, epsilon).numpy()[0]
    adversarial_top = predict_top3(model_instance, adversarial, class_names)[0]
    return {
        "epsilon": epsilon,
        "clean": {"label": clean_top[0], "probability": clean_top[1]},
        "adversarial": {"label": adversarial_top[0], "probability": adversarial_top[1]},
        "image": _data_url(adversarial * 255.0),
    }
