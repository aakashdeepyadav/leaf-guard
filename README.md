# LeafGuard

LeafGuard is a complete, explainable plant-disease detection project for the
INT422 Deep Learning course. It includes a custom CNN, MobileNetV2 transfer
learning, Grad-CAM explanations, FGSM adversarial attacks, optional
autoencoder-based leaf rejection, a FastAPI inference service, a Next.js
frontend, a Streamlit demo, and Docker deployment.

## What is included

- **Data:** PlantVillage through TFDS, or a Kaggle-style class-folder dataset
- **Models:** custom baseline CNN and MobileNetV2 transfer-learning classifier
- **Explainability:** Grad-CAM heatmaps for both models
- **Robustness:** FGSM attack generation and accuracy evaluation
- **OOD rejection:** convolutional autoencoder with a validation percentile threshold
- **Interfaces:** FastAPI API, Next.js web frontend, and Streamlit application
- **Reproducibility:** fixed seed (`42`) and centralized settings in `src/config.py`
- **Documentation:** report outline, Colab notebook, generated result formats

## Requirements

- Python **3.11** is recommended for the TensorFlow environment.
- Node.js 20+ and npm are required for the Next.js frontend.
- PlantVillage is downloaded automatically by TFDS on the first TFDS run.
- A trained model and `models/class_names.json` are required for inference.

## Installation

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The API dependencies are included in `requirements.txt`. They can also be
installed separately with `pip install -r api/requirements.txt`.

## Train the models

Run this complete quick pipeline first to verify the environment:

```powershell
python src/train_baseline.py --quick
python src/train_transfer.py --quick
python src/evaluate.py
python src/gradcam.py
python src/attack_eval.py --quick
python src/train_autoencoder.py --quick
```

The quick mode uses at most 2,000 images and one epoch. It is intended for
testing the pipeline, not for meaningful accuracy.

For the full GPU training pipeline:

```powershell
python src/train_baseline.py --epochs 15
python src/train_transfer.py --epochs 5
python src/train_autoencoder.py
python src/evaluate.py
python src/gradcam.py
python src/attack_eval.py
```

On Windows, if TFDS downloads PlantVillage but cannot enumerate the extracted
JPEG files, use the folder loader instead. Point `--data-dir` at the extracted
`Plant_leave_diseases_dataset_without_augmentation` directory:

```powershell
python src/train_baseline.py --quick --data-source folder --data-dir path\to\Plant_leave_diseases_dataset_without_augmentation
python src/train_transfer.py --quick --data-source folder --data-dir path\to\Plant_leave_diseases_dataset_without_augmentation
python src/train_autoencoder.py --quick --data-source folder --data-dir path\to\Plant_leave_diseases_dataset_without_augmentation
```

The folder loader excludes `Background_without_leaves` so the generated
artifacts keep the expected 38 PlantVillage classes.

The generated files are written to:

```text
models/
  baseline_cnn.keras
  mobilenetv2_leafguard.keras
  autoencoder.keras
  ae_threshold.json
  class_names.json
results/
  model_comparison.csv
  model_comparison.md
  fgsm_accuracy.csv
  gradcam_examples.png
  ood_reconstruction_errors.png
  classification reports, confusion matrices, curves, and plots
```

Generated models and results are intentionally ignored by Git. Train them
locally before starting an interface or building the Docker image.

## Using a local folder dataset

The expected structure is one directory per class:

```text
dataset/
  Apple___Apple_scab/
    image_001.jpg
  Tomato___Late_blight/
    image_002.jpg
```

Use the folder source with any training or evaluation script that accepts the
data arguments:

```powershell
python src/train_baseline.py --quick --data-source folder --data-dir path\to\dataset
python src/train_transfer.py --quick --data-source folder --data-dir path\to\dataset
python src/train_autoencoder.py --quick --data-source folder --data-dir path\to\dataset
```

## FastAPI backend and Next.js frontend

Start the backend from the project root:

```powershell
uvicorn api.main:app --reload --port 8000
```

In another terminal, install and start the frontend:

```powershell
cd frontend
npm ci
copy .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend reads the
backend URL from `NEXT_PUBLIC_API_URL`, which defaults to
`http://localhost:8000`.

The API provides:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/predict` | Prediction, top-3 probabilities, Grad-CAM, and OOD result |
| POST | `/attack` | FGSM adversarial image and clean/adversarial predictions |

Example health check:

```powershell
curl http://localhost:8000/health
```

Build the frontend for production:

```powershell
cd frontend
npm run build
npm start
```

## Streamlit demo

Streamlit remains available as a simple standalone demo:

```powershell
streamlit run app/app.py
```

It provides model selection, confidence warnings, image upload, top-3
predictions, Grad-CAM, optional OOD rejection, and an FGSM demonstration.

## Docker

The root Dockerfile packages the Streamlit application:

```powershell
docker build -t leafguard .
docker run --rm -p 8501:8501 leafguard
```

The frontend has its own production container:

```powershell
cd frontend
docker build -t leafguard-frontend .
docker run --rm -p 3000:3000 leafguard-frontend
```

For Docker inference, train or copy the required files into `models/` before
building. The frontend container expects the API at
`NEXT_PUBLIC_API_URL`; configure it for the deployment environment.

## Project layout

```text
src/                         Data, models, training, evaluation, inference
api/                         FastAPI inference service
app/                         Streamlit application
frontend/                    Next.js web frontend
models/                      Generated model artifacts
results/                     Generated report artifacts
data/real_world_samples/     Optional field photos
tests/                       TensorFlow smoke tests
notebooks/                   Colab training notebook
docs/report_outline.md       Report structure and result mapping
```

## Real-world evaluation

Place JPG or PNG files in `data/real_world_samples/`. Start each filename
with the exact PlantVillage class name, for example:

```text
Tomato___Late_blight_field_01.jpg
```

Then run:

```powershell
python src/evaluate_real_world.py
```

## Verification

Run the local checks:

```powershell
python -m pytest tests -q
python -m compileall src api app
cd frontend
npm ci
npm run lint
npm run build
npm audit --omit=dev --audit-level=high
```

The tests do not download PlantVillage and validate model construction,
preprocessing, Grad-CAM, and FGSM numerical bounds.

## Assumptions

- The TFDS PlantVillage training split is split deterministically into 70%
  train, 15% validation, and 15% test.
- Training images and inference images are float32 tensors in `[0, 1]`.
- MobileNetV2 performs its `[-1, 1]` rescaling inside the saved model.
- Data augmentation is applied only in the training input pipeline.
- OOD rejection is disabled gracefully when the autoencoder artifacts are absent.

## Limitations

PlantVillage contains mostly plain backgrounds and may contain several images
of the same leaf. Its test accuracy can therefore be inflated compared with
real field photographs. LeafGuard is an educational research demonstration,
not a replacement for expert agronomic diagnosis.
