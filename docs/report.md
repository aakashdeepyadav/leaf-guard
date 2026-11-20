# LeafGuard: Explainable Plant Disease Detection Using Deep Learning

## Abstract

Plant diseases affect agricultural productivity and food security. LeafGuard is
an explainable deep-learning system that detects plant diseases from leaf
images. It uses a custom convolutional neural network and a MobileNetV2
transfer-learning model. The system also provides Grad-CAM explanations, FGSM
adversarial attack analysis, optional non-leaf rejection, and web deployment
through FastAPI, Next.js, Streamlit, and Docker.

The models are evaluated using accuracy, top-3 accuracy, macro F1-score,
confusion matrices, inference time, model size, and adversarial robustness.
The application returns a predicted disease, confidence score, top-3
predictions, and a visual explanation of the image regions influencing the
prediction.

**Keywords:** Plant disease detection, CNN, transfer learning, MobileNetV2,
Grad-CAM, FGSM, explainable AI, TensorFlow, FastAPI, Next.js.

## 1. Introduction

Traditional plant disease identification depends on manual inspection by
farmers or agricultural experts. This can be slow and difficult when expert
knowledge is unavailable. Computer vision provides a way to classify plant
diseases automatically from leaf images.

However, a prediction alone is not always sufficient. Users need to understand
why a model produced a particular result. LeafGuard addresses this problem by
combining disease classification with Grad-CAM visual explanations.

The objectives are to:

1. Develop a custom CNN for plant disease classification.
2. Apply transfer learning with MobileNetV2.
3. Compare both models using standard classification metrics.
4. Generate Grad-CAM visual explanations.
5. Evaluate robustness using FGSM attacks.
6. Reject images that do not resemble leaves.
7. Deploy the system through a modern web interface.

## 2. Problem Statement

The project aims to build an automated plant disease detection system that
accepts a leaf image and provides:

- Predicted disease class.
- Confidence score.
- Top-three possible disease classes.
- Grad-CAM explanation heatmap.
- Optional non-leaf rejection.
- Interactive adversarial attack demonstration.

The system should be accurate, interpretable, reproducible, and suitable for a
live demonstration.

## 3. Dataset

The primary dataset is PlantVillage, loaded through TensorFlow Datasets. It
contains healthy and diseased plant leaf images from multiple plant species and
38 classes.

Examples include:

- Apple scab
- Apple black rot
- Corn common rust
- Grape black measles
- Potato early blight
- Tomato late blight
- Tomato leaf mold
- Tomato healthy

The dataset is split deterministically using a fixed random seed of 42:

- 70% training
- 15% validation
- 15% testing

The project also supports Kaggle-style folders with one directory per class.

## 4. Data Preprocessing

Each image is:

1. Converted to RGB.
2. Resized to 160 × 160 pixels.
3. Converted to `float32`.
4. Normalized from `[0, 255]` to `[0, 1]`.
5. Batched and prefetched through `tf.data`.

Training-only augmentation includes horizontal and vertical flipping, random
rotation, random zoom, and random contrast. Augmentation is applied in the
input pipeline and is not stored inside the inference models.

## 5. System Architecture

LeafGuard contains four main layers:

### 5.1 Data Layer

Loads PlantVillage or folder-based data, preprocesses images, applies
augmentation, and creates training, validation, and test datasets.

### 5.2 Model Layer

Contains the custom CNN and MobileNetV2 transfer-learning classifier.

### 5.3 Explainability and Robustness Layer

Provides Grad-CAM heatmaps, FGSM adversarial images, and autoencoder
reconstruction-error analysis.

### 5.4 Application Layer

Provides FastAPI, Next.js, Streamlit, and Docker interfaces.

## 6. Custom CNN Model

The custom CNN contains four convolutional blocks. Each block contains a
convolution layer, batch normalization, ReLU activation, and max pooling.
The filter sizes are:

```text
32 → 64 → 128 → 128
```

The features are passed through global average pooling, dropout with rate
`0.3`, and a dense softmax classification layer. The final convolution layer
is named `last_conv` for Grad-CAM.

The model uses Adam optimization, sparse categorical cross-entropy, early
stopping, learning-rate reduction, and model checkpointing.

## 7. Transfer Learning Model

The second model uses MobileNetV2 pretrained on ImageNet:

```text
Input [0, 1]
    ↓
Rescaling to [-1, 1]
    ↓
MobileNetV2 feature extractor
    ↓
Global average pooling
    ↓
Dropout
    ↓
Dense softmax classifier
```

During stage A, the MobileNetV2 feature extractor is frozen and only the
classifier head is trained with learning rate `0.001`.

During stage B, the upper feature layers are fine-tuned with learning rate
`0.00001`. Batch-normalization layers remain frozen for stability. The final
MobileNetV2 activation layer, `out_relu`, is used for Grad-CAM.

## 8. Explainable AI with Grad-CAM

Grad-CAM explains a prediction by measuring the importance of spatial
feature-map regions:

1. Run the image through the model.
2. Extract the final convolutional activation map.
3. Calculate gradients for the predicted class.
4. Average the gradients spatially.
5. Weight the activation maps.
6. Apply ReLU and normalize the heatmap.
7. Overlay the heatmap on the original image.

Warm colors indicate regions that contributed more strongly to the prediction.
This helps determine whether the model focuses on disease symptoms or
irrelevant background regions.

## 9. FGSM Adversarial Attack

The Fast Gradient Sign Method evaluates model robustness:

```text
x_adv = clip(x + ε sign(∇x J(θ, x, y)), 0, 1)
```

Here, `x` is the original image, `y` is the target label, `J` is the loss,
and `ε` controls perturbation strength.

The project evaluates:

```text
0, 0.005, 0.01, 0.02, 0.05, 0.1
```

As epsilon increases, the perturbation becomes stronger and model accuracy is
expected to decrease. The web application also includes an interactive FGSM
demonstration.

## 10. Out-of-Distribution Rejection

An optional convolutional autoencoder identifies images that do not resemble
leaf images. The reconstruction error is:

```text
MSE = mean((x - x_reconstructed)²)
```

The rejection threshold is the 95th percentile of validation leaf errors. If
an uploaded image exceeds this threshold, the application reports that the
image does not look like a leaf and skips disease prediction.

## 11. Evaluation Metrics

The models are evaluated using:

- **Accuracy:** percentage of correct predictions.
- **Top-3 accuracy:** percentage where the correct class appears in the top
  three predictions.
- **Macro F1-score:** equal-weight average F1-score across classes.
- **Confusion matrix:** identifies frequently confused classes.
- **Inference time:** average classification time per image.
- **Model size:** size of the saved `.keras` file.

## 12. Results

Insert values from `results/model_comparison.csv`.

| Model | Accuracy | Top-3 Accuracy | Macro F1 | Inference Time/Image | Model Size |
|---|---:|---:|---:|---:|---:|
| Custom CNN | [value] | [value] | [value] | [value] ms | [value] MB |
| MobileNetV2 | [value] | [value] | [value] | [value] ms | [value] MB |

The expected result is that MobileNetV2 performs better because it uses
features learned from ImageNet. The custom CNN is simpler and easier to
interpret but may require more data and training.

Supporting files:

- `results/model_comparison.csv`
- `results/model_comparison.md`
- `results/baseline_cnn_confusion_matrix.png`
- `results/mobilenetv2_leafguard_confusion_matrix.png`
- `results/baseline_training_curves.png`
- `results/transfer_training_curves.png`

## 13. Adversarial Results

Use `results/fgsm_accuracy.csv`.

| Epsilon | Custom CNN Accuracy | MobileNetV2 Accuracy |
|---:|---:|---:|
| 0.000 | [value] | [value] |
| 0.005 | [value] | [value] |
| 0.010 | [value] | [value] |
| 0.020 | [value] | [value] |
| 0.050 | [value] | [value] |
| 0.100 | [value] | [value] |

The plot is saved as `results/fgsm_accuracy.png`, and an example image is
saved as `results/fgsm_examples.png`.

Grad-CAM comparison examples for both classifiers are saved as
`results/gradcam_examples.png`.

## 14. Web Application

The Next.js frontend allows users to select a model, upload an image, view the
prediction, inspect top-3 probabilities, view Grad-CAM, receive OOD rejection
messages, and run an FGSM attack.

The FastAPI backend provides:

```text
GET  /health
POST /predict
POST /attack
```

Streamlit provides a simpler standalone interface with the same core features.

## 15. Deployment

Streamlit deployment:

```powershell
docker build -t leafguard .
docker run --rm -p 8501:8501 leafguard
```

Frontend deployment:

```powershell
cd frontend
docker build -t leafguard-frontend .
docker run --rm -p 3000:3000 leafguard-frontend
```

The frontend uses `NEXT_PUBLIC_API_URL` to locate the FastAPI service.

## 16. Limitations

PlantVillage contains mostly plain backgrounds and may contain several images
of the same leaf. Test accuracy may therefore be higher than performance on
real field photographs. Shadows, blur, occlusion, complex backgrounds, and
unseen plant species may reduce performance.

FGSM is only one adversarial attack. The autoencoder threshold also depends on
the quality and diversity of the validation images. LeafGuard is an
educational research system and not a replacement for expert diagnosis.

## 17. Future Work

Future improvements include:

- Training with more real field photographs.
- Adding more plant species and diseases.
- Testing PGD and other adversarial attacks.
- Adding adversarial training.
- Segmenting leaves from complex backgrounds.
- Calibrating confidence scores.
- Deploying the API to Azure or another cloud platform.
- Adding authentication and prediction history.
- Supporting mobile camera capture.

## 18. Conclusion

LeafGuard demonstrates an end-to-end explainable plant disease detection
system. It combines custom deep learning, transfer learning, visual
explanations, adversarial robustness analysis, OOD rejection, and full-stack
deployment.

The project provides both research functionality and a demonstration-ready
application. Further validation with real field images is required before
using the system outside an educational or research setting.
