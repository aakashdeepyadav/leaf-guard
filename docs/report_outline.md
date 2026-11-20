# LeafGuard report outline

## Problem statement
Motivation and user workflow. Use `results/sample_grid.png` for dataset
examples.

## Dataset and preprocessing
Describe PlantVillage, the deterministic split, normalization, and
augmentation. Use `results/class_distribution.png`.

## Methods
Explain the custom CNN, MobileNetV2, Grad-CAM, FGSM, and the optional
autoencoder. Cite `src/models.py`, `src/gradcam.py`, `src/fgsm.py`, and
`src/anomaly.py`.

## Experiments and results
Use `results/model_comparison.md`, classification reports, confusion matrices,
and training curve PNGs.

## Robustness
Use `results/fgsm_accuracy.csv` and `results/fgsm_accuracy.png`.

## Real-world challenges
Use `results/real_world.csv` when field samples are available.

## Deployment
Describe Streamlit and Docker using `app/app.py` and `Dockerfile`.

## Conclusion and limitations
Discuss the PlantVillage background and repeated-leaf limitations, then
outline future field-data collection.
