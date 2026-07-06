# PCBSegClassNet: PCB Component Segmentation using YOLO11-Seg + CBAM

> A research-oriented implementation of PCB component segmentation using YOLO11-Seg with a custom CBAM-enhanced architecture, optimized training pipeline, and reproducible experimentation.


## Overview

This repository contains my research work on **printed circuit board (PCB) component segmentation** using the **Ultralytics YOLO11-Seg** framework.

The project focuses on improving segmentation performance for densely packed PCB layouts containing **25 different electronic component classes**. Instead of using the default YOLO11 architecture, the model is extended with **CBAM (Convolutional Block Attention Module)** and trained using a carefully tuned pipeline including custom augmentations, multi-GPU training, and extensive experiment tracking.

The repository is intended as both:

- a reproducible training pipeline
- the foundation of an ongoing research paper in industrial computer vision.

# Project Goals

The objectives of this work are:

- Perform accurate PCB component segmentation
- Improve segmentation of small and densely packed components
- Integrate CBAM into YOLO11-Seg
- Build a reproducible training pipeline
- Perform systematic evaluation and ablation studies
- Explore architectural improvements for publication-quality research

---

# Features

- YOLO11-Seg based segmentation
- Custom CBAM integration into the backbone
- Modified Ultralytics architecture
- Multi-GPU training support
- Mixed Precision (AMP)
- AdamW optimizer
- Cosine Learning Rate Scheduler
- Automatic checkpoint saving
- Experiment logging
- Training visualization
- Validation metrics
- Publication-ready evaluation pipeline

---

# Dataset

The project uses a custom PCB segmentation dataset.

## Dataset Statistics

- 25 PCB component classes
- YOLO Segmentation format
- Polygon annotations
- High-resolution PCB images

### Classes

```text
C
R
U
J
L
Q
P
D
IC
RN
CR
RA
M
T
V
TP
FB
S
BTN
CRA
QA
LED
F
SW
JP
```

Dataset structure

```
data/
└── yolo/
    ├── train/
    │    ├── images/
    │    └── labels/
    ├── val/
    │    ├── images/
    │    └── labels/
    └── dataset.yaml
```

---

# Model Architecture

Baseline:

```
YOLO11-Seg
```

Proposed:

```
YOLO11-Seg
        │
        ▼
CBAM Enhanced Backbone
        │
        ▼
Segmentation Neck
        │
        ▼
Segmentation Head
```

The CBAM modules were integrated directly into the Ultralytics architecture rather than added externally.

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | Ultralytics YOLO11 |
| Task | Segmentation |
| Optimizer | AdamW |
| Scheduler | Cosine LR |
| Mixed Precision | AMP |
| Image Size | 640 |
| Epochs | Configurable |
| Multi GPU | Supported |
| Dataset | Custom PCB Dataset |

---

# Data Augmentation

The training pipeline includes:

- Mosaic
- MixUp
- Copy-Paste
- HSV augmentation
- Random scaling
- Rotation
- Horizontal Flip
- Vertical Flip
- Random Erasing

---

# Repository Structure

```
PCBSegClassNet/

│
├── data/
│
├── notebooks/
│
├── models/
│
├── runs/
│
├── ultralytics/
│
├── train.py
│
├── dataset.yaml
│
├── requirements.txt
│
└── README.md
```

---

# Training

Run training

```python
from ultralytics import YOLO

model = YOLO("yolo11s-seg-cbam.yaml")

model.train(
    data="dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=32
)
```

---

# Validation

```python
model.val()
```

---

# Inference

```python
model.predict(
    source="test_images",
    save=True
)
```

---

# Evaluation

The project supports evaluation using:

- Precision
- Recall
- mAP50
- mAP50-95
- IoU
- Dice Score
- Confusion Matrix
- Precision-Recall Curves
- Per-class AP

---

# Planned Research Extensions

This repository is actively being extended toward a publication-quality research paper.

Current research directions include:

- Adaptive Multi-Scale Feature Fusion
- Neck redesign for tiny PCB components
- Small-object enhancement
- PCB-specific data augmentation
- Improved segmentation losses
- Explainability using Grad-CAM
- Feature map visualization
- Comprehensive ablation studies

---

# Visualizations

The analysis pipeline includes:

- Dataset statistics
- Class distribution
- Component size distribution
- Training loss curves
- Precision-Recall curves
- Confusion Matrix
- Per-class AP
- Per-class IoU
- Per-class Dice
- Grad-CAM
- Feature Maps
- Qualitative predictions
- Failure case analysis
- FPS vs Accuracy
- GFLOPs comparison
- Ablation plots

---

# Results

Training automatically generates:

```
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
best.pt
last.pt

# Technologies Used

- Python
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- Matplotlib
- Pandas
- CUDA
- Kaggle

# Future Work

Future development will focus on:

- Adaptive feature fusion neck
- Tiny object segmentation
- Better PCB-specific augmentation
- Lightweight deployment
- Edge AI optimization
- Cross-dataset evaluation
- Industrial inspection applications
# Acknowledgements

This project builds upon the excellent work of:

- Ultralytics YOLO
- PyTorch
- OpenCV
- The open-source computer vision community

---
