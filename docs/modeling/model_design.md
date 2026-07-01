# Baseline CNN Design

## Objective

Design a lightweight Convolutional Neural Network capable of classifying poultry fecal images into Healthy and Unhealthy categories.

The model serves as a baseline architecture for future experimentation.

---

## Design Principles

* Lightweight enough for CPU training.
* Easy to understand and debug.
* Modular architecture.
* Suitable for binary image classification.

---

## Network Architecture

Input

3 × 100 × 100

↓

Conv Block 1

Conv2D (3 → 32)

ReLU

MaxPool

↓

Conv Block 2

Conv2D (32 → 64)

ReLU

MaxPool

↓

Conv Block 3

Conv2D (64 → 128)

ReLU

MaxPool

↓

Flatten

↓

Linear (18,432 → 256)

↓

ReLU

↓

Dropout

↓

Linear (256 → 2)

---

## Feature Map Sizes

| Stage        | Output Shape  |
| ------------ | ------------- |
| Input        | 3 × 100 × 100 |
| Conv Block 1 | 32 × 50 × 50  |
| Conv Block 2 | 64 × 25 × 25  |
| Conv Block 3 | 128 × 12 × 12 |
| Flatten      | 18,432        |
| Hidden Layer | 256           |
| Output Layer | 2             |

---

## Why This Architecture?

The dataset contains approximately 14,600 images with relatively simple backgrounds and only two target classes.

A lightweight CNN is expected to learn discriminative texture, color, and shape features while remaining computationally efficient on CPU hardware.

This architecture also provides a strong baseline against which more advanced architectures such as ResNet or EfficientNet can be compared in future experiments.

---

## Future Improvements

Potential improvements include:

* Batch Normalization
* Additional Data Augmentation
* Learning Rate Scheduling
* Transfer Learning
* Hyperparameter Optimization
* Model Ensembling
