# Dataset Audit Report

## Overview

The purpose of the dataset audit is to understand the quality, structure, and characteristics of the poultry fecal image dataset before building any machine learning model.

A proper dataset audit helps identify issues that could negatively affect model performance, such as corrupted images, inconsistent image sizes, missing labels, or severe class imbalance.

---

## Dataset Summary

| Property          |  Value |
| ----------------- | -----: |
| Number of Classes |      2 |
| Total Images      | 14,618 |
| Healthy Images    |  7,991 |
| Unhealthy Images  |  6,627 |

---

## Image Integrity Check

All image files were successfully opened.

### Result

* Corrupted Images: 0

This indicates that every image can be safely used for training and evaluation.

---

## Image Dimension Analysis

| Property       | Value |
| -------------- | ----: |
| Minimum Width  |   100 |
| Maximum Width  |   100 |
| Minimum Height |   100 |
| Maximum Height |   100 |

### Observation

Every image has the same resolution (100 × 100 pixels).

This removes the need for resizing during preprocessing.

---

## Image Format Analysis

### Image Format

JPEG

### Color Mode

RGB

All images are stored consistently in RGB format.

---

## Average RGB Values

| Channel |   Mean |
| ------- | -----: |
| Red     | 131.86 |
| Green   | 118.61 |
| Blue    | 101.14 |

These values provide insight into the overall color distribution of the dataset and may be useful when selecting normalization statistics in future experiments.

---

## Class Distribution

| Class     | Images |
| --------- | -----: |
| Healthy   |  7,991 |
| Unhealthy |  6,627 |

The dataset is relatively balanced, reducing the likelihood of severe class imbalance during training.

---

## Conclusion

The dataset passed all quality checks.

No corrupted images were found, image dimensions are consistent, color formats are uniform, and the class distribution is suitable for building a baseline CNN model.
