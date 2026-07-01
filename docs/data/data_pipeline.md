# Data Pipeline Documentation

## Overview

This document describes the data loading pipeline used in the Poultry Fecal Disease Classification project.

The goal of the pipeline is to efficiently load image data, apply preprocessing and augmentation, and prepare batches for CNN training.

---

## Pipeline Flow

Raw Images

↓

Dataset Manifest (CSV)

↓

Custom Dataset

↓

Image Transformations

↓

DataLoader

↓

CNN Model

---

## Dataset Manifest

The dataset manifest contains metadata for every image.

### Fields

| Column     | Description              |
| ---------- | ------------------------ |
| image_path | Full path to image file  |
| class_name | Healthy or Unhealthy     |
| label      | Numerical label (0 or 1) |

### Purpose

The manifest acts as a single source of truth for the dataset and removes the need to repeatedly scan image directories during training.

---

## Dataset Splitting

The dataset was split into:

| Split      | Images |
| ---------- | ------ |
| Training   | 11,694 |
| Validation | 1,462  |
| Test       | 1,462  |

### Class Distribution

The split was performed using stratified sampling to preserve the class distribution across all subsets.

Healthy: ~54.6%

Unhealthy: ~45.4%

This ensures fair evaluation and reduces sampling bias.

---

## Custom Dataset Class

File:

src/data/dataset.py

### Responsibilities

* Read image metadata from CSV.
* Load images from disk.
* Convert images to RGB.
* Apply transformations.
* Return image-label pairs.

### Output

(image, label)

where:

* image = transformed image tensor
* label = numerical class label

---

## Image Transformations

File:

src/data/transforms.py

### Training Transformations

* Random Horizontal Flip (p=0.5)
* Random Rotation (10 degrees)
* ToTensor()
* Normalize()

### Purpose

Data augmentation helps the model generalize better by exposing it to slightly modified versions of the training images.

### Validation and Test Transformations

* ToTensor()
* Normalize()

No augmentation is applied during evaluation to ensure consistent performance measurement.

---

## DataLoader

File:

src/data/dataloader.py

### Responsibilities

* Create mini-batches.
* Shuffle training data.
* Efficiently feed images to the CNN.

### Configuration

Batch Size: 32

Shuffle:

* Training: True
* Validation: False
* Test: False

Number of Workers: 0

### Output Shape

A batch has the following shape:

torch.Size([32, 3, 100, 100])

Meaning:

* 32 = batch size
* 3 = RGB channels
* 100 = image height
* 100 = image width

Labels have shape:

torch.Size([32])

---

## Verification Results

The data pipeline was successfully tested.

### Dataset

* Images loaded correctly.
* Labels loaded correctly.

### Transformations

* Augmentation applied successfully.
* Tensor conversion successful.

### DataLoader

* Batch generation successful.
* Tensor shapes verified.
* No loading errors encountered.

---

## Conclusion

A complete and reusable PyTorch data pipeline has been implemented.

The project is now ready for CNN model development and training.
