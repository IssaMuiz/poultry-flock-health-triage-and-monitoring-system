# 🐔 Poultry Flock Health Triage & Monitoring System

A production-inspired Computer Vision project that uses Deep Learning (CNNs) to screen poultry droppings for potential health abnormalities and integrates the prediction into a simple flock health triage workflow.


## Project Overview

Poultry farming is one of the largest livestock industries in Nigeria and many developing countries. Disease outbreaks can spread rapidly through a flock, leading to high mortality, economic losses, and food security challenges.

One of the earliest indicators of poor poultry health is a change in the appearance of bird droppings. Farmers often observe these changes before severe clinical signs appear.

This project explores how Computer Vision can assist in the early screening of poultry health by analyzing images of bird droppings.

Instead of replacing veterinarians, the system is designed as a **decision-support tool** that helps farmers document observations, assess risk, and determine when veterinary attention may be required.


## Problem Statement

Many poultry farmers rely solely on manual observation to detect disease.

Challenges include:

- Late identification of abnormal droppings.
- Limited access to veterinary professionals.
- Poor record keeping.
- Lack of structured disease monitoring.
- Delayed response to disease outbreaks.

This project investigates whether a Convolutional Neural Network (CNN) can classify poultry droppings into Healthy and Unhealthy categories and integrate the result into a simple flock health monitoring workflow.


## Project Goal

The goal of this project is to build an end-to-end Computer Vision pipeline that demonstrates industry-standard ML engineering practices.

The project will:

- Audit and understand the dataset.
- Perform Computer Vision exploratory data analysis.
- Build CNN models from scratch.
- Compare custom CNNs with transfer learning models.
- Evaluate model performance using appropriate metrics.
- Build an inference pipeline.
- Integrate the model into a simple poultry flock health triage application.


## Disclaimer

This project is intended for educational and research purposes.

The model does **not** diagnose poultry diseases.

Predictions represent image-based screening results only and should never replace veterinary examination or laboratory confirmation.


## Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- PyTorch
- TorchVision
- Albumentations
- OpenCV
- ImageHash
- Jupyter Notebook


## Project Roadmap

### Phase 0 — Project Setup

- Environment setup
- Folder structure
- Dataset acquisition


### Phase 1 — Dataset Audit

Objectives:

- Verify dataset integrity.
- Count images.
- Analyze class distribution.
- Detect corrupt images.
- Detect duplicate images.
- Generate dataset audit report.

Deliverables:

- Dataset Summary
- Audit Report
- Data Card


### Phase 2 — Exploratory Data Analysis

Objectives:

- Visualize sample images.
- Understand image characteristics.
- Analyze image dimensions.
- Explore image quality.
- Inspect class balance.

Deliverables:

- EDA Notebook
- Visualizations
- Findings Report


### Phase 3 — Image Preprocessing

Objectives:

- Resize images.
- Normalize images.
- Build augmentation pipeline.
- Create train/validation/test splits.

Deliverables:

- Data preprocessing pipeline.


### Phase 4 — CNN Development

Objectives:

Build a Convolutional Neural Network from scratch.

Topics include:

- Convolution layers
- Pooling
- Activation functions
- Fully Connected layers
- Training loop
- Validation
- Evaluation

Deliverables:

- Baseline CNN Model


### Phase 5 — Model Improvement

Objectives:

- Transfer Learning
- Hyperparameter tuning
- Learning rate scheduling
- Regularization
- Model comparison

Deliverables:

- Optimized CNN
- Model comparison report


### Phase 6 — Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Confusion Matrix
- Error Analysis


### Phase 7 — Inference Pipeline

Objectives:

- Image preprocessing
- Model loading
- Prediction pipeline
- Confidence scores

Deliverables:

- Reusable inference module


### Phase 8 — Poultry Flock Health Triage MVP

Features:

- Daily flock log
- Health incident reporting
- Image upload
- CNN screening
- Rule-based risk categorization
- Downloadable case report


### Phase 9 — Documentation

Final project documentation including:

- Data Card
- Model Card
- Architecture Diagram
- Experiment Log
- Lessons Learned
- Future Improvements

## Expected Learning Outcomes

By completing this project, I aim to gain practical experience in:

- Computer Vision
- CNN Architecture
- Deep Learning with PyTorch
- Image preprocessing
- Data augmentation
- Transfer learning
- ML Engineering
- Model evaluation
- Production-inspired project organization


## Future Improvements

Potential future enhancements include:

- Multi-class disease screening.
- Bird image analysis.
- Time-series flock health monitoring.
- Mobile application.
- Cloud deployment.
- Veterinary dashboard.
- Explainable AI (Grad-CAM).
- Multimodal learning combining images and flock records.


## Dataset

Dataset:

**Poultry Birds Poo Imagery Dataset for Health Status Prediction: A Case of South-West Nigeria**

The dataset consists of images of poultry droppings categorized into Healthy and Unhealthy classes.

The dataset is used solely for educational and research purposes.


## Current Progress

### Image Resize Strategy

**Decision**

* No resizing will be performed during preprocessing.

**Reason**

* The audit confirmed that all images already have a uniform resolution of 100×100 pixels.

* Resizing again would provide no benefit and could introduce interpolation artifacts.

Dataset Audit So Far
Task	Result
Dataset Summary	✅
Corrupt Images	✅ None Found
Image Dimensions	✅ Uniform (100×100)

Everything is looking healthy.

## Author

This project is being developed as a portfolio project to demonstrate practical skills in Computer Vision, Deep Learning, and Machine Learning Engineering.
