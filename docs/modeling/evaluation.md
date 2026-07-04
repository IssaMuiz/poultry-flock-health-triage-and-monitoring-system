# Model Evaluation

## Objective

The objective of the evaluation phase is to assess the performance of the trained baseline Convolutional Neural Network (CNN) on a previously unseen test dataset.

Unlike the training and validation datasets, the test dataset is used only after model development is complete. This provides an unbiased estimate of the model's ability to generalize to new data.

---

# Evaluation Workflow

The evaluation pipeline follows the steps below:

```
Load Trained Model
        ↓
Load Test Dataset
        ↓
Run Model Inference
        ↓
Collect Predictions
        ↓
Compute Evaluation Metrics
        ↓
Generate Classification Report
        ↓
Generate Confusion Matrix
        ↓
Analyze Model Performance
```

---

# Evaluation Metrics

The following evaluation metrics are computed.

## Accuracy

Accuracy measures the overall proportion of correctly classified samples.

Formula:

Accuracy = (Correct Predictions) / (Total Predictions)

Result:

* **98.29%**

Interpretation:

The trained CNN correctly classified approximately 98 out of every 100 poultry dropping images in the test dataset.

---

## Precision

Precision measures how reliable the model's positive (Unhealthy) predictions are.

Formula:

Precision = TP / (TP + FP)

Result:

* **98.04%**

Interpretation:

When the model predicts that a poultry dropping is unhealthy, it is correct approximately 98% of the time.

---

## Recall

Recall measures the model's ability to identify unhealthy poultry droppings.

Formula:

Recall = TP / (TP + FN)

Result:

* **98.19%**

Interpretation:

The model successfully detected approximately 98% of all unhealthy samples present in the test dataset.

Since this project focuses on poultry health screening, recall is an especially important metric because missing diseased samples may delay intervention.

---

## F1-Score

The F1-score combines precision and recall into a single performance metric.

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Result:

* **98.12%**

Interpretation:

The high F1-score indicates that the model maintains an excellent balance between detecting unhealthy samples and minimizing false alarms.

---

# Classification Report

The classification report provides performance metrics for each class independently.

| Class     | Precision | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| Healthy   |      0.98 |   0.98 |     0.98 |
| Unhealthy |      0.98 |   0.98 |     0.98 |

### Interpretation

The model performs consistently across both Healthy and Unhealthy classes.

No significant class imbalance or prediction bias is observed.

---

# Confusion Matrix

The confusion matrix obtained on the test dataset is shown below.

| Actual \ Predicted | Healthy | Unhealthy |
| ------------------ | ------: | --------: |
| Healthy            | **786** |    **13** |
| Unhealthy          |  **12** |   **651** |

Where:

* **786** Healthy samples were correctly classified.
* **651** Unhealthy samples were correctly classified.
* **13** Healthy samples were incorrectly classified as Unhealthy (False Positives).
* **12** Unhealthy samples were incorrectly classified as Healthy (False Negatives).

---

# Error Analysis

Total test images:

* **1462**

Correct predictions:

* **1437**

Incorrect predictions:

* **25**

Overall error rate:

* **1.71%**

The low number of classification errors demonstrates that the baseline CNN learned highly discriminative visual features for distinguishing between healthy and unhealthy poultry droppings.

---

# Discussion

The evaluation results indicate that the baseline CNN generalizes well to unseen test data.

Key observations include:

* High overall classification accuracy.
* Balanced precision and recall.
* Nearly identical performance across both classes.
* Very low number of false positives and false negatives.
* No evidence of severe class prediction bias.

These findings suggest that the baseline model provides a strong foundation for poultry dropping health classification.

---

# Current Project Status

Completed:

* Problem Definition
* Dataset Audit
* Dataset Splitting
* Dataset Class
* Data Augmentation
* DataLoader
* Baseline CNN
* Model Training
* Model Checkpointing
* Model Evaluation

Upcoming:

* Learning Curve Visualization
* Misclassified Image Analysis
* Model Error Analysis
* Model Improvement
* Inference Pipeline
* Deployment Preparation
