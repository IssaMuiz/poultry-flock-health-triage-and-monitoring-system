# Data Augmentation Experiment

## Objective

The objective of this experiment was to improve the generalization capability of the baseline Convolutional Neural Network (CNN) by applying data augmentation during training.

The baseline model achieved strong classification performance; however, qualitative error analysis revealed several limitations:

* Healthy watery droppings were occasionally classified as unhealthy.
* Some unhealthy droppings visually resembled healthy samples.
* Background variations appeared to influence model predictions.

These observations suggested that the model could benefit from exposure to a greater diversity of training samples.

---

# Data Augmentation Techniques

The following augmentation techniques were applied only to the training dataset.

| Augmentation              | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| Random Horizontal Flip    | Improve invariance to left-right orientation             |
| Random Rotation           | Improve robustness to slight camera angle variations     |
| Color Jitter              | Reduce sensitivity to illumination and color differences |
| Random Affine Translation | Improve robustness to object position within the image   |

Validation and test datasets were not augmented to ensure fair model evaluation.

---

# Training Procedure

The augmented model used the same CNN architecture, optimizer, learning rate, batch size, loss function, and training pipeline as the baseline model.

The only modification introduced in this experiment was the application of data augmentation during training.

---

# Quantitative Results

| Metric                   | Baseline CNN | Augmented CNN |
| ------------------------ | -----------: | ------------: |
| Best Validation Accuracy |       98.36% |        98.29% |
| Test Accuracy            |       98.29% |    **98.63%** |
| Precision                |       98.04% |    **98.06%** |
| Recall                   |       98.19% |    **98.94%** |
| F1-score                 |       98.12% |    **98.50%** |
| Misclassified Images     |           24 |        **20** |

Although the augmented model achieved a slightly lower best validation accuracy during training, it demonstrated improved performance on the independent test set.

---

# Error Analysis

The augmented model reduced the number of misclassified samples from 24 to 20.

The largest improvement was observed in the Unhealthy class, where fewer diseased samples were incorrectly classified as healthy.

This improvement is reflected by the higher recall obtained by the augmented model.

---

# Grad-CAM Comparison

Grad-CAM was used to compare the visual attention of the baseline and augmented models.

Compared with the baseline CNN, the augmented model exhibited:

* More localized activation regions.
* Stronger overlap between high-activation regions and the poultry droppings.
* Less fragmented attention across the image.
* Reduced dependence on surrounding background regions, although contextual information continued to contribute to predictions.

These observations suggest that data augmentation encouraged the network to rely more heavily on disease-related visual characteristics.

---

# Discussion

Data augmentation did not increase the highest validation accuracy obtained during training.

However, evaluation on the unseen test dataset demonstrated improvements across nearly all performance metrics.

This highlights an important principle in deep learning:

Validation accuracy alone should not be used to select the best model. Model selection should consider multiple evaluation metrics together with qualitative analyses such as error inspection and Grad-CAM visualization.

---

# Conclusion

Applying data augmentation improved the robustness and generalization capability of the baseline CNN.

Although the best validation accuracy decreased slightly, the augmented model achieved:

* Higher test accuracy,
* Higher precision,
* Higher recall,
* Higher F1-score,
* Fewer misclassified samples,
* More focused Grad-CAM visualizations.

Based on these findings, the augmented CNN is selected as the best-performing custom CNN and will serve as the baseline for subsequent experiments involving transfer learning.
