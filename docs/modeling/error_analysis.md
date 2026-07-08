# Misclassified Image Analysis

## Objective

The objective of this phase was to perform a qualitative analysis of the model's prediction errors by visually inspecting the misclassified test images.

While evaluation metrics provide an overall measure of performance, visual inspection helps identify common characteristics among incorrectly classified samples and reveals potential limitations of the model.

---

# Summary

A total of **24 misclassified images** were identified from the test dataset.

These images were manually reviewed to determine whether recurring visual patterns contributed to the incorrect predictions.

---

# Observations

## Observation 1: Water-like Appearance Strongly Influences Predictions

A recurring pattern among the misclassified samples was that poultry droppings with a watery appearance were frequently predicted as **Unhealthy**, even when their ground-truth label was **Healthy**.

This suggests that the model has learned to associate watery stool consistency with disease.

However, the presence of healthy samples exhibiting similar visual characteristics indicates that stool consistency alone is not a sufficient indicator of poultry health.

---

## Observation 2: Firm Droppings Can Also Be Unhealthy

Several misclassified samples demonstrated that some poultry droppings with a relatively firm appearance were labeled as **Unhealthy**.

This indicates that visual firmness alone does not determine the health status of a sample.

Consequently, the model occasionally predicted these samples as Healthy because they visually resembled healthy droppings.

---

## Observation 3: Background Variation

Most images in the dataset were captured on a typical poultry litter background.

However, a small number of healthy samples were photographed on a different floor surface.

Some of these images were incorrectly classified as Unhealthy.

Although this observation suggests that background differences may influence model predictions, additional analysis is required before concluding that the model relies on background information.

---

## Observation 4: Visually Ambiguous Samples

Some misclassified images appeared visually ambiguous even during manual inspection.

Healthy and unhealthy poultry droppings occasionally shared similar visual characteristics, making them difficult to distinguish based solely on appearance.

This suggests that certain prediction errors may reflect inherent ambiguity within the dataset rather than deficiencies in the model alone.

---

# Discussion

The qualitative analysis indicates that the baseline CNN relies heavily on visual appearance when distinguishing between healthy and unhealthy poultry droppings.

The model appears to assign considerable importance to stool consistency, particularly watery characteristics.

While this strategy performs well for most samples, it occasionally leads to incorrect predictions when healthy droppings exhibit similar visual features.

The analysis also raises the possibility that background variation may contribute to some prediction errors, although this hypothesis requires further investigation.

---

# Recommendations

Based on the observed failure cases, the following improvements are recommended:

* Increase the diversity of healthy samples exhibiting natural visual variation.
* Introduce additional data augmentation to improve robustness to environmental changes.
* Investigate the influence of image background on model predictions.
* Apply model explainability techniques, such as Grad-CAM, to visualize which image regions influence the CNN's decisions.
* Review ambiguous samples with domain experts where possible to verify label quality.

---

# Conclusion

The error analysis demonstrated that numerical evaluation metrics alone do not fully explain model behavior.

Manual inspection revealed that the model tends to associate watery stool appearance with disease and may also be influenced by variations in image background.

These findings provide valuable direction for future model improvements and motivate the use of explainability techniques to better understand the CNN's decision-making process.
