# Model Explainability Analysis using Grad-CAM

## Objective

The objective of this phase was to understand how the baseline Convolutional Neural Network (CNN) arrived at its predictions.

While evaluation metrics quantify model performance, they do not reveal which image regions influence the prediction. Therefore, Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize the areas of the image that contributed most to the model's decision.

---

# Methodology

Grad-CAM was implemented using the final convolutional layer of the baseline CNN.

For each input image:

1. A forward pass was performed to obtain the predicted class.
2. Gradients of the predicted class were backpropagated through the last convolutional layer.
3. Channel-wise importance weights were computed using global average pooling of the gradients.
4. The weighted feature maps were combined to produce a class activation map.
5. The activation map was resized and overlaid on the original image for visual interpretation.

The resulting heatmaps indicate the image regions that most strongly influenced the prediction.

---

# Observations

Visual inspection of Grad-CAM overlays across multiple healthy and unhealthy samples revealed the following patterns:

### Observation 1: Attention Primarily Around the Poultry Droppings

The model consistently focused on the poultry droppings and their immediate surroundings.

In many cases, the highest activation overlapped portions of the droppings, indicating that the model learned meaningful visual features associated with the target object.

---

### Observation 2: Contextual Information Also Influences Predictions

The activation maps frequently extended beyond the droppings into the surrounding poultry litter.

This suggests that the model does not rely exclusively on the droppings themselves but also incorporates contextual information from nearby regions when making predictions.

---

### Observation 3: Coarse Localization

The generated heatmaps appeared relatively coarse and fragmented.

This is expected because the final convolutional feature maps have a spatial resolution of approximately 12 × 12 pixels before being upsampled to the original image size.

Consequently, Grad-CAM provides an approximate region of attention rather than precise pixel-level localization.

---

# Relationship with Error Analysis

Earlier qualitative error analysis revealed that the model occasionally:

* classified healthy watery droppings as unhealthy,
* classified visually firm unhealthy droppings as healthy,
* produced errors on samples with different background characteristics.

Grad-CAM provides additional insight into these observations.

The visualizations suggest that the model primarily attends to the droppings while also incorporating surrounding contextual cues.

Therefore, some prediction errors may result from the model learning associations between stool appearance, surrounding litter, and disease labels rather than relying solely on morphological characteristics of the droppings.

---

# Limitations

Several limitations should be considered when interpreting the Grad-CAM visualizations:

* Grad-CAM highlights regions of importance but does not provide causal explanations.
* The low spatial resolution of the final feature maps results in relatively coarse heatmaps.
* Background activation does not necessarily indicate incorrect model behavior, as contextual information may legitimately contribute to classification.

---

# Conclusion

Grad-CAM improved the interpretability of the baseline CNN by providing visual evidence of the image regions influencing model predictions.

The analysis indicates that the network primarily focuses on poultry droppings while also utilizing information from surrounding regions.

These findings complement the quantitative evaluation metrics and provide valuable insights into the strengths and limitations of the baseline model.
