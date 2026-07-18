# Model Explainability Analysis using Grad-CAM

## Objective

The objective of this phase was to investigate how the trained deep learning models arrived at their predictions by identifying the image regions that contributed most to the classification decision.

While evaluation metrics such as accuracy, precision, recall, and F1-score quantify predictive performance, they do not explain the reasoning behind the model's decisions. To improve model interpretability, Gradient-weighted Class Activation Mapping (Grad-CAM) was employed to visualize the discriminative regions that influenced each prediction.

Grad-CAM analysis was performed for three trained models:

- Baseline CNN
- CNN with Data Augmentation
- Fine-Tuned ResNet18

This enabled a qualitative comparison of how different training strategies and model architectures learned visual representations of poultry droppings.

---

# Methodology

Grad-CAM was generated using the final convolutional layer of each model.

For every input image:

1. A forward pass was performed to obtain the predicted class.
2. Gradients of the predicted class were backpropagated through the selected convolutional layer.
3. Channel-wise importance weights were computed by applying global average pooling to the gradients.
4. The weighted feature maps were combined to generate a class activation map.
5. The activation map was resized to match the original image dimensions.
6. The heatmap was overlaid on the original image to highlight the regions that most strongly influenced the prediction.

For the custom CNN models (Baseline CNN and CNN with Data Augmentation), the final convolutional layer was used as the target layer. For the fine-tuned ResNet18 model, Grad-CAM was generated from the final residual block (`layer4[-1]`), which represents the deepest convolutional features before global average pooling.

---

# Observations

Visual inspection of the Grad-CAM overlays across multiple healthy and unhealthy samples revealed several consistent patterns.

## Observation 1: Primary Attention on Poultry Droppings

Across all three models, the highest activation consistently occurred around the poultry droppings, indicating that each model successfully learned to focus on the target object rather than irrelevant regions of the image.

---

## Observation 2: Influence of Surrounding Context

The activation maps frequently extended beyond the droppings into the surrounding poultry litter.

This suggests that the models utilized contextual information in addition to the appearance of the droppings when making predictions. Such behaviour is expected, as environmental characteristics may provide complementary information related to poultry health.

---

## Observation 3: Differences Between Models

Although all three models focused on the droppings, notable differences were observed in their attention patterns.

- **Baseline CNN** produced relatively coarse activation maps, with attention distributed across the droppings and nearby background regions.

- **CNN with Data Augmentation** demonstrated more consistent and stable attention, indicating improved robustness to variations in image appearance introduced during training.

- **Fine-Tuned ResNet18** generated the most concentrated activation maps, with strong localization directly over the poultry droppings and minimal activation in surrounding regions.

Despite producing the sharpest visual attention maps, the fine-tuned ResNet18 achieved lower quantitative performance than the custom CNN models. This demonstrates that stronger localization alone does not necessarily result in superior classification performance.

---

## Observation 4: Spatial Resolution

The Grad-CAM visualizations produced by the custom CNN models appeared relatively coarse due to the low spatial resolution of the final convolutional feature maps prior to upsampling.

Conversely, the ResNet18 visualizations exhibited finer localization, reflecting the richer hierarchical feature representations learned by the deeper pretrained architecture.

---

# Relationship with Error Analysis

Earlier qualitative error analysis revealed that the models occasionally:

- classified healthy watery droppings as unhealthy,
- classified visually firm unhealthy droppings as healthy,
- produced errors on samples with varying background characteristics.

Grad-CAM provides additional insight into these observations.

The visualizations indicate that the models primarily attend to the poultry droppings while also incorporating surrounding contextual information. Consequently, some prediction errors may arise from learned associations between stool appearance, environmental context, and disease labels rather than relying exclusively on the intrinsic characteristics of the droppings.

---

# Limitations

Although Grad-CAM provides valuable insight into model behaviour, several limitations should be considered.

- Grad-CAM identifies influential image regions but does not establish causal relationships.
- The localization quality depends on the spatial resolution of the selected convolutional feature maps.
- Background activation does not necessarily indicate incorrect model behaviour, as contextual information may legitimately contribute to disease classification.
- Visual explanations should always be interpreted alongside quantitative evaluation metrics rather than in isolation.

---

# Conclusion

Grad-CAM significantly improved the interpretability of the developed poultry disease classification models by revealing the image regions responsible for their predictions.

Across all experiments, the models consistently focused on the poultry droppings while also incorporating surrounding contextual information. The CNN trained with data augmentation demonstrated improved robustness over the baseline model, whereas the fine-tuned ResNet18 produced the most concentrated visual attention despite achieving lower classification performance.

Overall, the explainability analysis complements the quantitative evaluation results and provides greater confidence that the models learned meaningful visual representations relevant to poultry health assessment.