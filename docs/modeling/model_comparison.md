# Experiment 4: Model Comparison and Performance Analysis

## Objective

The objective of this experiment was to compare the performance of all developed deep learning models for poultry fecal image classification and identify the most suitable model for deployment. Three different approaches were evaluated, including a custom Convolutional Neural Network (CNN), a CNN enhanced with data augmentation, and a fine-tuned pretrained ResNet18 model. Each model was assessed using the same testing dataset and evaluation metrics to ensure a fair and consistent comparison.

---

## Experimental Summary

Three experiments were conducted throughout this study:

| Experiment   | Model                      | Objective                                                                                                    |
| ------------ | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Experiment 1 | Custom CNN                 | Establish a baseline model for poultry fecal image classification.                                           |
| Experiment 2 | CNN with Data Augmentation | Improve the baseline model by increasing data diversity and reducing overfitting through image augmentation. |
| Experiment 3 | Fine-Tuned ResNet18        | Evaluate the effectiveness of transfer learning using a pretrained deep convolutional neural network.        |

---

## Performance Comparison

| Model                   | Test Accuracy |  Precision |     Recall |   F1-Score |
| ----------------------- | ------------: | ---------: | ---------: | ---------: |
| Baseline CNN            |    **98.29%** | **98.04%** | **98.19%** | **98.12%** |
| CNN + Data Augmentation |    **98.63%** | **98.06%** | **98.94%** | **98.50%** |
| Fine-Tuned ResNet18     |    **88.03%** | **87.77%** | **85.52%** | **86.63%** |

The CNN enhanced with data augmentation achieved the highest performance across all evaluation metrics, making it the best-performing model in this study.

---

## Performance Analysis

### Baseline CNN

The baseline CNN achieved an excellent test accuracy of **98.29%**, demonstrating that a carefully designed convolutional neural network can effectively learn the visual characteristics of poultry fecal images. The model achieved high precision, recall and F1-score, indicating strong classification performance for both healthy and unhealthy classes.

### CNN with Data Augmentation

Introducing data augmentation resulted in a further improvement in model performance. By exposing the network to randomly transformed training images, the model learned more robust and generalized feature representations. This approach achieved the highest overall performance, with a test accuracy of **98.63%** and an F1-score of **98.50%**, making it the best-performing model among all evaluated approaches.

### Fine-Tuned ResNet18

Transfer learning was investigated using a pretrained ResNet18 model with Layer 4 and the final fully connected layer fine-tuned on the poultry fecal dataset. Although the model achieved reasonable performance with a test accuracy of **88.03%**, it performed substantially worse than both custom CNN models.

The lower performance suggests that pretrained ImageNet features were less suitable for the specialised poultry fecal image classification task than features learned directly from the target dataset.

---

## Discussion

The experimental results demonstrate that increasing model complexity does not necessarily lead to improved performance. Despite the strong reputation of pretrained convolutional neural networks, the fine-tuned ResNet18 did not outperform the custom CNN architecture.

Several factors may explain these findings:

* The poultry fecal dataset is highly specialised and visually different from the natural images contained in the ImageNet dataset.
* The custom CNN learned task-specific features directly from the poultry fecal images rather than adapting generic ImageNet representations.
* Data augmentation significantly improved the generalization capability of the custom CNN by exposing the model to more diverse training samples.
* Fine-tuning only the final residual block and classifier was insufficient for the pretrained ResNet18 to fully adapt to the target domain.

These observations demonstrate that transfer learning is not always the optimal solution, particularly when the target dataset differs substantially from the source dataset used for pretraining.

---

## Model Selection

Based on the experimental results, the **CNN with Data Augmentation** was selected as the final model for deployment.

The model was selected because it consistently achieved the highest performance across all evaluation metrics while maintaining excellent generalization on unseen test data.

### Selected Model

* **Model:** Custom CNN with Data Augmentation
* **Test Accuracy:** 98.63%
* **Precision:** 98.06%
* **Recall:** 98.94%
* **F1-Score:** 98.50%

This model provides the best balance between predictive performance, robustness and computational efficiency, making it the most suitable choice for practical poultry health monitoring applications.

---

## Key Findings

The following conclusions were drawn from the experiments:

* A carefully designed custom CNN can outperform a pretrained deep learning model on specialised image classification tasks.
* Data augmentation significantly improves model robustness and generalization performance.
* Transfer learning with ResNet18 did not provide a performance advantage for the poultry fecal image dataset.
* The CNN with data augmentation achieved the best overall performance and was therefore selected as the final deployment model.

---

## Conclusion

This comparative study evaluated three deep learning approaches for poultry fecal image classification. Among the evaluated models, the CNN with data augmentation achieved the best overall performance, obtaining a **98.63%** test accuracy together with the highest precision, recall and F1-score.

Although fine-tuning a pretrained ResNet18 demonstrated the feasibility of transfer learning for this task, it did not outperform the custom CNN models. These findings highlight the importance of experimentally evaluating multiple architectures rather than assuming that pretrained models will always achieve superior performance.

Consequently, the CNN with data augmentation was selected as the final model for deployment due to its superior predictive performance, excellent generalization capability and suitability for the poultry health monitoring system developed in this study.
