# Experiment 3: Fine-Tuned ResNet18 Transfer Learning

## Objective

The objective of this experiment was to investigate the effectiveness of transfer learning using a pretrained ResNet18 model for poultry fecal image classification. Unlike the previous transfer learning approach where only the classifier is trained, this experiment employed **fine-tuning** by unfreezing the final residual block (Layer 4) together with the final classification layer. This allows the model to adapt higher-level ImageNet features to the specialised poultry fecal image dataset.

---

## Model Architecture

A pretrained ResNet18 model was loaded using the ImageNet pretrained weights available in PyTorch. The original classifier, designed for 1000 ImageNet classes, was replaced with a new fully connected layer containing two output neurons corresponding to the Healthy and Unhealthy poultry fecal classes.

### Fine-Tuning Strategy

* Pretrained weights: ImageNet
* Frozen layers:

  * Conv1
  * BatchNorm1
  * Layer1
  * Layer2
  * Layer3
* Trainable layers:

  * Layer4
  * Fully Connected (FC) layer
* Number of output classes: 2

This strategy preserves the low-level and mid-level visual features learned from ImageNet while allowing the deepest feature representations to adapt to the poultry fecal image domain.

---

## Data Preprocessing

Since ResNet18 expects an input size of **224 × 224**, all images were resized accordingly.

### Training Transformations

* Resize (224 × 224)
* Random Horizontal Flip (p = 0.5)
* ToTensor()
* Normalize using ImageNet statistics

**ImageNet Mean**

* 0.485
* 0.456
* 0.406

**ImageNet Standard Deviation**

* 0.229
* 0.224
* 0.225

### Validation Transformations

* Resize (224 × 224)
* ToTensor()
* Normalize using ImageNet statistics

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Learning Rate | 0.0001                   |
| Batch Size    | 64                       |
| Epochs        | 20                       |
| Loss Function | CrossEntropyLoss         |
| Device        | NVIDIA GPU (Kaggle CUDA) |

---

## Training Results

The model converged steadily throughout the training process.

| Metric                   | Value      |
| ------------------------ | ---------- |
| Best Validation Accuracy | **90.49%** |

### Training Observations

* Validation accuracy improved steadily from **72.78%** in the first epoch to **90.49%** after 20 epochs.
* Training and validation losses decreased consistently throughout the training process.
* No evidence of unstable optimisation or severe overfitting was observed.
* Fine-tuning Layer 4 enabled the model to learn more task-specific features compared with training only the classifier; however, the overall performance remained below that of the custom CNN models.

---

## Comparison with Previous Models

| Model                   | Best Validation Accuracy |
| ----------------------- | -----------------------: |
| Baseline CNN            |                   98.36% |
| CNN + Data Augmentation |                   98.63% |
| Fine-Tuned ResNet18     |                   90.49% |

The fine-tuned ResNet18 achieved respectable performance but remained approximately **8 percentage points** below the custom CNN models.

---

## Discussion

Although fine-tuning allowed the deepest residual block to adapt to the poultry fecal dataset, the model still underperformed the custom CNN architecture.

Several factors may explain this outcome:

1. **Domain Difference**

   ResNet18 was pretrained on the ImageNet dataset, which contains natural images such as animals, vehicles, buildings and household objects. Poultry fecal images exhibit specialised colour distributions, textures and moisture patterns that differ substantially from ImageNet images.

2. **Limited Fine-Tuning**

   Only Layer 4 and the classifier were updated during training, while the earlier layers remained frozen. Although this preserves useful generic features, it may also restrict the model's ability to fully adapt to the target domain.

3. **Dataset Characteristics**

   The poultry fecal dataset is relatively specialised and visually homogeneous. A lightweight CNN designed specifically for this task may learn more discriminative features than a large pretrained architecture optimised for general-purpose image recognition.

4. **Custom CNN Suitability**

   The custom CNN was designed and trained entirely on the poultry fecal dataset, enabling all convolutional layers to learn task-specific representations from scratch. This likely contributed to its superior performance.

---

## Conclusion

The fine-tuned ResNet18 demonstrated that transfer learning can successfully classify poultry fecal images, achieving a best validation accuracy of **90.49%**. However, even after fine-tuning the final residual block, its performance remained noticeably below that of the custom CNN models.

This experiment highlights an important finding: **pretrained deep learning models do not always outperform carefully designed custom architectures on specialised datasets**. For this poultry fecal image classification task, the custom CNN proved more effective, suggesting that learning domain-specific features from scratch was more beneficial than adapting generic ImageNet representations.

The next stage of this study is to evaluate the fine-tuned ResNet18 on the independent test set, perform Grad-CAM visualisation, and compare all developed models using classification metrics, confusion matrices and explainability analysis.
