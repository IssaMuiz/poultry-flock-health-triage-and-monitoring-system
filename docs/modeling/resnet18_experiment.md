# Experiment 3: Frozen ResNet18 Transfer Learning

## Objective

The objective of this experiment was to evaluate the effectiveness of transfer learning using a pretrained ResNet18 model for poultry fecal image classification. Unlike the custom CNN developed in previous experiments, this model leveraged features learned from the ImageNet dataset. Only the final classification layer was trained while all pretrained convolutional layers remained frozen.

---

## Model Architecture

A pretrained ResNet18 model was loaded using the ImageNet pretrained weights provided by PyTorch. Since the original model was designed to classify 1000 image categories, the final fully connected (FC) layer was replaced with a new classifier containing two output neurons corresponding to the Healthy and Unhealthy poultry fecal classes.

### Transfer Learning Strategy

* Pretrained weights: ImageNet
* Backbone: Frozen
* Trainable layers: Final Fully Connected (FC) layer only
* Number of output classes: 2

By freezing the backbone, the model retained the generic visual features learned during ImageNet training while only adapting the classifier to the poultry fecal classification task.

---

## Data Preprocessing

Since ResNet18 expects images of size **224 × 224**, all input images were resized accordingly.

### Training Transformations

* Resize (224 × 224)
* Random Horizontal Flip (p = 0.5)
* ToTensor()
* Normalize using ImageNet statistics

Mean:

* 0.485
* 0.456
* 0.406

Standard Deviation:

* 0.229
* 0.224
* 0.225

### Validation Transformations

* Resize (224 × 224)
* ToTensor()
* Normalize using ImageNet statistics

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 0.0001           |
| Batch Size    | 64               |
| Epochs        | 20               |
| Loss Function | CrossEntropyLoss |

---

## Training Results

The model showed consistent improvement throughout training.

| Metric                   | Value      |
| ------------------------ | ---------- |
| Best Validation Accuracy | **90.49%** |

Training observations:

* Validation accuracy increased steadily from **72.78%** during the first epoch to **90.49%** after 20 epochs.
* Training and validation losses decreased consistently, indicating stable optimization.
* No signs of unstable training or divergence were observed.
* Although the model continued improving throughout training, it remained significantly below the performance achieved by the custom CNN models.

---

## Comparison with Previous Models

| Model                   | Best Validation Accuracy |
| ----------------------- | -----------------------: |
| Baseline CNN            |                   98.36% |
| CNN + Data Augmentation |                   98.63% |
| Frozen ResNet18         |                   90.49% |

The pretrained ResNet18 underperformed both custom CNN models by approximately **8 percentage points**.

---

## Discussion

The relatively lower performance of the frozen ResNet18 can be attributed to several factors.

First, only the final fully connected layer was updated during training, while all convolutional layers remained frozen. Consequently, the model relied entirely on ImageNet features that were originally learned from natural images such as animals, vehicles, and everyday objects.

Second, poultry fecal images differ substantially from ImageNet images. Characteristics such as colour distribution, texture, moisture content, and background composition are highly domain-specific and may not be adequately represented by generic ImageNet features.

Finally, restricting training to only the classifier prevented the deeper convolutional layers from adapting to these specialised visual characteristics, thereby limiting the model's ability to learn more discriminative features for poultry health classification.

---

## Conclusion

The frozen ResNet18 successfully demonstrated the applicability of transfer learning for poultry fecal image classification, achieving a respectable validation accuracy of **90.49%**. However, its performance remained considerably lower than that of the custom CNN models developed earlier in this study.

These findings indicate that freezing the entire pretrained backbone restricted the model's ability to adapt to the specialised poultry fecal dataset. Consequently, a second transfer learning experiment will be conducted using **fine-tuning**, where the final residual block (Layer 4) together with the classification layer will be unfrozen and trained. This approach is expected to allow higher-level feature representations to adapt more effectively to the target domain and improve overall classification performance.
