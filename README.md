# 🐔 Poultry Flock Health Triage and Monitoring System

## Overview

Poultry diseases pose a significant challenge to farmers, often resulting in reduced productivity, increased mortality, and substantial economic losses. Early identification of unhealthy birds is therefore essential for timely intervention and effective flock management.

This project presents a deep learning-based poultry health triage and monitoring system that automatically classifies poultry droppings as **Healthy** or **Unhealthy** using computer vision techniques. The system was developed using PyTorch and follows a complete machine learning workflow, including exploratory data analysis, data preprocessing, model development, evaluation, and explainability analysis.

Three different deep learning approaches were investigated:

- A custom Baseline Convolutional Neural Network (CNN)
- A CNN enhanced with data augmentation
- A Fine-Tuned ResNet18 model using transfer learning

Comprehensive experiments demonstrated that the custom CNN models outperformed the transfer learning approach on this dataset, achieving over **98% classification accuracy** while maintaining strong precision, recall, and F1-score. Model predictions were further interpreted using Gradient-weighted Class Activation Mapping (Grad-CAM), providing visual explanations of the image regions influencing each prediction.

The project demonstrates an end-to-end workflow for developing an interpretable computer vision solution for poultry disease screening and serves as a foundation for future intelligent livestock monitoring systems.

---

## Highlights

- End-to-end computer vision pipeline built with PyTorch.
- Explored three modelling approaches:
  - Custom CNN
  - CNN with Data Augmentation
  - Fine-Tuned ResNet18
- Achieved **98.63% test accuracy** using a custom CNN with data augmentation.
- Implemented Grad-CAM to interpret model predictions.
- Modular project structure supporting reproducible experiments.
- Comprehensive documentation covering EDA, preprocessing, model development, evaluation, and explainability.

## Dataset

The project utilizes the **Poultry Birds Poo Imagery Dataset for Health Status Prediction: A Case of South-West Nigeria**, a publicly available dataset developed for poultry disease classification based on fecal imagery.

### Dataset Summary

| Property | Value |
|----------|------:|
| Total Images | 14,618 |
| Healthy Samples | 7,991 |
| Unhealthy Samples | 6,627 |
| Number of Classes | 2 |
| Image Format | JPG |
| Original Image Size | 100 × 100 pixels |
| Color Space | RGB |

### Class Distribution

| Class | Label |
|-------|------:|
| Healthy | 0 |
| Unhealthy | 1 |

The dataset is relatively balanced, reducing the likelihood of severe class imbalance during training.

### Train / Validation / Test Split

The dataset was divided using stratified sampling to preserve the original class distribution.

| Dataset | Samples |
|---------|--------:|
| Training | 11,694 |
| Validation | 1,462 |
| Test | 1,462 |

The same data split was used across all experiments to ensure a fair comparison between different model architectures.

### Data Preparation

Prior to model training, the following preprocessing steps were performed:

- Dataset manifest generation
- Duplicate inspection
- Missing value verification
- Stratified train-validation-test splitting
- Label encoding
- Image normalization
- Data augmentation (augmentation experiment only)

The processed dataset was then loaded using a custom PyTorch `Dataset` class for efficient batch training.

## Project Structure

The project follows a modular structure to improve maintainability, reproducibility, and scalability. Source code, datasets, trained models, experimental results, and documentation are organized into dedicated directories, making it easier to extend or reproduce the complete machine learning pipeline.

```text
Poultry-Flock-Health-Triage-and-Monitoring-System
│
├── artifacts/
│   ├── figures/                # Training curves, confusion matrices, Grad-CAM visualizations
│   ├── model_comparison/       # Final comparison tables and project outputs
│   ├── models/                 # Trained model checkpoints and training history
│   └── reports/                # Dataset manifest and generated reports
│
├── data/
│   ├── processed/              # Train, validation and test CSV files
│   └── raw/                    # Original poultry image dataset
│
├── docs/
│   ├── modeling/               # Model documentation
│   └── images/                 # Images used in the README
│
├── notebooks/                  # Jupyter notebooks for experimentation
│
├── scripts/                    # Training, evaluation and visualization scripts
│
├── src/
│   ├── config.py               # Project configuration
│   ├── data/                   # Dataset loading and preprocessing
│   ├── evaluation/             # Evaluation metrics and utilities
│   ├── explainability/         # Grad-CAM implementation
│   ├── models/                 # CNN and ResNet18 architectures
│   ├── training/               # Training loops and utilities
│   └── config/                  # Helper functions
│
├── README.md                   # Project documentation
├── CHANGELOG.md                # Project version history
├── LICENSE                     # MIT License
├── requirements.txt            # Project dependencies
├── requirements-dev.txt        # Development dependencies
└── .gitignore                  # Ignored files
```
### Directory Description

| Directory | Purpose |
|-----------|---------|
| **artifacts/** | Stores all generated outputs, including trained models, training histories, evaluation figures, and experiment reports. |
| **data/** | Contains the original dataset and processed train, validation, and test splits used throughout the project. |
| **docs/** | Holds detailed experiment documentation and supporting images used in the project documentation. |
| **notebooks/** | Interactive notebooks used for exploratory data analysis and experimentation. |
| **scripts/** | Entry-point scripts for training, evaluation, explainability analysis, and other project tasks. |
| **src/** | Core project source code, including models, datasets, preprocessing, training, evaluation, and utility modules. |

The modular design separates data processing, model development, evaluation, and documentation, allowing each component to be developed, tested, and maintained independently. This organization also facilitates reproducibility and simplifies future extensions of the project.

## Data Audit

Before model development, a comprehensive data audit was conducted to verify the integrity and quality of the dataset. The objective was to identify potential issues that could negatively affect model training and evaluation.

The audit process included:

- Automatic dataset manifest generation.
- Verification of image file paths.
- Detection of missing or corrupted images.
- Duplicate image inspection.
- Validation of class labels.
- Image dimension verification.
- Dataset summary generation.

### Audit Summary

| Property | Value |
|----------|-------:|
| Total Images | 14,618 |
| Healthy Images | 7,991 |
| Unhealthy Images | 6,627 |
| Missing Files | 0 |
| Corrupted Images | 0 |
| Image Format | JPG |
| Color Mode | RGB |
| Image Resolution | 100 × 100 |

The audit confirmed that the dataset was complete and suitable for supervised learning without requiring additional cleaning or manual correction.

The generated dataset manifest subsequently served as the foundation for preprocessing, dataset splitting, and model training.

## Exploratory Data Analysis
A comprehensive exploratory data analysis (EDA) was conducted to better understand the dataset before model development. The objective was to assess data quality, identify potential issues, examine class distribution, and gain insights into the visual characteristics of healthy and unhealthy poultry droppings.

### Dataset Composition

The dataset consists of **14,618 RGB images** belonging to two classes:

| Class | Images |
|--------|-------:|
| Healthy | 7,991 |
| Unhealthy | 6,627 |

The relatively balanced class distribution minimizes the risk of model bias toward either class.

---

### Data Quality Assessment

Several quality checks were performed before training:

- Verified image integrity.
- Confirmed consistent image dimensions (100 × 100 pixels).
- Checked for missing or corrupted files.
- Examined duplicate images.
- Validated class labels.

The dataset was found to be clean and suitable for supervised image classification.

---

### Statistical Analysis

Basic image statistics were computed to understand the overall characteristics of the dataset.

| Statistic | Value |
|-----------|------:|
| Image Size | 100 × 100 pixels |
| Color Channels | RGB |
| Average Red Intensity | 131.86 |
| Average Green Intensity | 118.61 |
| Average Blue Intensity | 101.14 |

These statistics informed the preprocessing pipeline and normalization strategy used during model training.

---

### Visual Inspection

Representative samples from both classes were manually inspected to understand the visual patterns present in the dataset.

Key observations include:

- Healthy droppings generally appear firmer with more compact structures.
- Unhealthy droppings frequently exhibit watery or loose textures.
- Background litter remains relatively consistent across images.
- Some samples present subtle visual differences, making classification a challenging computer vision task.

---

### Key Findings

The exploratory analysis revealed several important characteristics of the dataset:

- The dataset is reasonably balanced between healthy and unhealthy classes.
- Images are uniform in size and color format, simplifying preprocessing.
- Disease-related visual differences are primarily associated with texture, consistency, colour, and moisture characteristics.
- Certain samples exhibit overlapping visual features, indicating that robust feature extraction is required for accurate classification.
- Background information may contribute useful contextual cues but also has the potential to introduce unwanted bias.

These findings informed the preprocessing pipeline, model architecture selection, and explainability analysis conducted throughout the project.

> **Note:** A detailed EDA report, including visualizations and statistical analyses, is available in the `docs/` directory.

## Data Preprocessing

A structured preprocessing pipeline was implemented to ensure data consistency, reproducibility, and fair model evaluation across all experiments.

### Dataset Manifest Generation

A dataset manifest was automatically generated by scanning the dataset directory and recording metadata for every image, including:

- Relative image path
- Class name
- Numerical class label

The manifest serves as the single source of truth for all subsequent preprocessing and training stages.

---

### Data Validation

Before model training, several validation checks were performed to ensure dataset integrity.

These included:

- Verification of image file existence
- Detection of missing values
- Validation of class labels
- Inspection of image dimensions
- Duplicate image analysis

The dataset passed all validation checks without requiring manual correction.

---

### Label Encoding

The categorical class labels were converted into numerical labels suitable for supervised learning.

| Class | Encoded Label |
|--------|--------------:|
| Healthy | 0 |
| Unhealthy | 1 |

---

### Train, Validation and Test Split

The dataset was divided using stratified sampling to preserve the original class distribution.

| Dataset | Samples | Percentage |
|---------|--------:|-----------:|
| Training | 11,694 | 80% |
| Validation | 1,462 | 10% |
| Test | 1,462 | 10% |

Using identical data splits across all experiments ensured fair comparison between different model architectures and training strategies.

---

### Image Preprocessing

Different preprocessing pipelines were employed depending on the experiment.

#### Baseline CNN

Images were:

- resized to **100 × 100 pixels**
- converted to PyTorch tensors

No data augmentation was applied during baseline training.

---

#### CNN with Data Augmentation

To improve generalization and reduce overfitting, online data augmentation was introduced during training.

The augmentation pipeline consisted of:

- Random Horizontal Flip (50% probability)
- Image Resize
- Tensor Conversion
- Normalization

Validation and test images were not augmented, ensuring unbiased model evaluation.

---

#### Fine-Tuned ResNet18

Since ResNet18 was initialized with ImageNet pretrained weights, the preprocessing pipeline followed the ImageNet standard.

Images were:

- resized to **224 × 224 pixels**
- converted to tensors
- normalized using ImageNet statistics

```python
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

Using the same normalization employed during ImageNet pretraining ensures compatibility with the learned feature representations.

---

### Custom PyTorch Dataset

A custom `PoultryDataset` class was developed to streamline data loading.

The dataset class is responsible for:

- reading image paths from the processed CSV files,
- loading images from disk,
- applying the appropriate preprocessing pipeline,
- returning image tensors together with their corresponding labels.

This modular design allows different preprocessing strategies to be used seamlessly across experiments while keeping the training pipeline clean and reusable.

---

### Outcome

The preprocessing pipeline established a reliable and reproducible foundation for all experiments by ensuring:

- consistent data loading,
- standardized preprocessing,
- fair train-validation-test separation,
- experiment-specific transformations,
- compatibility with both custom CNN and transfer learning models.

## Model Development

Three deep learning models were developed and evaluated to investigate the effectiveness of different training strategies for poultry dropping classification. Each experiment was designed to build upon the previous one, allowing a systematic comparison between a custom convolutional neural network, data augmentation, and transfer learning.

---

### Baseline CNN

The first experiment established a performance baseline using a custom Convolutional Neural Network (CNN) developed entirely from scratch.

The architecture consists of three convolutional blocks, each followed by a Rectified Linear Unit (ReLU) activation function and max pooling operation for progressive feature extraction and spatial downsampling.

The extracted feature maps are flattened and passed through fully connected layers to perform binary classification between healthy and unhealthy poultry droppings.

#### Model Architecture

- Three Convolutional Layers
- ReLU Activation
- Max Pooling
- Fully Connected Classifier
- Softmax Output (CrossEntropyLoss)

#### Training Configuration

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |
| Input Size | 100 × 100 |

The baseline model served as the reference point for all subsequent experiments.

---

### CNN with Data Augmentation

The second experiment aimed to improve the generalization capability of the baseline CNN through online data augmentation.

Rather than modifying the model architecture, only the training data pipeline was enhanced by introducing random horizontal flipping during training.

This exposed the model to more diverse image variations while preserving the original semantic content of the poultry droppings.

#### Data Augmentation Strategy

- Random Horizontal Flip (Probability = 0.5)
- transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
- transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
        )

The validation and test datasets remained unchanged to ensure unbiased model evaluation.

All other training hyperparameters remained identical to those used in the baseline experiment, allowing performance improvements to be attributed solely to the introduction of data augmentation.

#### Motivation

Data augmentation reduces overfitting by encouraging the model to learn more robust and invariant feature representations rather than memorizing specific image orientations.

---

### Fine-Tuned ResNet18

The final experiment investigated whether transfer learning could improve classification performance.

A pretrained ResNet18 model initialized with ImageNet weights was adopted as the feature extractor.

Instead of training the entire network from scratch, transfer learning was employed to leverage the rich visual representations learned from millions of natural images.

#### Transfer Learning Strategy

Initially:

- All pretrained convolutional layers were frozen.
- The final fully connected classification layer was replaced with a new binary classifier.
- Only the classifier was trained.

After establishing the frozen baseline, the final residual block (`layer4`) was unfrozen to allow limited fine-tuning on the poultry dataset.

This approach enabled higher-level feature representations to adapt to the domain-specific characteristics of poultry droppings while preserving the low-level features learned during ImageNet pretraining.

#### Training Configuration

| Parameter | Value |
|-----------|--------|
| Backbone | ResNet18 |
| Pretrained Weights | ImageNet |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Epochs | 20 |
| Batch Size | 64 |
| Input Size | 224 × 224 |

#### Motivation

Transfer learning was evaluated to determine whether pretrained visual features could outperform a task-specific CNN on a relatively small medical-style image dataset.

---

### Experimental Workflow

The overall model development process followed a progressive experimental design:

```text
Baseline CNN
        │
        ▼
Performance Evaluation
        │
        ▼
CNN + Data Augmentation
        │
        ▼
Performance Evaluation
        │
        ▼
Fine-Tuned ResNet18
        │
        ▼
Model Comparison
        │
        ▼
Explainability Analysis (Grad-CAM)
```

This iterative workflow enabled systematic evaluation of each improvement while maintaining identical train, validation, and test splits across all experiments.


## Experimental Results

Each model was evaluated on the same held-out test set using identical evaluation metrics to ensure a fair comparison.

The following performance metrics were computed for every experiment:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training and Validation Curves

---

### Experiment 1 — Baseline CNN

The baseline CNN established a strong performance benchmark despite being trained entirely from scratch.

| Metric | Value |
|---------|-------:|
| Accuracy | **98.29%** |
| Precision | **98.04%** |
| Recall | **98.19%** |
| F1-Score | **98.12%** |

#### Confusion Matrix

| | Predicted Healthy | Predicted Unhealthy |
|---|---:|---:|
| **Actual Healthy** | 786 | 13 |
| **Actual Unhealthy** | 12 | 651 |

The confusion matrix indicates that only **25 out of 1,462** test samples were misclassified, demonstrating excellent discrimination between healthy and unhealthy poultry droppings.

---

### Experiment 2 — CNN with Data Augmentation

Introducing online data augmentation improved the model's ability to generalize while preserving the original CNN architecture.

| Metric | Value |
|---------|-------:|
| Accuracy | **98.63%** |
| Precision | **98.48%** |
| Recall | **98.34%** |
| F1-Score | **98.41%** |

The augmented CNN achieved the best overall performance among all evaluated models, indicating that even a simple augmentation strategy can significantly improve generalization.

---

### Experiment 3 — Fine-Tuned ResNet18

Transfer learning was investigated using a pretrained ResNet18 model.

After replacing the final classifier and fine-tuning the last residual block, the following performance was obtained.

| Metric | Value |
|---------|-------:|
| Accuracy | **88.03%** |
| Precision | **87.77%** |
| Recall | **85.52%** |
| F1-Score | **86.63%** |

#### Confusion Matrix

| | Predicted Healthy | Predicted Unhealthy |
|---|---:|---:|
| **Actual Healthy** | 720 | 79 |
| **Actual Unhealthy** | 96 | 567 |

Although the ResNet18 model benefited from transfer learning and produced highly localized Grad-CAM visualizations, its quantitative performance was substantially lower than that of the custom CNN models.

This suggests that pretrained ImageNet features were less effective for this specialized poultry disease classification task than features learned directly from the target dataset.

---

### Training Behaviour

Training and validation curves were monitored throughout each experiment to assess convergence and identify potential overfitting.

Key observations include:

- The baseline CNN converged rapidly while maintaining excellent validation performance.
- Data augmentation improved validation stability and produced the highest overall test performance.
- Fine-tuned ResNet18 converged smoothly but plateaued at a considerably lower validation accuracy than the custom CNN models.

Overall, no severe overfitting was observed in the final CNN models, indicating that the chosen training strategy generalized effectively to unseen data.

## Model Comparison

Three different deep learning approaches were investigated to determine the most effective model for poultry dropping classification.

Although each model employed a different learning strategy, all experiments were evaluated using the same train, validation, and test splits to ensure a fair comparison.

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------:|----------:|--------:|---------:|
| Baseline CNN | **98.29%** | **98.04%** | **98.19%** | **98.12%** |
| CNN + Data Augmentation | **98.63%** | **98.48%** | **98.34%** | **98.41%** |
| Fine-Tuned ResNet18 | **88.03%** | **87.77%** | **85.52%** | **86.63%** |

---

### Comparative Analysis

#### Baseline CNN

The custom CNN established an excellent baseline despite being trained entirely from scratch.

**Strengths**

- High classification accuracy.
- Fast convergence during training.
- Lightweight architecture.
- Learned task-specific features directly from the poultry dataset.

**Limitations**

- Slightly more sensitive to variations in image appearance than the augmented model.
- Grad-CAM visualizations exhibited relatively coarse attention maps.

---

#### CNN with Data Augmentation

Introducing online data augmentation resulted in the strongest overall model.

Compared with the baseline CNN, data augmentation improved the model's ability to generalize without increasing architectural complexity.

**Strengths**

- Highest overall accuracy.
- Improved precision, recall and F1-score.
- Better robustness to unseen samples.
- More stable validation performance.
- Consistent Grad-CAM attention on poultry droppings.

**Limitations**

- Slight increase in training time due to online augmentation.

---

#### Fine-Tuned ResNet18

Transfer learning was evaluated using a pretrained ResNet18 initialized with ImageNet weights.

Although the model successfully learned to localize poultry droppings, it achieved substantially lower classification performance than both custom CNN models.

**Strengths**

- Benefited from pretrained ImageNet features.
- Produced the sharpest and most concentrated Grad-CAM visualizations.
- Required training of only a small portion of the network during fine-tuning.

**Limitations**

- Lowest quantitative performance.
- Higher computational complexity.
- Larger model size.
- Pretrained features transferred less effectively to this specialized poultry imagery dataset.

---

### Key Findings

Several important observations emerged from the experiments.

- A custom CNN trained specifically for poultry dropping classification outperformed the pretrained ResNet18 model.
- Introducing a simple data augmentation strategy improved generalization and produced the best overall performance.
- Strong visual localization, as demonstrated by ResNet18's Grad-CAM results, did not necessarily correspond to superior classification accuracy.
- The experimental results suggest that domain-specific feature learning can outperform transfer learning when sufficient labelled training data are available and the target domain differs substantially from the source domain.

---

### Recommended Model

Based on the overall experimental evaluation, the **CNN with Data Augmentation** is selected as the final model for deployment.

The model provides the best balance between predictive performance, computational efficiency, robustness, and interpretability.

Its superior classification accuracy, combined with stable training behaviour and meaningful Grad-CAM explanations, makes it the most suitable solution for automated poultry health screening within this project.


To improve model interpretability, Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize the image regions that most influenced each model's predictions.

Grad-CAM analysis was performed for:

- Baseline CNN
- CNN with Data Augmentation
- Fine-Tuned ResNet18

The generated heatmaps consistently showed that all models primarily focused on the poultry droppings rather than unrelated image regions, providing evidence that the models learned meaningful disease-related visual features.

### Key Observations

- The **Baseline CNN** focused on the droppings while also incorporating surrounding poultry litter.
- The **CNN with Data Augmentation** produced more stable and consistent attention maps across different samples.
- The **Fine-Tuned ResNet18** generated the sharpest and most concentrated activation maps, localizing attention almost exclusively on the droppings.

Interestingly, although ResNet18 produced the most visually focused Grad-CAM heatmaps, it achieved lower classification accuracy than the custom CNN models. This demonstrates that better visual localization does not necessarily translate into superior predictive performance.

Grad-CAM therefore complements the quantitative evaluation metrics by providing visual evidence that the models learned biologically relevant image features rather than relying on arbitrary background information.

> **Note:** A comprehensive Grad-CAM analysis, methodology, and discussion are available in the `docs/` directory.

## Documentation

Comprehensive technical documentation for every stage of the project is available in the `docs/` directory.

The documentation includes:

- Data Audit
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Baseline CNN Development
- CNN with Data Augmentation
- Fine-Tuned ResNet18
- Model Evaluation
- Error Analysis
- Grad-CAM Explainability Analysis

Each report describes the methodology, implementation details, observations, and conclusions for the corresponding stage of the machine learning pipeline.

## Installation

Follow the steps below to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/<IssaMuiz>/poultry-flock-health-triage-and-monitoring-system.git
cd poultry-flock-health-triage-and-monitoring-system
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

---

### 4. Download the Dataset

Download the **Poultry Birds Poo Imagery Dataset for Health Status Prediction: A Case of South-West Nigeria** and place it inside the following directory:

```text
data/
└── raw/
    └── Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/
```

---

### 5. Generate the Dataset Manifest

```bash
python scripts/create_dataset_manifest.py
```

---

### 6. Create the Train / Validation / Test Split

```bash
python scripts/create_data_split.py
```

---

### 7. Train a Model

#### Baseline CNN

```bash
python scripts/train_baseline.py
```

#### CNN with Data Augmentation

```bash
python scripts/train_augmentation.py
```

#### Fine-Tuned ResNet18

```bash
python scripts/train_resnet18.py
```

---

### 8. Evaluate a Trained Model

```bash
python scripts/evaluate_model.py
```

---

### 9. Generate Grad-CAM Visualizations

```bash
python scripts/generate_gradcam.py
```

---

### Project Requirements

- Python 3.10+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Pillow
- Scikit-learn

## Usage

After completing the installation and dataset preparation steps, the project can be used to reproduce the complete machine learning workflow.

### Generate the Dataset Manifest

Create a manifest containing image paths and labels.

```bash
python scripts/create_dataset_manifest.py
```

---

### Create the Train, Validation and Test Split

Generate stratified training, validation, and testing datasets.

```bash
python scripts/create_data_split.py
```

---

### Train a Model

Model training is performed using a single training script.

Before running the training script, configure the desired experiment by selecting the appropriate:

- model architecture,
- data transformation pipeline,
- optimizer settings (if applicable),
- checkpoint directory.

Once the desired configuration has been selected, train the model using:

```bash
python scripts/train.py
```

> **Note:** The project supports three experimental configurations:
>
> - Baseline CNN
> - CNN with Data Augmentation
> - Fine-Tuned ResNet18
>
> Each experiment requires the corresponding model and preprocessing pipeline to be selected before training.

---

### Evaluate a Trained Model

To evaluate a trained model, update the evaluation script with the appropriate:

- model architecture,
- model checkpoint,
- preprocessing pipeline.

Then run:

```bash
python scripts/evaluate_model.py
```

The evaluation pipeline reports:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

### Generate Grad-CAM Visualizations

Grad-CAM visualizations are generated using the saved model checkpoints.

Before running the visualization script, ensure that the following components match the model being analyzed:

- model architecture,
- model checkpoint,
- target convolutional layer,
- preprocessing transformation,
- input image size.

For example:

| Model | Input Size | Target Layer |
|--------|-----------:|--------------|
| Baseline CNN | 100 × 100 | Final convolutional layer |
| CNN + Data Augmentation | 100 × 100 | Final convolutional layer |
| Fine-Tuned ResNet18 | 224 × 224 | `layer4[-1]` |

Once the configuration has been updated, run:

```bash
python scripts/cnn_gradcam_demo.py
```

- or 

```bash
python scripts/resnet18_gradcam_demo.py
```

Generated Grad-CAM images are automatically saved to the `artifacts/figures/` directory.

---

### Project Outputs

Running the complete workflow produces:

- Trained model checkpoints
- Training history
- Dataset manifest
- Processed train, validation, and test splits
- Confusion matrices
- Training and validation curves
- Grad-CAM visualizations
- Experiment documentation

## Future Improvements

Although the developed models achieved excellent performance on the binary poultry dropping classification task, several opportunities exist to further improve the system.

### Expand to Multi-Class Disease Classification

The current system classifies poultry droppings as either **Healthy** or **Unhealthy**. Future work could extend the model to identify specific poultry diseases, enabling more informative diagnoses and targeted interventions.

---

### Collect a Larger and More Diverse Dataset

Model robustness could be improved by incorporating images captured under varying:

- lighting conditions,
- camera angles,
- backgrounds,
- poultry breeds,
- environmental conditions.

A larger and more diverse dataset would improve generalization to real-world farm environments.

---

### Investigate Advanced Deep Learning Architectures

Future experiments could compare the current models with more recent computer vision architectures, including:

- EfficientNet
- ConvNeXt
- DenseNet
- Vision Transformers (ViT)
- Swin Transformer

Such comparisons would provide additional insight into the suitability of modern architectures for poultry disease classification.

---

### Hyperparameter Optimization

The models were trained using manually selected hyperparameters.

Future work could employ automated hyperparameter optimization techniques, such as:

- Grid Search
- Random Search
- Bayesian Optimization
- Optuna

to identify more optimal training configurations.

---

### Improve Explainability

Although Grad-CAM provides valuable visual explanations, future work could integrate additional explainability techniques such as:

- Grad-CAM++
- Score-CAM
- Integrated Gradients
- SHAP
- LIME

Combining multiple explainability methods would provide a more comprehensive understanding of model behaviour.

---

### Deploy as a Real-Time Decision Support System

A practical extension of this project would be to deploy the trained model as an application that assists poultry farmers with rapid health assessment.

Possible deployment platforms include:

- Web applications
- Mobile applications
- Edge devices
- Cloud-based monitoring systems

Such a system could enable real-time poultry health screening directly from images captured in the field.

---

### Continuous Model Improvement

As additional labelled data become available, the models can be periodically retrained and evaluated to improve performance and adapt to new disease patterns.

Implementing a continuous learning workflow would support long-term system reliability and scalability.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

This project builds upon publicly available datasets, open-source libraries, and established deep learning research.

### Dataset

- Poultry Birds Poo Imagery Dataset for Health Status Prediction: A Case of South-West Nigeria.

### Frameworks and Libraries

- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Pillow

### Research and Documentation

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.*
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.*

## Author

**Issa Muiz**

Machine Learning and Deep Learning Engineer with a strong interest in Computer Vision, Explainable AI, and AI solutions for agriculture.

This project demonstrates an end-to-end deep learning workflow, covering data exploration, preprocessing, model development, evaluation, explainability, and technical documentation.

If you have questions, suggestions, or would like to collaborate, feel free to open an issue or submit a pull request.