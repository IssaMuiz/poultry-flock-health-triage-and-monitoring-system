# Learning Curve Analysis

## Objective

Learning curves were generated to monitor the training progress of the baseline CNN and assess its ability to generalize to unseen data.

The curves visualize how the model's accuracy and loss change throughout the training process.

---

## Accuracy Curve

The training and validation accuracy increased steadily over successive epochs.

Both curves followed a similar upward trend and remained close throughout training.

### Interpretation

* The model successfully learned meaningful visual features from the dataset.
* Validation accuracy improved alongside training accuracy, indicating good generalization.
* No significant divergence between the curves was observed, suggesting that the model did not exhibit severe overfitting.

---

## Loss Curve

The training and validation loss decreased consistently as training progressed.

Both curves showed a downward trend with only minor fluctuations near the final epochs.

### Interpretation

* The optimization process converged successfully.
* The model progressively reduced prediction errors on both the training and validation datasets.
* The absence of a large increase in validation loss indicates stable learning behavior.

---

## Overall Assessment

The learning curves demonstrate that the baseline CNN converged effectively during training.

The close agreement between the training and validation curves suggests that the model learned generalizable patterns rather than memorizing the training data.

These observations are consistent with the strong performance achieved on the independent test dataset, where the model obtained approximately 98% accuracy.
