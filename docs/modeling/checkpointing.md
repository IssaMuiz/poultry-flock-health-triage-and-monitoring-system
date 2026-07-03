# Model Checkpointing

## Objective

Model checkpointing ensures that the best-performing model is preserved during training.

Instead of saving the model after every epoch, the project saves the model only when the validation accuracy improves.

---

## Checkpoint Strategy

Metric monitored:

* Validation Accuracy

Saving condition:

* Current validation accuracy is greater than the best validation accuracy observed so far.

---

## Saved File

Location:

artifacts/models/baseline_cnn.pth

Contents:

* Model state dictionary (learned weights)

---

## Benefits

* Prevents loss of the best-performing model.
* Enables reproducible evaluation.
* Provides a trained model for inference and deployment.
* Avoids saving models with worse validation performance.

---

## Loading the Model

To use the trained model:

1. Instantiate the `BaselineCNN` architecture.
2. Load the saved `state_dict`.
3. Move the model to the target device.
4. Switch the model to evaluation mode using `model.eval()`.

This follows the recommended workflow for loading trained PyTorch models.
