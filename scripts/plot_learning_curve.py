import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

HISTORY_PATH = Path("artifacts/history/training_history.csv")

OUTPUT_DIR = Path("artifacts/reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

history = pd.read_csv(HISTORY_PATH)


# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png")
plt.close()


# Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
plt.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_curve.png")
plt.close()

print("Learning Curve saved successfully!")
