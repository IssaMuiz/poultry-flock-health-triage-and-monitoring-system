from pathlib import Path
import matplotlib.pyplot as plt

SAVE_DIR = Path("artifacts/reports/figures/comparison")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

models = [
    "Baseline CNN",
    "CNN +\nAugmentation",
    "Fine-tuned\nResNet18",
]

accuracy = [
    98.29,
    98.63,
    88.03,
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracy)

plt.ylabel("Test Accuracy (%)")
plt.title("Model Performance Comparison")

plt.ylim(80, 100)

for i, value in enumerate(accuracy):
    plt.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.grid(axis="y")

plt.savefig(
    SAVE_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Comparison chart saved successfully!")
