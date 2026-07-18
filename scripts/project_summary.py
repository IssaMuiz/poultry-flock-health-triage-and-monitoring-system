import pandas as pd

results = pd.DataFrame(
    {
        "Model": [
            "Baseline CNN",
            "CNN + Data Augmentation",
            "Fine-tuned ResNet18",
        ],
        "Accuracy": [
            98.29,
            98.63,
            88.03,
        ],
        "Precision": [
            98.04,
            98.06,
            87.77,
        ],
        "Recall": [
            98.19,
            98.94,
            85.52,
        ],
        "F1-score": [
            98.12,
            98.50,
            86.63,
        ],
    }
)

print(results)

results.to_csv(
    "artifacts/reports/project_summary.csv",
    index=False,
)

print("Summary saved successfully.")
