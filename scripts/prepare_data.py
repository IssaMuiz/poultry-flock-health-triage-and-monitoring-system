import pandas as pd
from pathlib import Path
from src.data.split_data import split_dataset

manifest_data_path = Path("artifacts/reports/dataset_manifest.csv")
OUTPUT_DIR = Path("data/processed")

manifest_data = pd.read_csv(manifest_data_path)

train_df, val_df, test_df = split_dataset(manifest_data)

train_df.to_csv(OUTPUT_DIR / "train_df.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_df.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_df.csv", index=False)

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

print(f"Train_image: {len(train_df)}")
print(f"Validation image: {len(val_df)}")
print(f"Test image: {len(test_df)}")

print("=" * 60)
print("DATASET SPLITTING CLASS DISTRIBUTION")
print("=" * 60)

print("\nTrain")
print(train_df["class_name"].value_counts(normalize=True))
print("\nValidation")
print(val_df["class_name"].value_counts(normalize=True))
print("\nTest")
print(test_df["class_name"].value_counts(normalize=True))
