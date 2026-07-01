from pathlib import Path

from src.data.dataset import PoultryDataset

DATA_CSV = Path("data/processed/train_df.csv")

dataset = PoultryDataset(DATA_CSV)


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of sample: {len(dataset)}")

image, label = dataset[0]


print(type(image))
print(image.size)
print(label)
