from pathlib import Path

from src.data.dataloader import create_dataloader

TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")


train_loader, val_loader, test_loader = create_dataloader(TRAIN, VAL, TEST)

print("=" * 60)
print("DATALOADER INFORMATION")
print("=" * 60)

print(f"Training batch: {len(train_loader)}")
print(f"Validation batch: {len(val_loader)}")
print(f"Test batch: {len(test_loader)}")

image, label = next(iter(train_loader))

print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Label dtype: {label.dtype}")
