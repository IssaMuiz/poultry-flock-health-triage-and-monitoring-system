from pathlib import Path
from src.data.dataloader import create_dataloader
from src.models.cnn import BaselineCNN

TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")

train_loader, _, _ = create_dataloader(TRAIN, VAL, TEST)

model = BaselineCNN()

image, label = next(iter(train_loader))

output = model(image)

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(f"Input shape: {image.shape}")
print(f"Output shape: {output.shape}")
print(f"label shape: {label.shape}")
