from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataloader import create_dataloader
from src.models.cnn import BaselineCNN
from src.training.engine import train_one_epoch, validate_one_epoch

# Configuration
TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# Dataloader
train_loader, val_loader, test_loader = create_dataloader(
    TRAIN, VAL, TEST, batch_size=BATCH_SIZE
)


# Model
model = BaselineCNN().to(device)


# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

print(
    f"{epoch+1} / {EPOCHS}",
    f"Train loss: {train_loss:.4f}",
    f"Train accuracy: {train_acc:.4f}",
    f"Val loss: {val_loss:.4f}",
    f"Val accuracy: {val_acc:.4f}",
)
