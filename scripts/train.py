import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataloader import create_dataloader
from src.models.resnet18_transfer import ResNet18Transfer
from src.training.engine import train_one_epoch, validate_one_epoch

# Configuration
TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# Dataloader
train_loader, val_loader, test_loader = create_dataloader(
    TRAIN, VAL, TEST, batch_size=BATCH_SIZE
)


# Model
model = ResNet18Transfer().to(device)


# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
)

# CHECKPOINT
MODEL_PATH = Path("artifacts/models/resnet18/best_model.pth")

best_val_accuracy = 0.0

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

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

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Best model saved successfully!  val_acc: {val_acc:.4f}")


HISTORY_PATH = Path("")
history_df = pd.DataFrame(history)

history_df.to_csv("artifacts/models/resnet18/training_history.csv", index=False)

print("Training history saved successfully!")
