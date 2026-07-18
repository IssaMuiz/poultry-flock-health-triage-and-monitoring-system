import torch
from pathlib import Path
from src.models.resnet18_transfer import ResNet18Transfer
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics import calculate_metrics
from src.data.dataloader import create_dataloader

TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")

MODEL_PATH = Path("artifacts/models/resnet18/resnet18_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = create_dataloader(TRAIN, VAL, TEST, return_paths=True)

model = ResNet18Transfer()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))

model.to(device)
predictions, labels = evaluate(model, test_loader, device)

metrics, confusion_matrix, classification_report = calculate_metrics(
    labels, predictions
)


print("=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

for key, value in metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(classification_report)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix)
