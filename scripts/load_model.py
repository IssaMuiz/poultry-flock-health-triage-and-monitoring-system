from pathlib import Path
import torch
from src.models.cnn import BaselineCNN

MODEL_PATH = Path("artifacts/models/baseline_cnn.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BaselineCNN()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

print("=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)

print(model)
