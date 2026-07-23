import torch
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import validation_transform
from src.models.cnn import BaselineCNN


class PoultryPredictor:
    """An inference pipeline for poultry health classification"""

    CLASS_NAMES = {0: "Healthy", 1: "Unhealthy"}

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BaselineCNN()

        model_path = Path("artifacts/models/augmentation/augmentation_model.pth")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )

        self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image):

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        input_tensor = validation_transform(image).unsqueeze(0).to(self.device)

        outputs = self.model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

        prediction = prediction.item()

        return {
            "prediction": self.CLASS_NAMES[prediction],
            "class_index": prediction,
            "confidence": confidence.item(),
            "probabilities": {
                "Healthy": probabilities[0][0].item(),
                "Unhealthy": probabilities[0][1].item(),
            },
        }
