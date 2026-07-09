from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.transforms import validation_transform
from src.explainability.gradcam import GradCAM
from src.models.cnn import BaselineCNN
from src.data.dataset import PoultryDataset

MODEL_PATH = Path("artifacts/models/augmentation/best_model.pth")

model = BaselineCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

model.to(device)

model.eval()

target_layer = model.features[7]

gradcam = GradCAM(model, target_layer)

dataset = PoultryDataset("data/processed/train_df.csv")

image_paths = dataset.data["image_path"].iloc[:10].tolist()


def show_gradcam(image_path):

    original = Image.open(image_path).convert("RGB")

    input_tensor = validation_transform(original).unsqueeze(0).to(device)

    cam, prediction = gradcam.generate(input_tensor)

    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(100, 100),
        mode="bilinear",
        align_corners=False,
    )

    cam = cam.squeeze().numpy()

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    original_np = np.array(original.resize((100, 100)))

    overlay = (0.4 * heatmap + 0.6 * original_np).astype(np.uint8)

    return overlay, prediction


fig, axes = plt.subplots(
    2,
    5,
    figsize=(18, 8),
)

class_names = {0: "Healthy", 1: "Unhealthy"}

for ax, image_path in zip(axes.flatten(), image_paths):

    overlay, prediction = show_gradcam(image_path)

    ax.imshow(overlay)

    ax.set_title(class_names[prediction])

    ax.axis("off")

plt.tight_layout()

plt.show()
