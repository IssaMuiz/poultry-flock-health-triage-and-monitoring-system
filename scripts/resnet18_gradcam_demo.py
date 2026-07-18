from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.transforms import validation_transform
from src.explainability.gradcam import GradCAM
from src.models.resnet18_transfer import ResNet18Transfer
from src.data.dataset import PoultryDataset
from src.config import DATA_ROOT

MODEL_PATH = Path("artifacts/models/resnet18/resnet18_model.pth")

SAVE_DIR = Path("artifacts/figures/resnet18_learning_chart")

model = ResNet18Transfer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

model.to(device)

model.eval()

target_layer = model.model.layer4[-1]

gradcam = GradCAM(model, target_layer)

dataset = PoultryDataset("data/processed/train_df.csv")

image_paths = dataset.data["image_path"].iloc[:10].tolist()


def show_gradcam(image_path):

    full_path = DATA_ROOT / image_path

    original = Image.open(full_path).convert("RGB")

    input_tensor = validation_transform(original).unsqueeze(0).to(device)

    cam, prediction = gradcam.generate(input_tensor)

    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(224, 224),
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

    original_np = np.array(original.resize((224, 224)))

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

plt.savefig(
    SAVE_DIR / "gradcam_grid.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
