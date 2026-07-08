import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.dataloader import create_dataloader
from src.evaluation.error_analysis import collect_misclassified_images
from src.models.cnn import BaselineCNN

TRAIN = Path("data/processed/train_df.csv")
VAL = Path("data/processed/val_df.csv")
TEST = Path("data/processed/test_df.csv")


MODEL_PATH = Path("artifacts/models/baseline_cnn.pth")

model = BaselineCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = create_dataloader(TRAIN, VAL, TEST, return_paths=True)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

model.to(device)


misclassified_images = collect_misclassified_images(model, test_loader, device)

print(f"Total misclassifed images: {len(misclassified_images)}")


class_names = {0: "Healthy", 1: "Unhealthy"}

for misclassified in misclassified_images[:10]:

    image = misclassified["image"]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverse normalization
    image = image * std + mean

    # Clamp pixel values to [0, 1]
    image = image.clamp(0, 1)

    # Convert to HWC format for matplotlib
    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))

    plt.imshow(image)

    plt.title(
        f"Actual: {class_names[misclassified['actual_label']]}\n"
        f"Predicted: {class_names[misclassified['predicted_label']]}"
    )

    plt.axis("off")

    plt.show()

    print(misclassified["path"])
