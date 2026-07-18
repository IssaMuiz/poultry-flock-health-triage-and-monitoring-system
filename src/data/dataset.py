from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.config import DATA_ROOT


class PoultryDataset(Dataset):
    """
    Custom PyTorch Dataset for poultry disease classification.
    """

    def __init__(self, csv_file: Path, transform=None, return_paths=False):
        """
        Parameters
        ----------
        csv_file : Path
            Path to the dataset manifest CSV.

        transform : callable, optional
            Image transformations.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = DATA_ROOT / row["image_path"]

        image = Image.open(image_path).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return image, label, row["image_path"]

        return image, label
