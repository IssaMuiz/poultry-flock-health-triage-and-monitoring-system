from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PoultryDataset(Dataset):
    """
    Custom PyTorch Dataset for poultry disease classification.
    """

    def __init__(self, csv_file: Path, transform=None):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
