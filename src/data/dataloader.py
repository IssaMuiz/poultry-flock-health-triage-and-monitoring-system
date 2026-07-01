from pathlib import Path

from torch.utils.data import DataLoader
from src.data.dataset import PoultryDataset
from src.data.transforms import train_transform, validation_transform


def create_dataloader(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create train, validation and test DataLoaders.
    """

    train_dataset = PoultryDataset(csv_file=train_csv, transform=train_transform)

    val_dataset = PoultryDataset(csv_file=val_csv, transform=validation_transform)

    test_dataset = PoultryDataset(csv_file=test_csv, transform=validation_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
