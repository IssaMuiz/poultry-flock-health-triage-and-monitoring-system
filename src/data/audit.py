from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter

# Supported image formats
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def get_dataset_summary(data_root: Path) -> dict:
    """
    Generate a summary of an image classification dataset.

    Parameters
    ----------
    data_root : Path
        Root directory containing one folder per class.

    Returns
    -------
    dict
        Dataset summary.
    """

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_root}")

    if not data_root.is_dir():
        raise NotADirectoryError(f"{data_root} is not a directory.")

    summary = {
        "dataset_path": str(data_root.resolve()),
        "num_classes": 0,
        "class_names": [],
        "class_distribution": {},
        "total_images": 0,
    }

    # Find class folders
    class_folders = sorted(
        [
            folder
            for folder in data_root.iterdir()
            if folder.is_dir() and not folder.name.startswith(".")
        ]
    )

    summary["num_classes"] = len(class_folders)

    total_images = 0

    for class_folder in class_folders:

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        image_count = len(image_files)

        summary["class_names"].append(class_folder.name)

        summary["class_distribution"][class_folder.name] = image_count

        total_images += image_count

    summary["total_images"] = total_images

    return summary


def find_corrupt_images(data_root: Path) -> pd.DataFrame:
    """
    Scan the dataset and identify corrupt or unreadable images.

    Parameters
    ----------
    data_root : Path
        Root dataset directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing corrupt image paths and error messages.
    """

    corrupt_images = []

    class_folders = [
        folder
        for folder in data_root.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
    ]

    for class_folder in class_folders:

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for image_path in tqdm(
            image_files,
            desc=f"Checking {class_folder.name}",
        ):

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception as error:

                corrupt_images.append(
                    {
                        "class": class_folder.name,
                        "image_path": str(image_path),
                        "error": str(error),
                    }
                )

    return pd.DataFrame(corrupt_images, columns=["class", "image_path", "error"])


def analyze_image_dimensions(data_root: Path) -> pd.DataFrame:
    """
    Analyze image dimensions in the dataset.

    Parameters
    ----------
    data_root : Path
        Root dataset directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing image dimension information.
    """

    image_info = []

    # save the image folder in a list
    class_folders = [
        folder
        for folder in data_root.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
    ]

    # checking each image file in the image folder
    for class_folder in class_folders:
        image_file = [
            file
            for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for image_path in tqdm(image_file, desc=f"Analyzing {class_folder.name}"):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                    image_info.append(
                        {
                            "class": class_folder.name,
                            "image_path": str(image_path),
                            "width": width,
                            "height": height,
                            "aspect_ratio": round(width / height, 3),
                        }
                    )
            except Exception:
                continue
    return pd.DataFrame(image_info)
