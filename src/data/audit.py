from pathlib import Path

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
