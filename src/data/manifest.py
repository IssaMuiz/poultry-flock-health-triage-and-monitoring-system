from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def create_dataset_manifest(data_root: Path) -> pd.DataFrame:
    """
    Create a dataset manifest containing every image and its label.

    Parameters
    ----------
    data_root : Path
        Root dataset directory.

    Returns
    -------
    pd.DataFrame
        Dataset manifest.
    """

    records = []

    class_names = sorted(
        [
            folder.name
            for folder in data_root.iterdir()
            if folder.is_dir and not folder.name.startswith(".")
        ]
    )

    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:

        class_folder = data_root / class_name

        for image_path in class_folder.iterdir():

            if image_path.is_file and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append(
                    {
                        "image_path": str(image_path),
                        "class_name": class_name,
                        "label": label_map[class_name],
                    }
                )

    return pd.DataFrame(records)
