from pathlib import Path
import os

# Detect Kaggle
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:

    DATA_ROOT = Path(
        "/kaggle/input/datasets/issamuiz/"
        "poultry-birds-poo-imagery-dataset/"
        "Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/"
        "Dataset"
    )

else:

    DATA_ROOT = Path(
        "data/raw/Poultry Birds Poo Imagery Dataset for Health Status Prediction A Case of South-West Nigeria/Dataset"
    )
