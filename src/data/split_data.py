import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation and test sets.

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest.

    Returns
    -------
    tuple
        Train, validation and test DataFrames.
    """

    train_df, temp_df = train_test_split(
        manifest, test_size=0.2, random_state=42, stratify=manifest["label"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
