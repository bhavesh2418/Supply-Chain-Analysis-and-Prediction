"""
data_loader.py — Load and validate the raw supply chain dataset.
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_CSV


def load_raw_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV and run basic validation checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            "Run: python scripts/download_data.py"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded DataFrame is empty — check the CSV file.")

    # Normalize column names: strip whitespace, replace spaces with underscores
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

    print(f"[data_loader] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[data_loader] Columns: {list(df.columns)}")
    return df


def quick_summary(df: pd.DataFrame) -> None:
    """Print a quick health summary of the DataFrame."""
    print("\n--- Dataset Summary ---")
    print(f"Shape        : {df.shape}")
    print(f"Duplicates   : {df.duplicated().sum()}")
    print(f"Missing vals :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    print(f"\nNumeric describe:\n{df.describe().T.to_string()}")
