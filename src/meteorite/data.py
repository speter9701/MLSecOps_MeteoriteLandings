"""Loading and sampling of the raw meteorite landings dataset."""
from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV file of the NASA meteorite landings dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path} (did you run 'dvc pull'?)")
    return pd.read_csv(path)
