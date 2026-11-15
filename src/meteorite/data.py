"""Loading and sampling of the raw meteorite landings dataset."""
from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV file of the NASA meteorite landings dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path} (did you run 'dvc pull'?)")
    return pd.read_csv(path)


def sample_dataset(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    """Return a reproducible random sample of at most ``n_rows`` rows."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if len(df) <= n_rows:
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)
