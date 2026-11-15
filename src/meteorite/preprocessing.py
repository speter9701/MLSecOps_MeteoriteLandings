"""Cleaning rules and the sklearn preprocessing pipeline."""
from typing import Final

import pandas as pd

from meteorite.schemas import TARGET_COLUMN

COLUMN_ALIASES: Final[dict[str, str]] = {"mass (g)": TARGET_COLUMN}
DROP_COLUMNS: Final[tuple[str, ...]] = ("name", "id", "GeoLocation")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules and return a new frame."""
    out = df.rename(columns=COLUMN_ALIASES).drop(columns=list(DROP_COLUMNS), errors="ignore")
    return out.reset_index(drop=True)
