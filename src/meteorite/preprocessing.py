"""Cleaning rules and the sklearn preprocessing pipeline."""
from typing import Final

import pandas as pd

from meteorite.schemas import TARGET_COLUMN

COLUMN_ALIASES: Final[dict[str, str]] = {"mass (g)": TARGET_COLUMN}
DROP_COLUMNS: Final[tuple[str, ...]] = ("name", "id", "GeoLocation")


def parse_year(values: pd.Series) -> pd.Series:
    """Accept both ``1880.0`` and ``01/01/1880 12:00:00 AM`` style years."""
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index, dtype=float)
    digits = values.astype("string").str.extract(r"(\d{4})", expand=False)
    from_text = pd.Series(pd.to_numeric(digits, errors="coerce"), index=values.index, dtype=float)
    return numeric.fillna(from_text)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules and return a new frame."""
    out = df.rename(columns=COLUMN_ALIASES).drop(columns=list(DROP_COLUMNS), errors="ignore")
    out["year"] = parse_year(pd.Series(out["year"], dtype=object))
    return out.reset_index(drop=True)
