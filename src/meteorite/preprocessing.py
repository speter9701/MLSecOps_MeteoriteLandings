"""Cleaning rules and the sklearn preprocessing pipeline."""
from typing import Final

import numpy as np
import pandas as pd

from meteorite.schemas import TARGET_COLUMN

MIN_YEAR: Final[int] = 800
MAX_YEAR: Final[int] = 2016
COLUMN_ALIASES: Final[dict[str, str]] = {"mass (g)": TARGET_COLUMN}
DROP_COLUMNS: Final[tuple[str, ...]] = ("name", "id", "GeoLocation")


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Typed access to one column as floats (``df[name]`` is typed as Series | DataFrame)."""
    return pd.Series(pd.to_numeric(df[name], errors="coerce"), index=df.index, dtype=float, name=name)


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
    year = numeric_column(out, "year")
    out.loc[~year.between(MIN_YEAR, MAX_YEAR), "year"] = np.nan
    lat, lon = numeric_column(out, "reclat"), numeric_column(out, "reclong")
    out.loc[~lat.between(-90, 90), "reclat"] = np.nan
    out.loc[~lon.between(-180, 180), "reclong"] = np.nan
    out.loc[(lat == 0) & (lon == 0), ["reclat", "reclong"]] = np.nan
    return out.reset_index(drop=True)
