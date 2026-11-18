"""Cleaning rules and the sklearn preprocessing pipeline."""
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from meteorite.schemas import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

MIN_YEAR: Final[int] = 800
MAX_YEAR: Final[int] = 2016
ANTARCTICA_MAX_LAT: Final[float] = -60.0
ENGINEERED_FEATURES: Final[tuple[str, ...]] = ("is_antarctica", "abs_lat")
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
    """Apply the cleaning rules; returns a new frame with only valid targets."""
    out = df.rename(columns=COLUMN_ALIASES).drop(columns=list(DROP_COLUMNS), errors="ignore")
    out["year"] = parse_year(pd.Series(out["year"], dtype=object))
    year = numeric_column(out, "year")
    out.loc[~year.between(MIN_YEAR, MAX_YEAR), "year"] = np.nan
    lat, lon = numeric_column(out, "reclat"), numeric_column(out, "reclong")
    out.loc[~lat.between(-90, 90), "reclat"] = np.nan
    out.loc[~lon.between(-180, 180), "reclong"] = np.nan
    out.loc[(lat == 0) & (lon == 0), ["reclat", "reclong"]] = np.nan
    mass = numeric_column(out, TARGET_COLUMN)
    out = out.loc[mass.notna() & (mass > 0)]
    return out.drop_duplicates().reset_index(drop=True)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features: is the find in Antarctica, distance from the equator."""
    out = df.copy()
    lat = numeric_column(out, "reclat")
    out["is_antarctica"] = (lat < ANTARCTICA_MAX_LAT).astype(float)
    out["abs_lat"] = lat.abs()
    return out


def build_preprocessor(min_category_frequency: int = 10) -> Pipeline:
    """Feature engineering, imputation, scaling and one-hot encoding."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_category_frequency, sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, [*NUMERIC_FEATURES, *ENGINEERED_FEATURES]),
        ("cat", categorical, list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("geo", FunctionTransformer(add_geo_features)), ("columns", columns)])
