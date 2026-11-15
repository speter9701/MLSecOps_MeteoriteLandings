import math

import pandas as pd
import pytest

from meteorite.preprocessing import clean_data, parse_year


def make_frame(**overrides: list[object]) -> pd.DataFrame:
    """Two valid raw rows; keyword arguments replace whole columns."""
    data: dict[str, list[object]] = {
        "name": ["a", "b"],
        "id": [1, 2],
        "nametype": ["Valid", "Valid"],
        "recclass": ["L6", "H5"],
        "mass (g)": [10.0, 20.0],
        "fall": ["Fell", "Found"],
        "year": [1990.0, 2000.0],
        "reclat": [10.0, -75.0],
        "reclong": [20.0, 160.0],
        "GeoLocation": ["(10.0, 20.0)", "(-75.0, 160.0)"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_mass_column_is_renamed_and_identifiers_dropped() -> None:
    cleaned = clean_data(make_frame())
    assert "mass" in cleaned.columns
    assert not {"mass (g)", "name", "id", "GeoLocation"} & set(cleaned.columns)


@pytest.mark.parametrize(
    "raw,expected",
    [("1880.0", 1880.0), ("01/01/1880 12:00:00 AM", 1880.0), (1998, 1998.0)],
)
def test_parse_year_formats(raw: object, expected: float) -> None:
    assert parse_year(pd.Series([raw], dtype=object)).iloc[0] == expected


@pytest.mark.parametrize("raw", [None, "abc"])
def test_parse_year_unparseable_is_nan(raw: object) -> None:
    assert math.isnan(parse_year(pd.Series([raw], dtype=object)).iloc[0])
