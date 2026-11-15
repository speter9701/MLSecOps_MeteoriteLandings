import math

import numpy as np
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


def test_out_of_range_years_become_nan() -> None:
    cleaned = clean_data(make_frame(year=[2101.0, 500.0]))
    assert cleaned["year"].isna().all()


def test_invalid_and_unknown_coordinates_become_nan() -> None:
    cleaned = clean_data(make_frame(reclat=[0.0, 95.0], reclong=[0.0, 354.0]))
    assert cleaned[["reclat", "reclong"]].isna().all().all()


def test_zero_latitude_alone_is_kept() -> None:
    cleaned = clean_data(make_frame(reclat=[0.0, 1.0], reclong=[5.0, 1.0]))
    assert cleaned["reclat"].tolist() == [0.0, 1.0]


def test_rows_without_positive_mass_are_dropped() -> None:
    frame = pd.concat([make_frame(), make_frame(**{"mass (g)": [0.0, np.nan]})], ignore_index=True)
    assert clean_data(frame)["mass"].tolist() == [10.0, 20.0]


def test_duplicates_are_dropped() -> None:
    assert len(clean_data(pd.concat([make_frame(), make_frame()], ignore_index=True))) == 2


def test_clean_data_on_synthetic_frame(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    assert len(cleaned) == len(raw_frame)
    assert (cleaned["mass"] > 0).all()
