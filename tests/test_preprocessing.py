import pandas as pd

from meteorite.preprocessing import clean_data


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
