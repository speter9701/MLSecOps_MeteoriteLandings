import pytest
from pydantic import ValidationError

from meteorite.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    MeteoriteFeatures,
)


def test_feature_columns_are_disjoint_from_target() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert FEATURE_COLUMNS == CATEGORICAL_FEATURES + NUMERIC_FEATURES


def test_feature_columns_match_schema_fields() -> None:
    assert set(FEATURE_COLUMNS) == set(MeteoriteFeatures.model_fields)


def test_valid_record_uses_defaults() -> None:
    record = MeteoriteFeatures(recclass="L6")
    assert record.nametype == "Valid"
    assert record.fall == "Found"
    assert record.year is None and record.reclat is None and record.reclong is None


@pytest.mark.parametrize(
    "field,value",
    [("reclat", 95.0), ("reclong", -181.0), ("year", 500.0), ("fall", "Lost"), ("nametype", "Other"), ("recclass", "")],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MeteoriteFeatures.model_validate({"recclass": "L6", field: value})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MeteoriteFeatures.model_validate({"recclass": "L6", "mass": 10.0})
