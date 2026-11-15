from meteorite.schemas import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def test_feature_columns_are_disjoint_from_target() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert FEATURE_COLUMNS == CATEGORICAL_FEATURES + NUMERIC_FEATURES
