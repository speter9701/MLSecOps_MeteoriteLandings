"""Shared data contract between the ML core, the REST API and monitoring."""
from typing import Final

TARGET_COLUMN: Final[str] = "mass"
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("recclass", "nametype", "fall")
NUMERIC_FEATURES: Final[tuple[str, ...]] = ("year", "reclat", "reclong")
FEATURE_COLUMNS: Final[tuple[str, ...]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MODEL_NAME: Final[str] = "meteorite-mass-regressor"
CHAMPION_ALIAS: Final[str] = "champion"
