"""Shared data contract between the ML core, the REST API and monitoring."""
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

TARGET_COLUMN: Final[str] = "mass"
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("recclass", "nametype", "fall")
NUMERIC_FEATURES: Final[tuple[str, ...]] = ("year", "reclat", "reclong")
FEATURE_COLUMNS: Final[tuple[str, ...]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MODEL_NAME: Final[str] = "meteorite-mass-regressor"
CHAMPION_ALIAS: Final[str] = "champion"


class MeteoriteFeatures(BaseModel):
    """One meteorite record as accepted by the predict endpoint."""

    model_config = ConfigDict(extra="forbid")

    recclass: str = Field(min_length=1, examples=["L6"])
    nametype: Literal["Valid", "Relict"] = "Valid"
    fall: Literal["Fell", "Found"] = "Found"
    year: float | None = Field(default=None, ge=800, le=2100, examples=[1998])
    reclat: float | None = Field(default=None, ge=-90, le=90, examples=[-76.7])
    reclong: float | None = Field(default=None, ge=-180, le=180, examples=[159.4])
