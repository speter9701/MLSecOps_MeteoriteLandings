"""Request and response models of the REST API."""
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from meteorite.schemas import MeteoriteFeatures

# bounds memory and latency of a single request
MAX_RECORDS: Final[int] = 1000


class PredictRequest(BaseModel):
    """One or more meteorite records to predict."""

    model_config = ConfigDict(extra="forbid")

    records: list[MeteoriteFeatures] = Field(min_length=1, max_length=MAX_RECORDS)


class PredictResponse(BaseModel):
    """Predicted masses in grams, in the order of the request records."""

    model_version: str = Field(examples=["2"])
    predictions: list[float]
