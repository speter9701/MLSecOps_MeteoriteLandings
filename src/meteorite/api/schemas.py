"""Request and response models of the REST API."""
from pydantic import BaseModel, ConfigDict, Field

from meteorite.schemas import MeteoriteFeatures


class PredictRequest(BaseModel):
    """One or more meteorite records to predict."""

    model_config = ConfigDict(extra="forbid")

    records: list[MeteoriteFeatures] = Field(min_length=1)


class PredictResponse(BaseModel):
    """Predicted masses in grams, in the order of the request records."""

    predictions: list[float]
