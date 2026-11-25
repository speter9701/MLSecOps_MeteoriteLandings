"""Request and response models of the REST API."""
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from meteorite.schemas import MeteoriteFeatures

# bounds memory and latency of a single request
MAX_RECORDS: Final[int] = 1000


class HealthResponse(BaseModel):
    """Liveness of the API and availability of the champion model."""

    status: Literal["ok"] = "ok"
    champion: bool = Field(description="True if a model with the champion alias is loaded and /predict works.")
    model_version: str | None = Field(default=None, examples=["2"])


class PredictRequest(BaseModel):
    """One or more meteorite records to predict."""

    model_config = ConfigDict(extra="forbid")

    records: list[MeteoriteFeatures] = Field(min_length=1, max_length=MAX_RECORDS)


class PredictResponse(BaseModel):
    """Predicted masses in grams, in the order of the request records."""

    request_id: str = Field(description="Identifies the request in logs/predictions.csv and in MLflow.")
    model_version: str = Field(examples=["2"])
    predictions: list[float]


class TrainRequest(BaseModel):
    """Optional overrides for one training run."""

    model_config = ConfigDict(extra="forbid")

    min_improvement: float | None = Field(
        default=None, ge=0, lt=1, examples=[0.02],
        description="Relative RMSE gain the challenger needs to become champion (default: MIN_IMPROVEMENT env).",
    )


class TrainResponse(BaseModel):
    """Outcome of one training run (mirrors ``meteorite.training.TrainResult``)."""

    run_id: str
    best_model: str = Field(examples=["random_forest"])
    model_version: str = Field(examples=["3"])
    metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    promoted: bool
