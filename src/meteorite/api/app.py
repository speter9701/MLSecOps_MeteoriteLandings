"""FastAPI application exposing the meteorite mass model."""
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

import meteorite
from meteorite.api.schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from meteorite.config import Settings
from meteorite.inference import predict
from meteorite.model_store import ModelStore, NoChampionError
from meteorite.training import TrainResult, train_and_register

Trainer = Callable[[Settings], TrainResult]

app = FastAPI(
    title="Meteorite mass API",
    version=meteorite.__version__,
    description="Predicts the mass (grams) of a meteorite and retrains the model through MLflow.",
)


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """One model store per process; tests replace it through ``app.dependency_overrides``."""
    return ModelStore(Settings.from_env().tracking_uri)


StoreDep = Annotated[ModelStore, Depends(get_model_store)]


def get_settings() -> Settings:
    """Training settings, read from the environment on every call."""
    return Settings.from_env()


def get_trainer() -> Trainer:
    """The training function; tests replace it with a fast fake."""
    return train_and_register


SettingsDep = Annotated[Settings, Depends(get_settings)]
TrainerDep = Annotated[Trainer, Depends(get_trainer)]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for docker compose and monitoring."""
    return {"status": "ok"}


@app.post("/predict", responses={503: {"description": "No champion model yet, call /train first"}})
def predict_mass(request: PredictRequest, store: StoreDep) -> PredictResponse:
    """Predict the mass in grams of every record with the champion model."""
    try:
        model, version = store.get()
    except NoChampionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return PredictResponse(model_version=version, predictions=predict(model, request.records))


@app.post("/train")
def train(
    settings: SettingsDep, trainer: TrainerDep, store: StoreDep, request: TrainRequest | None = None
) -> TrainResponse:
    """Retrain and register a new model version; it becomes champion only if it is not worse (takes ~45 s)."""
    if request is not None and request.min_improvement is not None:
        settings = replace(settings, min_improvement=request.min_improvement)
    result = trainer(settings)
    if result.promoted:
        store.reload()
    return TrainResponse.model_validate(asdict(result))
