"""FastAPI application exposing the meteorite mass model."""
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

import meteorite
from meteorite.api.schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from meteorite.api_tracking import ApiTracker
from meteorite.config import Settings
from meteorite.inference import predict
from meteorite.model_store import ModelStore, NoChampionError
from meteorite.prediction_log import PredictionLogger, TrainingLogger
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


def get_prediction_logger() -> PredictionLogger:
    """Prediction log at ``PREDICTION_LOG_PATH``, read from the environment on every call."""
    return PredictionLogger.from_env()


def get_training_logger() -> TrainingLogger:
    """Training log at ``TRAINING_LOG_PATH``, read from the environment on every call."""
    return TrainingLogger.from_env()


def get_api_tracker() -> ApiTracker:
    """MLflow tracker of the API calls, on the tracking server of ``MLFLOW_TRACKING_URI``."""
    return ApiTracker(Settings.from_env().tracking_uri)


SettingsDep = Annotated[Settings, Depends(get_settings)]
TrainerDep = Annotated[Trainer, Depends(get_trainer)]
PredictionLogDep = Annotated[PredictionLogger, Depends(get_prediction_logger)]
TrainingLogDep = Annotated[TrainingLogger, Depends(get_training_logger)]
TrackerDep = Annotated[ApiTracker, Depends(get_api_tracker)]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for docker compose and monitoring."""
    return {"status": "ok"}


@app.post("/predict", responses={503: {"description": "No champion model yet, call /train first"}})
def predict_mass(request: PredictRequest, store: StoreDep, prediction_log: PredictionLogDep) -> PredictResponse:
    """Predict the mass in grams of every record with the champion model and log the request."""
    try:
        model, version = store.get()
    except NoChampionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    predictions = predict(model, request.records)
    request_id = prediction_log.log(request.records, predictions, version)
    return PredictResponse(request_id=request_id, model_version=version, predictions=predictions)


@app.post("/train")
def train(
    settings: SettingsDep,
    trainer: TrainerDep,
    store: StoreDep,
    training_log: TrainingLogDep,
    tracker: TrackerDep,
    request: TrainRequest | None = None,
) -> TrainResponse:
    """Retrain and register a new model version; it becomes champion only if it is not worse (takes ~45 s)."""
    if request is not None and request.min_improvement is not None:
        settings = replace(settings, min_improvement=request.min_improvement)
    result = trainer(settings)
    tracker.tag_training_run(result.run_id)
    training_log.log(result)
    if result.promoted:
        store.reload()
    return TrainResponse.model_validate(asdict(result))
