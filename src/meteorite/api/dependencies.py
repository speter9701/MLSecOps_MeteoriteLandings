"""FastAPI dependencies of the endpoints; tests replace them through ``app.dependency_overrides``."""
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from meteorite.api_tracking import ApiTracker
from meteorite.config import Settings
from meteorite.model_store import ModelStore
from meteorite.prediction_log import PredictionLogger, TrainingLogger
from meteorite.training import TrainResult, train_and_register

Trainer = Callable[[Settings], TrainResult]


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """One model store per process, so the champion is loaded only once."""
    return ModelStore(Settings.from_env().tracking_uri)


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


StoreDep = Annotated[ModelStore, Depends(get_model_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TrainerDep = Annotated[Trainer, Depends(get_trainer)]
PredictionLogDep = Annotated[PredictionLogger, Depends(get_prediction_logger)]
TrainingLogDep = Annotated[TrainingLogger, Depends(get_training_logger)]
TrackerDep = Annotated[ApiTracker, Depends(get_api_tracker)]
