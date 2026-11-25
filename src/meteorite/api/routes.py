"""Endpoints of the REST API."""
import logging
import time
from dataclasses import asdict, replace
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from meteorite.api.dependencies import (
    PredictionLogDep,
    SettingsDep,
    StoreDep,
    TrackerDep,
    TrainerDep,
    TrainingLogDep,
)
from meteorite.api.schemas import HealthResponse, PredictRequest, PredictResponse, TrainRequest, TrainResponse
from meteorite.api_tracking import ApiTracker
from meteorite.inference import predict
from meteorite.model_store import NoChampionError

logger = logging.getLogger(__name__)

router = APIRouter()


def track_prediction(
    tracker: ApiTracker, request_id: str, request: dict[str, Any], response: dict[str, Any], model_version: str
) -> None:
    """Write the MLflow run of a predict call; a tracking outage is logged, the CSV log still has the request."""
    try:
        tracker.log_prediction(request_id, request, response, model_version)
    except Exception:  # any MLflow / network error: the response is already sent
        logger.exception("could not log predict request_id=%s to MLflow", request_id)


@router.get("/health")
def health(store: StoreDep) -> HealthResponse:
    """Liveness probe; the API is up even without a champion, ``champion`` tells whether /predict can answer."""
    try:
        _, version = store.get()
    except NoChampionError:
        return HealthResponse(champion=False)
    return HealthResponse(champion=True, model_version=version)


@router.post("/predict", responses={503: {"description": "No champion model yet, call /train first"}})
def predict_mass(
    request: PredictRequest,
    store: StoreDep,
    prediction_log: PredictionLogDep,
    tracker: TrackerDep,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    """Predict the mass in grams of every record with the champion model and log the request."""
    try:
        model, version = store.get()
    except NoChampionError as exc:
        logger.warning("predict rejected: no champion model")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    predictions = predict(model, request.records)
    request_id = prediction_log.log(request.records, predictions, version)
    logger.info("predict request_id=%s records=%d model_version=%s", request_id, len(predictions), version)
    response = PredictResponse(request_id=request_id, model_version=version, predictions=predictions)
    # MLflow round trips run after the response is sent, they do not add to the latency
    background_tasks.add_task(
        track_prediction,
        tracker,
        request_id,
        request.model_dump(mode="json"),
        response.model_dump(mode="json"),
        version,
    )
    return response


@router.post("/train")
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
    logger.info("train started min_improvement=%s", settings.min_improvement)
    started = time.perf_counter()
    result = trainer(settings)
    logger.info(
        "train finished run_id=%s best_model=%s model_version=%s promoted=%s seconds=%.1f",
        result.run_id, result.best_model, result.model_version, result.promoted, time.perf_counter() - started,
    )
    tracker.tag_training_run(result.run_id)
    training_log.log(result)
    if result.promoted:
        store.reload()
    return TrainResponse.model_validate(asdict(result))
