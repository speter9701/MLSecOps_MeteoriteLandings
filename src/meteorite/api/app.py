"""FastAPI application exposing the meteorite mass model."""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

import meteorite
from meteorite.api.schemas import PredictRequest, PredictResponse
from meteorite.config import Settings
from meteorite.inference import predict
from meteorite.model_store import ModelStore, NoChampionError

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


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for docker compose and monitoring."""
    return {"status": "ok"}


@app.post("/predict", responses={503: {"description": "No champion model yet, call /train first"}})
def predict_mass(request: PredictRequest, store: StoreDep) -> PredictResponse:
    """Predict the mass in grams of every record with the champion model."""
    try:
        model = store.model
    except NoChampionError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return PredictResponse(predictions=predict(model, request.records))
