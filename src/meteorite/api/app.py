"""FastAPI application exposing the meteorite mass model."""
from fastapi import FastAPI

import meteorite

app = FastAPI(
    title="Meteorite mass API",
    version=meteorite.__version__,
    description="Predicts the mass (grams) of a meteorite and retrains the model through MLflow.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for docker compose and monitoring."""
    return {"status": "ok"}
