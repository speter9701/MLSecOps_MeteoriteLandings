"""FastAPI application exposing the meteorite mass model."""
import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

import meteorite
from meteorite.api.routes import router
from meteorite.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meteorite mass API",
    version=meteorite.__version__,
    description="Predicts the mass (grams) of a meteorite and retrains the model through MLflow.",
)
app.include_router(router)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Same ``{"detail": ...}`` body as for 4xx errors, without leaking internals to the client."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error"})
