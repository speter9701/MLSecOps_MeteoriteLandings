import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from meteorite.api.app import app
from meteorite.api.dependencies import get_api_tracker, get_model_store, get_settings, get_trainer
from meteorite.config import Settings
from meteorite.model_store import ModelStore
from meteorite.training import TrainResult
from tests.conftest import FakeTracker


class ConstantModel:
    def predict(self, frame: pd.DataFrame) -> list[float]:
        return [10.0] * len(frame)


class BrokenTracker(FakeTracker):
    def log_prediction(self, request_id: str, request: dict[str, Any], response: dict[str, Any],
                       model_version: str) -> str:
        raise MlflowException("tracking server unreachable")


def failing_trainer(settings: Settings) -> TrainResult:
    raise RuntimeError("secret internal detail")


def no_alias(tracking_uri: str | None) -> str:
    raise MlflowException("alias champion not found")


def store_with_champion() -> ModelStore:
    return ModelStore(loader=lambda _: cast(PyFuncModel, ConstantModel()), version_loader=lambda _: "4")


@pytest.fixture(autouse=True)
def overrides(tracker: FakeTracker) -> Iterator[None]:
    app.dependency_overrides[get_api_tracker] = lambda: tracker
    app.dependency_overrides[get_model_store] = store_with_champion
    app.dependency_overrides[get_settings] = Settings
    yield
    app.dependency_overrides.clear()


def test_unexpected_errors_return_a_json_detail(caplog: pytest.LogCaptureFixture) -> None:
    app.dependency_overrides[get_trainer] = lambda: failing_trainer
    with caplog.at_level(logging.ERROR, logger="meteorite"):
        response = TestClient(app, raise_server_exceptions=False).post("/train")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert "unhandled error on POST /train" in caplog.text and "secret internal detail" in caplog.text


def test_predict_survives_a_tracking_outage(isolated_logs: Path, caplog: pytest.LogCaptureFixture) -> None:
    app.dependency_overrides[get_api_tracker] = BrokenTracker
    with caplog.at_level(logging.INFO, logger="meteorite"):
        response = TestClient(app).post("/predict", json={"records": [{"recclass": "L6"}]})
    assert response.status_code == 200 and response.json()["predictions"] == [10.0]
    assert len(pd.read_csv(isolated_logs / "predictions.csv")) == 1
    assert f"could not log predict request_id={response.json()['request_id']} to MLflow" in caplog.text


def test_predict_and_rejection_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="meteorite"):
        body = client.post("/predict", json={"records": [{"recclass": "L6"}, {"recclass": "H5"}]}).json()
        app.dependency_overrides[get_model_store] = lambda: ModelStore(version_loader=no_alias)
        assert client.post("/predict", json={"records": [{"recclass": "L6"}]}).status_code == 503
    assert f"predict request_id={body['request_id']} records=2 model_version=4" in caplog.text
    assert ("meteorite.api.routes", logging.WARNING, "predict rejected: no champion model") in caplog.record_tuples


def test_train_start_and_result_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    result = TrainResult("run-7", "ridge", "4", {"rmse_log": 1.0}, None, False)
    app.dependency_overrides[get_trainer] = lambda: lambda settings: result
    with caplog.at_level(logging.INFO, logger="meteorite"):
        assert TestClient(app).post("/train").status_code == 200
    assert "train started min_improvement=0.0" in caplog.text
    assert "train finished run_id=run-7 best_model=ridge model_version=4 promoted=False" in caplog.text
