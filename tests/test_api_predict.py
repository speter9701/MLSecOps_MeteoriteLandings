from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from meteorite.api.app import app, get_model_store, get_prediction_logger
from meteorite.api.schemas import MAX_RECORDS
from meteorite.config import Settings
from meteorite.model_store import ModelStore
from meteorite.prediction_log import PREDICTION_COLUMNS
from meteorite.training import train_and_register

RECORD = {"recclass": "L6", "nametype": "Valid", "fall": "Found", "year": 1998, "reclat": -76.7, "reclong": 159.4}


class FakeModel:
    """Predicts row index - 1, so the first prediction is negative and must be clipped to 0."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> list[float]:
        self.frames.append(frame)
        return [float(index) - 1 for index in range(len(frame))]


def serve(store: ModelStore) -> TestClient:
    app.dependency_overrides[get_model_store] = lambda: store
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client(fake_model: FakeModel) -> TestClient:
    return serve(ModelStore(loader=lambda _: fake_model, version_loader=lambda _: "7"))


def test_predict_returns_one_value_per_record(client: TestClient, fake_model: FakeModel) -> None:
    response = client.post("/predict", json={"records": [RECORD, {"recclass": "H5"}, {"recclass": "Iron"}]})
    assert response.status_code == 200
    body = response.json()
    assert body.pop("request_id")
    assert body == {"model_version": "7", "predictions": [0.0, 0.0, 1.0]}
    assert list(fake_model.frames[0]["recclass"]) == ["L6", "H5", "Iron"]


def test_predict_applies_schema_defaults(client: TestClient, fake_model: FakeModel) -> None:
    client.post("/predict", json={"records": [{"recclass": "H5"}]})
    row = fake_model.frames[0].iloc[0]
    assert row["nametype"] == "Valid" and row["fall"] == "Found" and pd.isna(row["year"])


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"records": []},
        {"records": [RECORD] * (MAX_RECORDS + 1)},
        {"records": [RECORD], "model": "other"},
        {"records": [{"nametype": "Valid"}]},
        {"records": [{"recclass": ""}]},
        {"records": [{**RECORD, "reclat": 91}]},
        {"records": [{**RECORD, "fall": "Unknown"}]},
        {"records": [{**RECORD, "mass": 10.0}]},
    ],
    ids=["no-records", "empty", "too-many", "extra-top-level", "no-recclass", "blank-recclass",
         "lat-out-of-range", "bad-fall", "unknown-field"],
)
def test_predict_rejects_invalid_input(client: TestClient, fake_model: FakeModel, body: dict[str, object]) -> None:
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert fake_model.frames == []


def test_predict_accepts_the_maximum_batch(client: TestClient) -> None:
    response = client.post("/predict", json={"records": [RECORD] * MAX_RECORDS})
    assert response.status_code == 200 and len(response.json()["predictions"]) == MAX_RECORDS


def test_predict_without_champion_returns_503(fake_model: FakeModel) -> None:
    def no_alias(_: str | None) -> str:
        raise MlflowException("alias champion not found")

    client = serve(ModelStore(loader=lambda _: fake_model, version_loader=no_alias))
    response = client.post("/predict", json={"records": [RECORD]})
    assert response.status_code == 503
    assert response.json() == {"detail": "No champion model yet, call /train first"}
    assert fake_model.frames == []


def test_predict_end_to_end_with_a_trained_ridge_model(settings: Settings) -> None:
    client = serve(ModelStore(settings.tracking_uri))
    assert client.post("/predict", json={"records": [RECORD]}).status_code == 503

    result = train_and_register(settings, {"ridge": {"alpha": 1.0}})
    response = client.post("/predict", json={"records": [RECORD, {"recclass": "never-seen-class"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == result.model_version
    assert len(body["predictions"]) == 2 and all(value >= 0 for value in body["predictions"])


def test_default_store_is_created_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    get_model_store.cache_clear()
    store = get_model_store()
    assert isinstance(store, ModelStore) and get_model_store() is store
    get_model_store.cache_clear()


def test_predict_logs_every_record(client: TestClient, isolated_logs: Path) -> None:
    first = client.post("/predict", json={"records": [RECORD, {"recclass": "H5"}]}).json()
    second = client.post("/predict", json={"records": [{"recclass": "Iron"}]}).json()
    log = pd.read_csv(isolated_logs / "predictions.csv")
    assert list(log.columns) == list(PREDICTION_COLUMNS)
    assert list(log["request_id"]) == [first["request_id"]] * 2 + [second["request_id"]]
    assert list(log["recclass"]) == ["L6", "H5", "Iron"]
    assert list(log["prediction"]) == first["predictions"] + second["predictions"]
    assert set(log["model_version"]) == {7}


def test_rejected_requests_are_not_logged(fake_model: FakeModel, isolated_logs: Path) -> None:
    def no_alias(_: str | None) -> str:
        raise MlflowException("alias champion not found")

    client = serve(ModelStore(loader=lambda _: fake_model, version_loader=no_alias))
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [RECORD]}).status_code == 503
    assert not (isolated_logs / "predictions.csv").exists()


def test_prediction_logger_follows_the_environment(isolated_logs: Path) -> None:
    assert get_prediction_logger().path == isolated_logs / "predictions.csv"
