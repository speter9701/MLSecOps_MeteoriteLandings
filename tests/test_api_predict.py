from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from meteorite.api.app import app, get_model_store
from meteorite.model_store import ModelStore

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
    assert response.json() == {"model_version": "7", "predictions": [0.0, 0.0, 1.0]}
    assert list(fake_model.frames[0]["recclass"]) == ["L6", "H5", "Iron"]


def test_predict_applies_schema_defaults(client: TestClient, fake_model: FakeModel) -> None:
    client.post("/predict", json={"records": [{"recclass": "H5"}]})
    row = fake_model.frames[0].iloc[0]
    assert row["nametype"] == "Valid" and row["fall"] == "Found" and pd.isna(row["year"])


def test_default_store_is_created_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    get_model_store.cache_clear()
    store = get_model_store()
    assert isinstance(store, ModelStore) and get_model_store() is store
    get_model_store.cache_clear()
