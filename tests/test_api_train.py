from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from meteorite.api.app import app, get_model_store, get_settings, get_trainer
from meteorite.config import Settings
from meteorite.model_store import ModelStore
from meteorite.training import TrainResult, train_and_register
from tests.test_model_store import FakeRegistry

METRICS = {"rmse_log": 1.2, "mae_log": 0.9, "r2_log": 0.4, "median_ae_g": 50.0}
BASE = Settings(experiment_name="api-test", min_improvement=0.0)


class FakeTrainer:
    """Records the settings it is called with and returns a canned result."""

    def __init__(self, promoted: bool) -> None:
        self.promoted = promoted
        self.calls: list[Settings] = []

    def __call__(self, settings: Settings) -> TrainResult:
        self.calls.append(settings)
        return TrainResult("run-1", "ridge", "2", METRICS, {**METRICS, "rmse_log": 1.3}, self.promoted)


def serve(trainer: object, store: ModelStore, settings: Settings = BASE) -> TestClient:
    app.dependency_overrides[get_trainer] = lambda: trainer
    app.dependency_overrides[get_model_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def store(registry: FakeRegistry) -> ModelStore:
    store = ModelStore(loader=registry.load, version_loader=registry.version)
    store.get()
    return store


def test_train_with_promotion_reloads_the_champion(store: ModelStore, registry: FakeRegistry) -> None:
    trainer = FakeTrainer(promoted=True)
    response = serve(trainer, store).post("/train")
    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-1",
        "best_model": "ridge",
        "model_version": "2",
        "metrics": METRICS,
        "champion_metrics": {**METRICS, "rmse_log": 1.3},
        "promoted": True,
    }
    assert trainer.calls == [BASE]
    assert registry.loads == 2 and store.model_version == "2"


def test_train_end_to_end_promotes_the_first_model(settings: Settings) -> None:
    client = serve(lambda s: train_and_register(s, {"ridge": {"alpha": 1.0}}), ModelStore(settings.tracking_uri),
                   settings)
    body = client.post("/train").json()
    assert body["promoted"] is True and body["champion_metrics"] is None and body["best_model"] == "ridge"
    predicted = client.post("/predict", json={"records": [{"recclass": "L6"}]}).json()
    assert predicted["model_version"] == body["model_version"]


def test_default_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_IMPROVEMENT", "0.1")
    assert get_settings().min_improvement == 0.1
    assert get_trainer() is train_and_register
