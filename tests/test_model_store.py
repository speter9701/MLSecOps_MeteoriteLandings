from typing import cast

import pytest
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from meteorite.config import Settings
from meteorite.inference import predict
from meteorite.model_store import ModelStore, NoChampionError
from meteorite.schemas import MeteoriteFeatures
from meteorite.training import train_and_register

URI = "sqlite:///fake.db"


class FakeRegistry:
    """Stands in for MLflow: every load returns a new fake model, the version counts the loads."""

    def __init__(self) -> None:
        self.loads = 0
        self.uris: list[str | None] = []
        self.has_champion = True

    def version(self, tracking_uri: str | None) -> str:
        if not self.has_champion:
            raise MlflowException("alias champion not found")
        return str(self.loads + 1)

    def load(self, tracking_uri: str | None) -> PyFuncModel:
        self.loads += 1
        self.uris.append(tracking_uri)
        return cast(PyFuncModel, f"model-{self.loads}")


def make_store(registry: FakeRegistry) -> ModelStore:
    return ModelStore(URI, loader=registry.load, version_loader=registry.version)


def test_store_does_not_load_before_first_use() -> None:
    registry = FakeRegistry()
    make_store(registry)
    assert registry.loads == 0


def test_store_loads_once_and_caches_the_model() -> None:
    registry = FakeRegistry()
    store = make_store(registry)
    model = store.model
    assert store.model is model
    assert registry.loads == 1 and registry.uris == [URI]


def test_store_exposes_the_champion_version() -> None:
    store = make_store(FakeRegistry())
    assert store.model_version == "1"
    assert store.get() == ("model-1", "1")


def test_missing_champion_raises_dedicated_error() -> None:
    registry = FakeRegistry()
    registry.has_champion = False
    store = make_store(registry)
    with pytest.raises(NoChampionError, match="call /train first"):
        store.get()
    assert registry.loads == 0


def test_store_loads_once_the_champion_appears() -> None:
    registry = FakeRegistry()
    registry.has_champion = False
    store = make_store(registry)
    with pytest.raises(NoChampionError):
        store.get()
    registry.has_champion = True
    assert store.model_version == "1"


def test_reload_picks_up_the_new_champion() -> None:
    store = make_store(FakeRegistry())
    store.get()
    assert store.reload() == "2"
    assert store.get() == ("model-2", "2")


def test_failed_reload_keeps_serving_the_old_model() -> None:
    registry = FakeRegistry()
    store = make_store(registry)
    store.get()
    registry.has_champion = False
    with pytest.raises(NoChampionError):
        store.reload()
    assert store.get() == ("model-1", "1")


def test_store_with_the_real_registry(settings: Settings) -> None:
    store = ModelStore(settings.tracking_uri)
    with pytest.raises(NoChampionError):
        store.get()
    result = train_and_register(settings, {"ridge": {"alpha": 1.0}})
    assert store.model_version == result.model_version
    assert len(predict(store.model, [MeteoriteFeatures(recclass="L6")])) == 1
