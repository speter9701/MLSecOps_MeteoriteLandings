from typing import cast

from mlflow.pyfunc import PyFuncModel

from meteorite.model_store import ModelStore

URI = "sqlite:///fake.db"


class FakeRegistry:
    """Stands in for MLflow: every load returns a new fake model, the version counts the loads."""

    def __init__(self) -> None:
        self.loads = 0
        self.uris: list[str | None] = []

    def version(self, tracking_uri: str | None) -> str:
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
