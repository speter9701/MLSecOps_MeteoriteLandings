from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

import meteorite
from meteorite.api.app import app
from meteorite.api.dependencies import get_model_store
from meteorite.model_store import ModelStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def champion_store() -> Iterator[None]:
    store = ModelStore(loader=lambda _: cast(PyFuncModel, object()), version_loader=lambda _: "5")
    app.dependency_overrides[get_model_store] = lambda: store
    yield
    app.dependency_overrides.clear()


def test_app_metadata() -> None:
    assert app.title == "Meteorite mass API"
    assert app.version == meteorite.__version__


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_lists_health() -> None:
    assert "/health" in client.get("/openapi.json").json()["paths"]


def test_health_reports_the_champion_version() -> None:
    assert client.get("/health").json() == {"status": "ok", "champion": True, "model_version": "5"}


def test_health_is_up_without_champion() -> None:
    def no_alias(_: str | None) -> str:
        raise MlflowException("alias champion not found")

    store = ModelStore(loader=lambda _: cast(PyFuncModel, object()), version_loader=no_alias)
    app.dependency_overrides[get_model_store] = lambda: store
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "champion": False, "model_version": None}
