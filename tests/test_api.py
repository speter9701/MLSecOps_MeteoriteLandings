from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from mlflow.pyfunc import PyFuncModel

import meteorite
from meteorite.api.app import app, get_model_store
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
