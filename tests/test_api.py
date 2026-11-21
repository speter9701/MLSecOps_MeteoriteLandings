from fastapi.testclient import TestClient

import meteorite
from meteorite.api.app import app

client = TestClient(app)


def test_app_metadata() -> None:
    assert app.title == "Meteorite mass API"
    assert app.version == meteorite.__version__


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_lists_health() -> None:
    assert "/health" in client.get("/openapi.json").json()["paths"]
