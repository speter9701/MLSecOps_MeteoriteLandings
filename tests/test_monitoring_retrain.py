"""Scheduled retraining sends one POST and surfaces upstream failures."""
import logging
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from meteorite.monitoring import retrain


def test_call_retrain(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fake_open(request: Request, timeout: int) -> BytesIO:
        assert request.full_url == "http://api:8000/train"
        assert request.get_method() == "POST"
        assert timeout == 300
        return BytesIO(b'{"run_id":"abc","model_version":"2","promoted":false}')

    monkeypatch.setattr(retrain, "urlopen", fake_open)
    with caplog.at_level(logging.INFO, logger=retrain.__name__):
        retrain.call_retrain("http://api:8000/")

    assert "run_id=abc model_version=2 promoted=False" in caplog.text


def test_failed_retrain_fails_task(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(request: Request, timeout: int) -> BytesIO:
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

    monkeypatch.setattr(retrain, "urlopen", fail)
    with pytest.raises(HTTPError, match="503"):
        retrain.call_retrain("http://api:8000")
