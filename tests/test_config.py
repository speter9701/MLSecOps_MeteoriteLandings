from pathlib import Path

import pytest

from meteorite.config import Settings

ENV_VARS = ("DATA_PATH", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MIN_IMPROVEMENT")


def test_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert Settings.from_env() == Settings()


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PATH", "x.csv")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp")
    monkeypatch.setenv("MIN_IMPROVEMENT", "0.05")
    settings = Settings.from_env()
    assert settings.data_path == Path("x.csv")
    assert settings.tracking_uri == "http://mlflow:5000"
    assert settings.experiment_name == "exp"
    assert settings.min_improvement == 0.05


def test_settings_are_immutable() -> None:
    with pytest.raises(AttributeError):
        Settings().cv_folds = 3  # type: ignore[misc]
