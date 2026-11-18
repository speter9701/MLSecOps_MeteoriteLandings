import dataclasses
from typing import Any

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from meteorite.config import Settings
from meteorite.preprocessing import clean_data
from meteorite.registry import champion_metrics, should_promote
from meteorite.schemas import CHAMPION_ALIAS, MODEL_NAME
from meteorite.training import split_features, train_and_register

FAST: dict[str, dict[str, Any]] = {"ridge": {"alpha": 1.0}}


@pytest.mark.parametrize(
    "new,old,margin,expected",
    [
        (1.0, None, 0.0, True),
        (0.9, 1.0, 0.0, True),
        (1.0, 1.0, 0.0, True),
        (1.1, 1.0, 0.0, False),
        (0.99, 1.0, 0.05, False),
        (0.90, 1.0, 0.05, True),
    ],
)
def test_should_promote(new: float, old: float | None, margin: float, expected: bool) -> None:
    assert should_promote(new, old, margin) is expected


def test_no_champion_yields_no_metrics(settings: Settings, raw_frame: pd.DataFrame) -> None:
    mlflow.set_tracking_uri(settings.tracking_uri)
    x, y = split_features(clean_data(raw_frame))
    assert champion_metrics(x, y) is None


def test_first_model_becomes_champion(settings: Settings) -> None:
    result = train_and_register(settings, FAST)
    assert result.model_version == "1"
    assert result.promoted and result.champion_metrics is None
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == "1"


def test_equal_challenger_is_promoted(settings: Settings) -> None:
    train_and_register(settings, FAST)
    result = train_and_register(settings, FAST)
    assert result.promoted and result.champion_metrics is not None
    assert str(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version) == "2"


def test_challenger_without_required_gain_keeps_old_champion(settings: Settings) -> None:
    train_and_register(settings, FAST)
    result = train_and_register(dataclasses.replace(settings, min_improvement=0.99), FAST)
    assert result.model_version == "2" and not result.promoted
    assert str(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version) == "1"
    tags = mlflow.get_run(result.run_id).data.tags
    assert tags["promoted"] == "False"
    assert "champion_rmse_log" in mlflow.get_run(result.run_id).data.metrics
