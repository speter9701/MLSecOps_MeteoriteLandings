import mlflow
import pandas as pd
import pytest

from meteorite.config import Settings
from meteorite.preprocessing import clean_data
from meteorite.registry import champion_metrics, should_promote
from meteorite.training import split_features


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
