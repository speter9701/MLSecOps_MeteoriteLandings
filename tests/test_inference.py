import pytest
from mlflow.exceptions import MlflowException

from meteorite.config import Settings
from meteorite.inference import load_champion


def test_load_champion_without_model_fails(settings: Settings) -> None:
    with pytest.raises(MlflowException):
        load_champion(settings.tracking_uri)
