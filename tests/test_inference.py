import pytest
from mlflow.exceptions import MlflowException

from meteorite.config import Settings
from meteorite.inference import load_champion, predict, to_frame
from meteorite.schemas import FEATURE_COLUMNS, MeteoriteFeatures
from meteorite.training import train_and_register

RECORDS = [
    MeteoriteFeatures(recclass="L6", fall="Found", year=1990, reclat=-75.0, reclong=160.0),
    MeteoriteFeatures(recclass="never-seen-class"),
]


def test_to_frame_matches_signature_layout() -> None:
    frame = to_frame(RECORDS)
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame["year"].dtype == float and frame["reclat"].isna().iloc[1]


def test_load_champion_without_model_fails(settings: Settings) -> None:
    with pytest.raises(MlflowException):
        load_champion(settings.tracking_uri)


def test_predict_with_champion(settings: Settings) -> None:
    train_and_register(settings, {"ridge": {"alpha": 1.0}})
    model = load_champion(settings.tracking_uri)
    predictions = predict(model, RECORDS)
    assert len(predictions) == 2 and all(value >= 0 for value in predictions)
    assert predict(model, []) == []
