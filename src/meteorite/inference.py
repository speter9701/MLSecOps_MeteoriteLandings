"""Champion model loading and prediction, used by the REST API."""
from collections.abc import Sequence

import mlflow
import pandas as pd
from mlflow.pyfunc import PyFuncModel

from meteorite.registry import CHAMPION_URI
from meteorite.schemas import FEATURE_COLUMNS, NUMERIC_FEATURES, MeteoriteFeatures


def load_champion(tracking_uri: str | None = None) -> PyFuncModel:
    """Load the model version that currently holds the champion alias."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(CHAMPION_URI)


def to_frame(records: Sequence[MeteoriteFeatures]) -> pd.DataFrame:
    """Convert validated records to the column layout of the model signature."""
    frame = pd.DataFrame([record.model_dump() for record in records], columns=list(FEATURE_COLUMNS))
    frame[list(NUMERIC_FEATURES)] = frame[list(NUMERIC_FEATURES)].astype(float)
    return frame


def predict(model: PyFuncModel, records: Sequence[MeteoriteFeatures]) -> list[float]:
    """Predict the meteorite mass in grams for each record."""
    if not records:
        return []
    return [max(float(value), 0.0) for value in model.predict(to_frame(records))]
