"""Champion model loading and prediction, used by the REST API."""
import mlflow
from mlflow.pyfunc import PyFuncModel

from meteorite.registry import CHAMPION_URI


def load_champion(tracking_uri: str | None = None) -> PyFuncModel:
    """Load the model version that currently holds the champion alias."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(CHAMPION_URI)
