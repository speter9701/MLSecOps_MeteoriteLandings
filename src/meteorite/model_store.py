"""Champion model holder used by the REST API."""
from collections.abc import Callable

from mlflow.pyfunc import PyFuncModel

from meteorite.inference import load_champion

ModelLoader = Callable[[str | None], PyFuncModel]


class ModelStore:
    """Holds the champion model served by the API process."""

    def __init__(self, tracking_uri: str | None = None, loader: ModelLoader = load_champion) -> None:
        self._tracking_uri = tracking_uri
        self._loader = loader

    @property
    def model(self) -> PyFuncModel:
        """The current champion model."""
        return self._loader(self._tracking_uri)
