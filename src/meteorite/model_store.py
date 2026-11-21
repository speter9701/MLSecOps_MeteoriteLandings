"""Champion model holder used by the REST API."""
from collections.abc import Callable
from threading import Lock

from mlflow.pyfunc import PyFuncModel

from meteorite.inference import load_champion

ModelLoader = Callable[[str | None], PyFuncModel]


class ModelStore:
    """Holds the champion model served by the API process, loaded lazily on first use."""

    def __init__(self, tracking_uri: str | None = None, loader: ModelLoader = load_champion) -> None:
        self._tracking_uri = tracking_uri
        self._loader = loader
        self._model: PyFuncModel | None = None
        # uvicorn runs sync endpoints in a thread pool: load the model only once
        self._lock = Lock()

    @property
    def model(self) -> PyFuncModel:
        """The current champion model."""
        with self._lock:
            if self._model is None:
                self._model = self._loader(self._tracking_uri)
            return self._model
