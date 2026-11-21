"""Champion model holder used by the REST API."""
from collections.abc import Callable
from threading import Lock

from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from meteorite.inference import load_champion
from meteorite.schemas import CHAMPION_ALIAS, MODEL_NAME

ModelLoader = Callable[[str | None], PyFuncModel]
VersionLoader = Callable[[str | None], str]


class NoChampionError(RuntimeError):
    """The model registry has no version with the champion alias yet."""

    def __init__(self) -> None:
        super().__init__("No champion model yet, call /train first")


def champion_version(tracking_uri: str | None = None) -> str:
    """Return the registry version that currently holds the champion alias."""
    client = MlflowClient(tracking_uri=tracking_uri)
    return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version)


class ModelStore:
    """Holds the champion model served by the API process, loaded lazily on first use."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        loader: ModelLoader = load_champion,
        version_loader: VersionLoader = champion_version,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._loader = loader
        self._version_loader = version_loader
        self._model: PyFuncModel | None = None
        self._version: str | None = None
        # uvicorn runs sync endpoints in a thread pool: load the model only once
        self._lock = Lock()

    def get(self) -> tuple[PyFuncModel, str]:
        """The champion model together with its registry version; raises NoChampionError if there is none."""
        with self._lock:
            if self._model is None or self._version is None:
                try:
                    self._version = self._version_loader(self._tracking_uri)
                    self._model = self._loader(self._tracking_uri)
                except MlflowException as exc:
                    self._model, self._version = None, None
                    raise NoChampionError() from exc
            return self._model, self._version

    @property
    def model(self) -> PyFuncModel:
        """The current champion model."""
        return self.get()[0]

    @property
    def model_version(self) -> str:
        """Registry version of the current champion model."""
        return self.get()[1]
