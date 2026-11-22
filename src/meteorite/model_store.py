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

    def _load(self) -> tuple[PyFuncModel, str]:
        try:
            version = self._version_loader(self._tracking_uri)
            model = self._loader(self._tracking_uri)
        except MlflowException as exc:
            raise NoChampionError() from exc
        self._model, self._version = model, version
        return model, version

    def get(self) -> tuple[PyFuncModel, str]:
        """The champion model together with its registry version; raises NoChampionError if there is none."""
        with self._lock:
            if self._model is None or self._version is None:
                return self._load()
            return self._model, self._version

    def reload(self) -> str:
        """Load the champion again (after a promotion) and return its version.

        If loading fails, the previously loaded model keeps serving.
        """
        with self._lock:
            return self._load()[1]

    @property
    def model(self) -> PyFuncModel:
        """The current champion model."""
        return self.get()[0]

    @property
    def model_version(self) -> str:
        """Registry version of the current champion model."""
        return self.get()[1]
