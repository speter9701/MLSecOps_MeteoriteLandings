"""MLflow tracking of the REST API calls."""
from collections.abc import Mapping
from typing import Any, Final

from mlflow.tracking import MlflowClient

PREDICTION_EXPERIMENT: Final[str] = "meteorite-predictions"


class ApiTracker:
    """Records the API calls in MLflow.

    Uses ``MlflowClient`` instead of the fluent ``mlflow.*`` API: the fluent API keeps a process-wide active
    experiment, which a concurrent ``/train`` call would switch under our feet.
    """

    def __init__(self, tracking_uri: str | None = None, prediction_experiment: str = PREDICTION_EXPERIMENT) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._prediction_experiment = prediction_experiment

    def tag_training_run(self, run_id: str) -> None:
        """Mark a training run as started through the REST API (``source=api``)."""
        self._client.set_tag(run_id, "source", "api")

    def _prediction_experiment_id(self) -> str:
        experiment = self._client.get_experiment_by_name(self._prediction_experiment)
        if experiment is not None:
            return str(experiment.experiment_id)
        return str(self._client.create_experiment(self._prediction_experiment))

    def log_prediction(
        self, request_id: str, request: Mapping[str, Any], response: Mapping[str, Any], model_version: str
    ) -> str:
        """Store one predict call as a run with ``request.json`` and ``response.json``; returns the run id."""
        run = self._client.create_run(
            self._prediction_experiment_id(),
            run_name=f"predict-{request_id[:8]}",
            tags={"source": "api", "request_id": request_id, "model_version": model_version},
        )
        run_id = run.info.run_id
        self._client.log_dict(run_id, dict(request), "request.json")
        self._client.log_dict(run_id, dict(response), "response.json")
        self._client.set_terminated(run_id)
        return run_id
