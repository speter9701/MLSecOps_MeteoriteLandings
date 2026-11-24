"""MLflow tracking of the REST API calls."""
from mlflow.tracking import MlflowClient


class ApiTracker:
    """Records the API calls in MLflow."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def tag_training_run(self, run_id: str) -> None:
        """Mark a training run as started through the REST API (``source=api``)."""
        self._client.set_tag(run_id, "source", "api")
