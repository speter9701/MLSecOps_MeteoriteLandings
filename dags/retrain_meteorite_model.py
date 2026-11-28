"""Run the meteorite API's training endpoint daily."""
import os
from datetime import UTC, datetime

from airflow.sdk import dag, task

from meteorite.monitoring.retrain import call_retrain


@dag(
    dag_id="retrain_meteorite_model",
    description="Compare a newly trained meteorite model with the current champion.",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
)
def retrain_meteorite_model() -> None:
    """Schedule one HTTP task per run."""

    @task(retries=0)
    def train_challenger() -> None:
        """Call the FastAPI service inside the Compose network."""
        call_retrain(os.environ["METEORITE_API_URL"])

    train_challenger()


retrain_meteorite_model()
