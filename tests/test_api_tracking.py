import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from meteorite.api_tracking import PREDICTION_EXPERIMENT, ApiTracker
from meteorite.config import Settings

REQUEST = {"records": [{"recclass": "L6", "nametype": "Valid", "fall": "Found", "year": None, "reclat": None,
                        "reclong": None}]}
RESPONSE = {"request_id": "0b6f3c1e-8a52", "model_version": "3", "predictions": [12.5]}


def read_artifact(client: MlflowClient, run_id: str, name: str, tmp_path: Path) -> object:
    return json.loads(Path(client.download_artifacts(run_id, name, str(tmp_path))).read_text(encoding="utf-8"))


def test_prediction_run_holds_request_response_and_metric(settings: Settings, tmp_path: Path) -> None:
    run_id = ApiTracker(settings.tracking_uri).log_prediction("0b6f3c1e-8a52", REQUEST, RESPONSE, "3")
    client = MlflowClient(settings.tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == PREDICTION_EXPERIMENT == "meteorite-predictions"
    assert run.info.status == "FINISHED" and run.info.run_name == "predict-0b6f3c1e"
    assert run.data.tags["source"] == "api" and run.data.tags["model_version"] == "3"
    assert run.data.tags["request_id"] == "0b6f3c1e-8a52"
    assert run.data.metrics == {"n_records": 1.0}
    assert sorted(artifact.path for artifact in client.list_artifacts(run_id)) == ["request.json", "response.json"]
    assert read_artifact(client, run_id, "request.json", tmp_path) == REQUEST
    assert read_artifact(client, run_id, "response.json", tmp_path) == RESPONSE


def test_prediction_runs_share_one_experiment(settings: Settings) -> None:
    tracker = ApiTracker(settings.tracking_uri, prediction_experiment="test-predictions")
    first = tracker.log_prediction("a" * 36, REQUEST, RESPONSE, "1")
    second = tracker.log_prediction("b" * 36, REQUEST, {**RESPONSE, "predictions": [1.0, 2.0]}, "1")
    client = MlflowClient(settings.tracking_uri)
    runs = {run_id: client.get_run(run_id) for run_id in (first, second)}
    assert len({run.info.experiment_id for run in runs.values()}) == 1
    assert runs[second].data.metrics["n_records"] == 2.0
