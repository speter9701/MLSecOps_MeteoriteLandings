from typing import Any

import mlflow
import pandas as pd
import pytest
import skops.io as sio

from meteorite import training
from meteorite.config import Settings
from meteorite.model import build_pipeline
from meteorite.preprocessing import clean_data

FAST: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 10, "random_state": 0},
}


def test_split_features_casts_numeric(raw_frame: pd.DataFrame) -> None:
    x, y = training.split_features(clean_data(raw_frame))
    assert list(x.columns) == ["recclass", "nametype", "fall", "year", "reclat", "reclong"]
    assert x["year"].dtype == float and y.dtype == float and len(x) == len(y)


def test_train_logs_parent_and_nested_runs(settings: Settings) -> None:
    result = training.train(settings, FAST)
    assert result.best_model in FAST
    assert set(result.metrics) == {"rmse_log", "mae_log", "r2_log", "median_ae_g"}
    runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert len(runs) == 1 + len(FAST)
    parent = mlflow.get_run(result.run_id)
    assert parent.data.tags["best_model"] == result.best_model
    assert parent.data.params["candidates"] == "ridge,random_forest"
    assert "rmse_log" in parent.data.metrics


def test_logged_model_can_be_loaded(settings: Settings, raw_frame: pd.DataFrame) -> None:
    result = training.train(settings, FAST)
    model = mlflow.pyfunc.load_model(f"runs:/{result.run_id}/model")
    x, _ = training.split_features(clean_data(raw_frame))
    assert len(model.predict(x.head(5))) == 5


def test_main_reads_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PATH", str(settings.data_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", settings.tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setattr(training, "DEFAULT_CANDIDATES", {"ridge": {"alpha": 1.0}})
    assert training.main() == 0


@pytest.mark.parametrize("name", sorted(training.DEFAULT_CANDIDATES))
def test_every_candidate_only_needs_allow_listed_types(name: str, raw_frame: pd.DataFrame) -> None:
    x, y = training.split_features(clean_data(raw_frame))
    small = {"n_estimators": 3} if name == "random_forest" else {"max_iter": 3} if name == "hist_gb" else {}
    model = build_pipeline(name, small).fit(x, y)
    assert set(sio.get_untrusted_types(data=sio.dumps(model))) <= set(training.SKOPS_TRUSTED_TYPES)
