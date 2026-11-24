"""Shared fixtures: synthetic data shaped like the NASA export (no real data needed in CI)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meteorite.config import Settings


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Request logs of the API go to a temp folder, never to ./logs."""
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "logs" / "predictions.csv"))
    monkeypatch.setenv("TRAINING_LOG_PATH", str(tmp_path / "logs" / "training.csv"))
    return tmp_path / "logs"


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """120 synthetic raw rows with realistic columns and a skewed mass distribution."""
    rng = np.random.default_rng(0)
    n = 120
    lat = rng.uniform(-85, 70, n)
    return pd.DataFrame({
        "name": [f"m{i}" for i in range(n)],
        "id": range(n),
        "nametype": rng.choice(["Valid", "Relict"], n, p=[0.95, 0.05]),
        "recclass": rng.choice(["L6", "H5", "Iron", "LL5"], n),
        "mass (g)": np.exp(rng.normal(5, 2, n) - (lat < -60)),
        "fall": rng.choice(["Fell", "Found"], n),
        "year": rng.integers(1850, 2013, n).astype(float),
        "reclat": lat,
        "reclong": rng.uniform(-180, 180, n),
        "GeoLocation": [f"({a:.2f}, 0.0)" for a in lat],
    })


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_frame: pd.DataFrame) -> Settings:
    """Temp CSV + temp MLflow sqlite store, so tests never touch ./mlruns or the real data."""
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "sample.csv"
    raw_frame.to_csv(data_path, index=False)
    return Settings(
        data_path=data_path,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment_name="test",
        cv_folds=2,
    )
