"""CSV logs of the API: predictions (current data of the Evidently drift report) and training calls."""
import csv
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Final
from uuid import uuid4

from meteorite.schemas import FEATURE_COLUMNS, MeteoriteFeatures
from meteorite.training import TrainResult

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp", "request_id", *FEATURE_COLUMNS, "prediction", "model_version"
)
DEFAULT_PREDICTION_LOG: Final[Path] = Path("logs/predictions.csv")
TRAINING_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "run_id", "model_version", "promoted")
DEFAULT_TRAINING_LOG: Final[Path] = Path("logs/training.csv")

# one lock for every writer: concurrent requests must not interleave their lines
_WRITE_LOCK = Lock()


def append_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Append rows to a CSV file, creating the folder and writing the header for a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


class PredictionLogger:
    """Appends one CSV line per predicted record."""

    def __init__(self, path: Path = DEFAULT_PREDICTION_LOG) -> None:
        self.path = path

    @classmethod
    def from_env(cls) -> "PredictionLogger":
        """Logger writing to ``PREDICTION_LOG_PATH`` (default ``logs/predictions.csv``)."""
        return cls(Path(os.getenv("PREDICTION_LOG_PATH", str(DEFAULT_PREDICTION_LOG))))

    def log(self, records: Sequence[MeteoriteFeatures], predictions: Sequence[float], model_version: str) -> str:
        """Log a predict request and return its request id."""
        request_id = str(uuid4())
        timestamp = datetime.now(UTC).isoformat()
        rows = [
            {"timestamp": timestamp, "request_id": request_id, **record.model_dump(),
             "prediction": prediction, "model_version": model_version}
            for record, prediction in zip(records, predictions, strict=True)
        ]
        append_csv(self.path, PREDICTION_COLUMNS, rows)
        return request_id


class TrainingLogger:
    """Appends one CSV line per /train call."""

    def __init__(self, path: Path = DEFAULT_TRAINING_LOG) -> None:
        self.path = path

    @classmethod
    def from_env(cls) -> "TrainingLogger":
        """Logger writing to ``TRAINING_LOG_PATH`` (default ``logs/training.csv``)."""
        return cls(Path(os.getenv("TRAINING_LOG_PATH", str(DEFAULT_TRAINING_LOG))))

    def log(self, result: TrainResult) -> None:
        """Log the outcome of one training run."""
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": result.run_id,
            "model_version": result.model_version,
            "promoted": result.promoted,
        }
        append_csv(self.path, TRAINING_COLUMNS, [row])
