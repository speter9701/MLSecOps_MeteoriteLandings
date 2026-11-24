"""CSV log of the API predictions: the current data of the Evidently drift report."""
import csv
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Final
from uuid import uuid4

from meteorite.schemas import FEATURE_COLUMNS, MeteoriteFeatures

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp", "request_id", *FEATURE_COLUMNS, "prediction", "model_version"
)
DEFAULT_PREDICTION_LOG: Final[Path] = Path("logs/predictions.csv")

# one lock for every writer: concurrent requests must not interleave their lines
_WRITE_LOCK = Lock()


class PredictionLogger:
    """Appends one CSV line per predicted record."""

    def __init__(self, path: Path = DEFAULT_PREDICTION_LOG) -> None:
        self.path = path

    def log(self, records: Sequence[MeteoriteFeatures], predictions: Sequence[float], model_version: str) -> str:
        """Log a predict request and return its request id."""
        request_id = str(uuid4())
        timestamp = datetime.now(UTC).isoformat()
        rows = [
            {"timestamp": timestamp, "request_id": request_id, **record.model_dump(),
             "prediction": prediction, "model_version": model_version}
            for record, prediction in zip(records, predictions, strict=True)
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS).writerows(rows)
        return request_id
