import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from meteorite.prediction_log import (
    DEFAULT_PREDICTION_LOG,
    DEFAULT_TRAINING_LOG,
    PREDICTION_COLUMNS,
    TRAINING_COLUMNS,
    PredictionLogger,
    TrainingLogger,
)
from meteorite.schemas import MeteoriteFeatures
from meteorite.training import TrainResult

RECORDS = [
    MeteoriteFeatures(recclass="L6", fall="Fell", year=1998, reclat=-76.7, reclong=159.4),
    MeteoriteFeatures(recclass="H5"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prediction_log_writes_one_row_per_record(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    request_id = PredictionLogger(path).log(RECORDS, [12.5, 0.0], "3")
    rows = read_rows(path)
    assert UUID(request_id).version == 4
    assert [row["request_id"] for row in rows] == [request_id, request_id]
    assert rows[0] | {"timestamp": ""} == {
        "timestamp": "", "request_id": request_id, "recclass": "L6", "nametype": "Valid", "fall": "Fell",
        "year": "1998.0", "reclat": "-76.7", "reclong": "159.4", "prediction": "12.5", "model_version": "3",
    }
    assert rows[1]["year"] == "" and rows[1]["reclat"] == "" and rows[1]["prediction"] == "0.0"


def test_prediction_log_uses_utc_timestamps_and_new_ids(tmp_path: Path) -> None:
    logger = PredictionLogger(tmp_path / "predictions.csv")
    first, second = logger.log(RECORDS[:1], [1.0], "1"), logger.log(RECORDS[:1], [2.0], "1")
    assert first != second
    timestamp = datetime.fromisoformat(read_rows(logger.path)[0]["timestamp"])
    assert timestamp.utcoffset() == timedelta(0) and datetime.now(UTC) - timestamp < timedelta(minutes=1)


def test_prediction_log_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PredictionLogger(tmp_path / "predictions.csv").log(RECORDS, [1.0], "1")


def test_prediction_log_path_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "p.csv"))
    assert PredictionLogger.from_env().path == tmp_path / "p.csv"
    monkeypatch.delenv("PREDICTION_LOG_PATH")
    assert PredictionLogger.from_env().path == DEFAULT_PREDICTION_LOG == Path("logs/predictions.csv")


def test_training_log_writes_the_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAINING_LOG_PATH", str(tmp_path / "training.csv"))
    logger = TrainingLogger.from_env()
    logger.log(TrainResult("run-9", "ridge", "4", {"rmse_log": 1.0}, None, False))
    rows = read_rows(tmp_path / "training.csv")
    assert list(rows[0]) == list(TRAINING_COLUMNS)
    assert rows[0]["run_id"] == "run-9" and rows[0]["model_version"] == "4" and rows[0]["promoted"] == "False"
    monkeypatch.delenv("TRAINING_LOG_PATH")
    assert TrainingLogger.from_env().path == DEFAULT_TRAINING_LOG == Path("logs/training.csv")


def test_prediction_columns_match_the_agreed_format() -> None:
    assert PREDICTION_COLUMNS == (
        "timestamp", "request_id", "recclass", "nametype", "fall", "year", "reclat", "reclong",
        "prediction", "model_version",
    )


def test_header_is_written_only_once(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    logger = PredictionLogger(path)
    logger.log(RECORDS, [1.0, 2.0], "1")
    logger.log(RECORDS[:1], [3.0], "2")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(PREDICTION_COLUMNS)
    assert len(lines) == 4 and sum(line.startswith("timestamp,") for line in lines) == 1


def test_new_logger_appends_to_an_existing_log(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    PredictionLogger(path).log(RECORDS[:1], [1.0], "1")
    PredictionLogger(path).log(RECORDS[:1], [2.0], "1")
    assert [row["prediction"] for row in read_rows(path)] == ["1.0", "2.0"]


def test_header_is_written_to_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.touch()
    PredictionLogger(path).log(RECORDS[:1], [1.0], "1")
    assert list(read_rows(path)[0]) == list(PREDICTION_COLUMNS)


def test_missing_folders_are_created(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "predictions.csv"
    PredictionLogger(path).log(RECORDS[:1], [1.0], "1")
    TrainingLogger(tmp_path / "c" / "training.csv").log(TrainResult("r", "ridge", "1", {}, None, True))
    assert len(read_rows(path)) == 1 and len(read_rows(tmp_path / "c" / "training.csv")) == 1
