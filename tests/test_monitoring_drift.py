"""Drift reports use real Evidently calculations on synthetic input."""
from pathlib import Path

import pandas as pd
import pytest

from meteorite.monitoring.drift import build_drift_report, drift_summary, load_current, load_reference
from meteorite.schemas import FEATURE_COLUMNS


@pytest.fixture
def reference(tmp_path: Path, raw_frame: pd.DataFrame) -> pd.DataFrame:
    path = tmp_path / "reference.csv"
    raw_frame.to_csv(path, index=False)
    return load_reference(path)


@pytest.mark.parametrize("contents", [None, "", "timestamp,recclass\n"])
def test_load_current_without_predictions(tmp_path: Path, contents: str | None) -> None:
    path = tmp_path / "predictions.csv"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert load_current(path) is None


def test_load_current_keeps_only_features(tmp_path: Path, reference: pd.DataFrame) -> None:
    path = tmp_path / "predictions.csv"
    reference.assign(timestamp="2026-01-01T00:00:00Z", prediction=1.0).to_csv(path, index=False)

    current = load_current(path)
    assert current is not None
    assert list(current.columns) == list(FEATURE_COLUMNS)
    assert len(current) == len(reference)


def test_identical_distributions_do_not_drift(reference: pd.DataFrame) -> None:
    assert drift_summary(reference, reference.copy()) == {"count": 0, "columns": [], "skipped": []}


def test_shifted_distributions_and_missing_optional_fields(reference: pd.DataFrame) -> None:
    current = reference.assign(reclat=90.0, recclass="Unknown", year=float("nan"))
    summary = drift_summary(reference, current)

    assert summary["count"] == 2
    assert set(summary["columns"]) == {"recclass", "reclat"}
    assert summary["skipped"] == ["year"]


def test_report_html(reference: pd.DataFrame) -> None:
    assert "<html" in build_drift_report(reference, reference).lower()


def test_no_comparable_columns(reference: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No feature has values"):
        drift_summary(reference, reference.iloc[:0])
