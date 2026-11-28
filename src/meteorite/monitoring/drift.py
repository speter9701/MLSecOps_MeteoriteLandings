"""Compare incoming model features with the versioned training sample."""
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.legacy.tests.base_test import TestStatus
from evidently.presets import DataDriftPreset

from meteorite.data import load_raw
from meteorite.preprocessing import clean_data
from meteorite.schemas import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class DriftSummary(TypedDict):
    """Columns flagged by Evidently and columns with no comparable values."""

    count: int
    columns: list[str]
    skipped: list[str]


@dataclass(frozen=True)
class DriftReport:
    """Evidently snapshot and metadata derived from the same run."""

    snapshot: Snapshot
    summary: DriftSummary

    def html(self) -> str:
        """Return the self-contained Evidently report."""
        return str(self.snapshot.get_html_str(as_iframe=False))


def load_reference(path: Path) -> pd.DataFrame:
    """Read the cleaned feature columns from the DVC training sample."""
    return clean_data(load_raw(path)).loc[:, list(FEATURE_COLUMNS)]


def load_prediction_log(path: Path) -> pd.DataFrame | None:
    """Read the API log, treating a missing, empty, or header-only file alike."""
    try:
        predictions = pd.read_csv(path, dtype={name: str for name in CATEGORICAL_FEATURES})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    return None if predictions.empty else predictions


def load_current(log_path: Path) -> pd.DataFrame | None:
    """Read the feature columns from the current predictions."""
    predictions = load_prediction_log(log_path)
    return None if predictions is None else predictions.loc[:, list(FEATURE_COLUMNS)]


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
    """Run the Evidently preset once and summarize its per-column outcomes."""
    reference_columns = reference.dropna(axis="columns", how="all").columns
    current_columns = current.dropna(axis="columns", how="all").columns
    columns = [name for name in FEATURE_COLUMNS if name in reference_columns and name in current_columns]
    if not columns:
        raise ValueError("No feature has values in both the reference sample and the prediction log.")

    definition = DataDefinition(
        numerical_columns=[name for name in NUMERIC_FEATURES if name in columns],
        categorical_columns=[name for name in CATEGORICAL_FEATURES if name in columns],
    )
    snapshot = Report([DataDriftPreset(columns=columns)], include_tests=True).run(
        current_data=Dataset.from_pandas(current.loc[:, columns], data_definition=definition),
        reference_data=Dataset.from_pandas(reference.loc[:, columns], data_definition=definition),
    )
    drifted = [
        str(test.metric_config.params["column"])
        for test in snapshot.tests_results
        if "column" in test.metric_config.params and test.status == TestStatus.FAIL
    ]
    return DriftReport(snapshot, {
        "count": len(drifted),
        "columns": drifted,
        "skipped": [name for name in FEATURE_COLUMNS if name not in columns],
    })


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    """Return a full Evidently report as HTML."""
    return evaluate_drift(reference, current).html()


def drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> DriftSummary:
    """Get drifted feature names and count without serializing HTML."""
    return evaluate_drift(reference, current).summary
