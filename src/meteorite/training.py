"""End-to-end training: candidate selection with cross-validation and MLflow tracking."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from meteorite.config import Settings
from meteorite.data import load_raw
from meteorite.evaluation import regression_metrics, rmse_log
from meteorite.model import build_pipeline
from meteorite.preprocessing import clean_data
from meteorite.schemas import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

Candidates = Mapping[str, Mapping[str, Any]]

DEFAULT_CANDIDATES: Final[dict[str, dict[str, Any]]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 3, "random_state": 42, "n_jobs": -1},
    "hist_gb": {"max_iter": 300, "learning_rate": 0.05, "random_state": 42},
}

# MLflow stores sklearn models with skops (no arbitrary pickle code execution on load);
# our own feature function has to be allow-listed explicitly.
SKOPS_TRUSTED_TYPES: Final[list[str]] = [
    "meteorite.preprocessing.add_geo_features",
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


@dataclass(frozen=True)
class TrainResult:
    """Summary of one training run, returned by the /train endpoint."""

    run_id: str
    best_model: str
    metrics: dict[str, float]


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model inputs (numeric columns as float) and the target."""
    features = pd.DataFrame(df[list(FEATURE_COLUMNS)]).copy()
    features[list(NUMERIC_FEATURES)] = features[list(NUMERIC_FEATURES)].astype(float)
    return features, pd.Series(df[TARGET_COLUMN], dtype=float)


def select_best_candidate(x: pd.DataFrame, y: pd.Series, candidates: Candidates, settings: Settings) -> str:
    """Cross-validate each candidate in a nested MLflow run and return the best name."""
    scorer = make_scorer(rmse_log, greater_is_better=False)
    folds = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    scores: dict[str, float] = {}
    for name, params in candidates.items():
        with mlflow.start_run(run_name=f"cv-{name}", nested=True):
            mlflow.log_params({"model": name, **params})
            score = -float(cross_val_score(build_pipeline(name, params), x, y, scoring=scorer, cv=folds).mean())
            mlflow.log_metric("cv_rmse_log", score)
            scores[name] = score
    return min(scores, key=scores.__getitem__)


def train(settings: Settings | None = None, candidates: Candidates | None = None) -> TrainResult:
    """Pick the best candidate, refit it, evaluate on the holdout set and log everything to MLflow."""
    settings = settings or Settings.from_env()
    candidates = candidates or DEFAULT_CANDIDATES
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = clean_data(load_raw(settings.data_path))
    x, y = split_features(df)
    parts = train_test_split(x, y, test_size=settings.test_size, random_state=settings.random_state)
    x_train, x_test = pd.DataFrame(parts[0]), pd.DataFrame(parts[1])
    y_train, y_test = pd.Series(parts[2], dtype=float), pd.Series(parts[3], dtype=float)

    with mlflow.start_run(run_name="train") as run:
        dataset = from_pandas(df, source=str(settings.data_path), name="meteorites")
        mlflow.log_input(dataset, context="training")
        mlflow.log_params({
            "rows": len(df),
            "test_size": settings.test_size,
            "cv_folds": settings.cv_folds,
            "candidates": ",".join(candidates),
        })
        best = select_best_candidate(x_train, y_train, candidates, settings)
        model = build_pipeline(best, candidates[best]).fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = regression_metrics(y_test, predictions)
        mlflow.log_metrics(metrics)
        log_model(
            model,
            name="model",
            signature=infer_signature(x_test, predictions),
            input_example=x_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        mlflow.set_tag("best_model", best)
    return TrainResult(run.info.run_id, best, metrics)


def main() -> int:
    """CLI entry point used by the MLproject file and Airflow."""
    print(train())
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper, main() is tested directly
    raise SystemExit(main())
