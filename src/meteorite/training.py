"""End-to-end training: candidate selection with cross-validation and MLflow tracking."""
from collections.abc import Mapping
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score

from meteorite.config import Settings
from meteorite.evaluation import rmse_log
from meteorite.model import build_pipeline
from meteorite.schemas import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

Candidates = Mapping[str, Mapping[str, Any]]


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
