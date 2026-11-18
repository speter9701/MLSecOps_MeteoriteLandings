"""Candidate regressors wrapped into a full training pipeline."""
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from meteorite.preprocessing import build_preprocessor

REGRESSORS: Final[dict[str, Callable[..., RegressorMixin]]] = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "hist_gb": HistGradientBoostingRegressor,
}


def build_regressor(name: str, params: Mapping[str, Any] | None = None) -> RegressorMixin:
    """Instantiate a candidate regressor by name."""
    if name not in REGRESSORS:
        raise ValueError(f"Unknown model '{name}', choose from {sorted(REGRESSORS)}")
    return REGRESSORS[name](**dict(params or {}))


def build_pipeline(name: str, params: Mapping[str, Any] | None = None) -> TransformedTargetRegressor:
    """Preprocessing + regressor, trained on log1p(mass) and predicting grams."""
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", build_regressor(name, params))])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
