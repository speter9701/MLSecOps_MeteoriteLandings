"""Candidate regressors wrapped into a full training pipeline."""
from collections.abc import Callable, Mapping
from typing import Any, Final

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

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
