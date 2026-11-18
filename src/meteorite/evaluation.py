"""Regression metrics computed on the log scale (mass is heavily skewed)."""
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def rmse_log(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE between log1p(true) and log1p(prediction); the model selection metric."""
    true_log = np.log1p(np.asarray(y_true, dtype=float))
    pred_log = np.log1p(np.clip(np.asarray(y_pred, dtype=float), 0, None))
    return float(np.sqrt(mean_squared_error(true_log, pred_log)))


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """All metrics logged to MLflow for one evaluation."""
    true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    true_log, pred_log = np.log1p(true), np.log1p(pred)
    return {
        "rmse_log": rmse_log(true, pred),
        "mae_log": float(mean_absolute_error(true_log, pred_log)),
        "r2_log": float(r2_score(true_log, pred_log)),
        "median_ae_g": float(median_absolute_error(true, pred)),
    }
