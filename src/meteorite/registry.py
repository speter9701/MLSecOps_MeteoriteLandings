"""Model registry helpers: champion / challenger promotion."""
from typing import Final

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from meteorite.evaluation import regression_metrics
from meteorite.schemas import CHAMPION_ALIAS, MODEL_NAME

CHAMPION_URI: Final[str] = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"


def should_promote(new_rmse: float, champion_rmse: float | None, min_improvement: float = 0.0) -> bool:
    """Promote when there is no champion yet or the challenger is not worse.

    ``min_improvement`` is a relative margin: 0.02 means the challenger needs a 2 % lower RMSE.
    """
    if champion_rmse is None:
        return True
    return new_rmse <= champion_rmse * (1.0 - min_improvement)


def champion_metrics(x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float] | None:
    """Score the current champion on the same holdout set, or None if there is no champion."""
    try:
        model = mlflow.pyfunc.load_model(CHAMPION_URI)
    except MlflowException:
        return None
    return regression_metrics(y_test, model.predict(x_test))
