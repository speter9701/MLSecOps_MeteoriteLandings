from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from meteorite.model import REGRESSORS, build_pipeline, build_regressor
from meteorite.preprocessing import clean_data
from meteorite.schemas import FEATURE_COLUMNS, TARGET_COLUMN

FAST_PARAMS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 10, "random_state": 0},
    "hist_gb": {"max_iter": 20, "random_state": 0},
}


@pytest.mark.parametrize(
    "name,expected",
    [("ridge", Ridge), ("random_forest", RandomForestRegressor), ("hist_gb", HistGradientBoostingRegressor)],
)
def test_build_regressor_types(name: str, expected: type) -> None:
    assert isinstance(build_regressor(name), expected)


def test_build_regressor_passes_params() -> None:
    model = build_regressor("ridge", {"alpha": 2.5})
    assert isinstance(model, Ridge) and model.alpha == 2.5


def test_build_regressor_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_regressor("svm")


def test_fast_params_cover_every_regressor() -> None:
    assert set(FAST_PARAMS) == set(REGRESSORS)


@pytest.mark.parametrize("name", sorted(FAST_PARAMS))
def test_pipeline_fits_and_predicts_grams(name: str, raw_frame: pd.DataFrame) -> None:
    cleaned = clean_data(raw_frame)
    features, target = cleaned[list(FEATURE_COLUMNS)], cleaned[TARGET_COLUMN]
    predictions = np.asarray(build_pipeline(name, FAST_PARAMS[name]).fit(features, target).predict(features))
    assert predictions.shape == (len(cleaned),)
    assert (predictions > -1).all()
