import math

import pytest

from meteorite.evaluation import regression_metrics, rmse_log


def test_rmse_log_is_zero_for_perfect_predictions() -> None:
    assert rmse_log([1.0, 10.0, 100.0], [1.0, 10.0, 100.0]) == 0.0


def test_rmse_log_known_value() -> None:
    expected = math.sqrt(((math.log1p(10.0) - math.log1p(0.0)) ** 2 + 0.0) / 2)
    assert rmse_log([10.0, 5.0], [0.0, 5.0]) == pytest.approx(expected)


def test_negative_predictions_are_clipped_to_zero() -> None:
    assert rmse_log([10.0], [-5.0]) == pytest.approx(rmse_log([10.0], [0.0]))


def test_regression_metrics_keys_and_values() -> None:
    metrics = regression_metrics([1.0, 10.0, 100.0], [1.0, 10.0, 100.0])
    assert set(metrics) == {"rmse_log", "mae_log", "r2_log", "median_ae_g"}
    assert metrics["rmse_log"] == 0.0 and metrics["mae_log"] == 0.0
    assert metrics["r2_log"] == 1.0 and metrics["median_ae_g"] == 0.0
