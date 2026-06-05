import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_safe,
    error_summary,
    mae,
    mape,
    r2_score,
    rmse,
)


def test_compute_metrics_matches_expected_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(math.sqrt((100 + 100 + 900) / 3))
    assert metrics["mae"] == pytest.approx((10 + 10 + 30) / 3)
    assert metrics["r2"] == pytest.approx(0.945)
    assert metrics["mape"] == pytest.approx(((0.1 + 0.05 + 0.1) / 3) * 100)


def test_individual_metric_functions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(1 / 3))
    assert mae(y_true, y_pred) == pytest.approx(1 / 3)
    assert r2_score(y_true, y_pred) == pytest.approx(0.5)
    assert mape(y_true, y_pred) == pytest.approx((0 + 0 + 1 / 3) / 3 * 100)


def test_metrics_reject_invalid_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="one-dimensional"):
        compute_metrics(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))


def test_metrics_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        compute_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_compute_metrics_safe_filters_non_finite_pairs():
    metrics = compute_metrics_safe(
        np.array([100.0, np.nan, 300.0]),
        np.array([110.0, 200.0, 330.0]),
    )

    assert metrics["mae"] == pytest.approx(20.0)


def test_constant_target_r2_behavior():
    y_true = np.array([5.0, 5.0, 5.0])

    assert r2_score(y_true, np.array([5.0, 5.0, 5.0])) == pytest.approx(1.0)
    assert r2_score(y_true, np.array([4.0, 5.0, 6.0])) == pytest.approx(0.0)


def test_error_summary():
    summary = error_summary(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 35.0]))

    assert summary["mean_error"] == pytest.approx((2 - 2 + 5) / 3)
    assert summary["median_error"] == pytest.approx(2.0)
    assert summary["mean_abs_error"] == pytest.approx((2 + 2 + 5) / 3)
    assert summary["median_abs_error"] == pytest.approx(2.0)
    assert summary["max_abs_error"] == pytest.approx(5.0)
