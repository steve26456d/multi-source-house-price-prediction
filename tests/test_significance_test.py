import numpy as np
import pytest

from src.evaluation.significance_test import compare_absolute_errors, improvement_ratio


def test_improvement_ratio():
    assert improvement_ratio(100.0, 80.0) == pytest.approx(20.0)


def test_improvement_ratio_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="positive"):
        improvement_ratio(0.0, 80.0)


def test_compare_absolute_errors_identical_predictions_are_not_significant():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 210.0, 290.0])

    result = compare_absolute_errors(y_true, y_pred, y_pred, method="wilcoxon")

    assert result.method == "wilcoxon"
    assert result.p_value == pytest.approx(1.0)
    assert result.significant is False


def test_compare_absolute_errors_supports_ttest():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    better = np.array([101.0, 201.0, 299.0, 401.0])
    worse = np.array([120.0, 180.0, 330.0, 360.0])

    result = compare_absolute_errors(y_true, better, worse, method="ttest")

    assert result.method == "ttest"
    assert 0.0 <= result.p_value <= 1.0


def test_compare_absolute_errors_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        compare_absolute_errors(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            method="unknown",
        )
