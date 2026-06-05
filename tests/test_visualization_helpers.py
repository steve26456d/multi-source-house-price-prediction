import numpy as np
import pytest

from src.evaluation.visualization import price_bin_errors


def test_price_bin_errors_returns_quantile_bin_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 360.0])

    rows = price_bin_errors(y_true, y_pred, n_bins=2)

    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["mae"] == pytest.approx(10.0)
    assert rows[0]["rmse"] == pytest.approx(10.0)
    assert rows[1]["count"] == 2
    assert rows[1]["mae"] == pytest.approx(35.0)
    assert rows[1]["rmse"] == pytest.approx((30.0**2 + 40.0**2) ** 0.5 / 2**0.5)


def test_price_bin_errors_rejects_constant_targets():
    with pytest.raises(ValueError, match="variation"):
        price_bin_errors(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
