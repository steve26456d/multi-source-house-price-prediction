"""Regression metrics for house price prediction experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _as_1d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Convert input values to a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = _as_1d_float_array(y_true, "y_true")
    pred = _as_1d_float_array(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return true, pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    true, pred = _validate_targets(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute error."""
    true, pred = _validate_targets(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return coefficient of determination."""
    true, pred = _validate_targets(y_true, y_pred)
    total_sum_squares = np.sum(np.square(true - np.mean(true)))
    if total_sum_squares == 0:
        return 1.0 if np.allclose(true, pred) else 0.0
    residual_sum_squares = np.sum(np.square(true - pred))
    return float(1.0 - residual_sum_squares / total_sum_squares)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Return mean absolute percentage error in percent."""
    true, pred = _validate_targets(y_true, y_pred)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all project regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth house prices with shape ``(n_samples,)``.
    y_pred:
        Predicted house prices with shape ``(n_samples,)``.

    Returns
    -------
    dict
        Metrics keyed by ``rmse``, ``mae``, ``r2``, and ``mape``.
    """
    true, pred = _validate_targets(y_true, y_pred)
    return {
        "rmse": rmse(true, pred),
        "mae": mae(true, pred),
        "r2": r2_score(true, pred),
        "mape": mape(true, pred),
    }


def compute_metrics_safe(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics after dropping non-finite prediction pairs.

    This preserves compatibility with the algorithm branch's temporary helper
    while keeping the formal metric definitions strict by default.
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
        }
    return compute_metrics(true[mask], pred[mask])


def error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for prediction errors."""
    true, pred = _validate_targets(y_true, y_pred)
    residual = pred - true
    absolute_error = np.abs(residual)
    return {
        "mean_error": float(np.mean(residual)),
        "median_error": float(np.median(residual)),
        "mean_abs_error": float(np.mean(absolute_error)),
        "median_abs_error": float(np.median(absolute_error)),
        "max_abs_error": float(np.max(absolute_error)),
    }
