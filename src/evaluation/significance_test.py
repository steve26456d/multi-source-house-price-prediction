"""Statistical tests for comparing model prediction errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SignificanceResult:
    """Container for a pairwise significance test result."""

    method: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool

    def to_dict(self) -> Dict[str, float | str | bool]:
        return {
            "method": self.method,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
        }


def _absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("inputs must not be empty.")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("inputs must contain only finite values.")
    return np.abs(true - pred)


def compare_absolute_errors(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    method: str = "wilcoxon",
    alpha: float = 0.05,
) -> SignificanceResult:
    """Compare two models using paired absolute errors.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred_a:
        Predictions from the first model.
    y_pred_b:
        Predictions from the second model.
    method:
        ``"wilcoxon"`` or ``"ttest"``.
    alpha:
        Significance level.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1.")

    errors_a = _absolute_errors(y_true, y_pred_a)
    errors_b = _absolute_errors(y_true, y_pred_b)
    if errors_a.shape != errors_b.shape:
        raise ValueError("prediction arrays must have the same shape.")

    method_normalized = method.lower()
    if method_normalized == "wilcoxon":
        if np.allclose(errors_a, errors_b):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = stats.wilcoxon(errors_a, errors_b)
    elif method_normalized in {"ttest", "paired_ttest", "paired-t-test"}:
        statistic, p_value = stats.ttest_rel(errors_a, errors_b)
    else:
        raise ValueError("method must be 'wilcoxon' or 'ttest'.")

    return SignificanceResult(
        method=method_normalized,
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=float(alpha),
        significant=bool(p_value < alpha),
    )


def improvement_ratio(baseline_metric: float, candidate_metric: float) -> float:
    """Return relative error reduction from baseline to candidate in percent."""
    if baseline_metric <= 0:
        raise ValueError("baseline_metric must be positive.")
    return float((baseline_metric - candidate_metric) / baseline_metric * 100.0)
