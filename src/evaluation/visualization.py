"""Visualization helpers for regression evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _prepare_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("inputs must not be empty.")
    return true, pred


def _save_figure(fig, save_path: Optional[str | Path]) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs True House Prices",
    save_path: Optional[str | Path] = None,
):
    """Create a predicted-vs-true scatter plot."""
    import matplotlib.pyplot as plt

    true, pred = _prepare_targets(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true, pred, alpha=0.65, edgecolors="none")
    lower = float(min(np.min(true), np.min(pred)))
    upper = float(max(np.max(true), np.max(pred)))
    ax.plot([lower, upper], [lower, upper], color="tab:red", linewidth=1.5, label="Ideal")
    ax.set_title(title)
    ax.set_xlabel("True price")
    ax.set_ylabel("Predicted price")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
    save_path: Optional[str | Path] = None,
    bins: int = 30,
):
    """Create a residual histogram."""
    import matplotlib.pyplot as plt

    true, pred = _prepare_targets(y_true, y_pred)
    residuals = pred - true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=bins, color="tab:blue", alpha=0.75)
    ax.axvline(0.0, color="tab:red", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Prediction residual")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    _save_figure(fig, save_path)
    return fig


def plot_feature_importance(
    importances: Sequence[float],
    feature_names: Sequence[str],
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str | Path] = None,
):
    """Create a horizontal feature-importance chart."""
    import matplotlib.pyplot as plt

    values = np.asarray(importances, dtype=float)
    names = np.asarray(feature_names, dtype=str)
    if values.ndim != 1:
        raise ValueError("importances must be one-dimensional.")
    if values.shape[0] != names.shape[0]:
        raise ValueError("importances and feature_names must have the same length.")
    if values.size == 0:
        raise ValueError("importances must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    selected = np.argsort(values)[-min(top_k, values.size) :]
    selected = selected[np.argsort(values[selected])]

    fig_height = max(4.0, len(selected) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(names[selected], values[selected], color="tab:green", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, save_path)
    return fig


def price_bin_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 4,
) -> list[dict[str, float | int | str]]:
    """Compute MAE and RMSE by true-price quantile bins."""
    true, pred = _prepare_targets(y_true, y_pred)
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than 1.")

    quantiles = np.quantile(true, np.linspace(0, 1, n_bins + 1))
    quantiles = np.unique(quantiles)
    if quantiles.size <= 1:
        raise ValueError("y_true must contain enough variation for price bins.")

    rows: list[dict[str, float | int | str]] = []
    for index in range(quantiles.size - 1):
        left = quantiles[index]
        right = quantiles[index + 1]
        if index == quantiles.size - 2:
            mask = (true >= left) & (true <= right)
        else:
            mask = (true >= left) & (true < right)
        if not np.any(mask):
            continue
        errors = pred[mask] - true[mask]
        rows.append(
            {
                "bin": f"{left:.2f}-{right:.2f}",
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            }
        )
    return rows
