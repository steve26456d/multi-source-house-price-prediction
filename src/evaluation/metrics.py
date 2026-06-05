"""
临时评估工具函数（供 experiment_runner.py 使用）

⚠️ 注意：评估指标模块的正式版本由石韫嘉（评估负责人）负责。
此文件仅为支撑算法实验运行而创建的临时工具，包含最基础的
RMSE / MAE / R² / MAPE 计算。

正式版本见 石韫嘉 提交的 src/evaluation/metrics.py。
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """计算所有评估指标

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        真实房价（原始尺度）。
    y_pred : np.ndarray, shape (n_samples,)
        预测房价（原始尺度）。

    Returns
    -------
    metrics : Dict[str, float]
        包含 rmse, mae, r2, mape 的字典。

    Notes
    -----
    MAPE 计算时对接近零的真实值做了保护处理。
    """
    # 确保输入是一维数组
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE：对接近零的值做保护，避免除零
    epsilon = 1e-8
    mask = np.abs(y_true) > epsilon
    if mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (y_true[mask] - y_pred[mask]) / y_true[mask]
                )
            )
            * 100
        )
    else:
        mape = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def compute_metrics_safe(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """安全版指标计算（与 compute_metrics 相同，增加异常处理）

    在包含 NaN 或 Inf 值时不会崩溃。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 过滤 NaN/Inf
    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
        }

    return compute_metrics(y_true, y_pred)
