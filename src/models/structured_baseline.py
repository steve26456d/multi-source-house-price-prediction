"""
结构化基线模型

负责人：孙钰淼
周次：W14

包含三个仅使用结构化特征的基线模型：
1. LinearBaseline —— 线性回归（性能下界）
2. RandomForestBaseline —— 随机森林（强基线）
3. XGBoostBaseline —— XGBoost（结构化 SOTA 基线）

所有模型继承 BaseHousePriceModel，实现 fit / predict 统一接口。
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .base import BaseHousePriceModel


class LinearBaseline(BaseHousePriceModel):
    """线性回归基线模型

    使用 sklearn 的 LinearRegression。
    作为性能下界参考：任何复杂模型应至少优于此模型。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = LinearRegression(
            **self.config.get("linear_regression", {})
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "LinearBaseline":
        """训练线性回归模型（仅使用结构化特征）"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用线性回归预测房价"""
        self._check_fitted()
        return self.model.predict(X)

    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")


class RandomForestBaseline(BaseHousePriceModel):
    """随机森林基线模型

    使用 sklearn 的 RandomForestRegressor。
    默认参数：200 棵树，最大深度 20，随机种子 42。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        rf_config = self.config.get("random_forest", {})
        self.model = RandomForestRegressor(
            n_estimators=rf_config.get("n_estimators", 200),
            max_depth=rf_config.get("max_depth", 20),
            random_state=rf_config.get("random_state", 42),
            n_jobs=-1,  # 并行训练
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "RandomForestBaseline":
        """训练随机森林模型（仅使用结构化特征）"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用随机森林预测房价"""
        self._check_fitted()
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性

        Returns
        -------
        importance : np.ndarray, shape (n_features,)
            各特征的重要性分数。
        """
        self._check_fitted()
        return self.model.feature_importances_

    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")


class XGBoostBaseline(BaseHousePriceModel):
    """XGBoost 基线模型

    使用 XGBoost 的 XGBRegressor。
    默认参数：500 棵树，最大深度 8，学习率 0.05，早停 50 轮。

    XGBoost 是结构化数据的 SOTA 基线，预期性能优于 RF 和 LR。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed; install xgboost to use XGBoostBaseline."
            )
        xgb_config = self.config.get("xgboost", {})
        self.model = XGBRegressor(
            n_estimators=xgb_config.get("n_estimators", 500),
            max_depth=xgb_config.get("max_depth", 8),
            learning_rate=xgb_config.get("learning_rate", 0.05),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            random_state=xgb_config.get("random_state", 42),
            n_jobs=-1,
            verbosity=0,
        )
        self._early_stopping_rounds = xgb_config.get(
            "early_stopping_rounds", 50
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "XGBoostBaseline":
        """训练 XGBoost 模型（仅使用结构化特征）

        支持验证集早停：传入 X_val 和 y_val 参数启用。
        """
        fit_params = {}
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)

        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用 XGBoost 预测房价"""
        self._check_fitted()
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性

        Returns
        -------
        importance : np.ndarray, shape (n_features,)
            各特征的重要性分数（gain 类型）。
        """
        self._check_fitted()
        return self.model.feature_importances_

    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
