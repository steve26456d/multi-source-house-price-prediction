"""
超参数调优模块

负责人：孙钰淼
周次：W15

使用 Optuna 进行超参数调优，支持以下模型的优化：
1. RandomForest
2. XGBoost（结构化基线 + 早期融合）
3. Ridge（TF-IDF 基线）
4. BERT+MLP
5. 早期融合 MLP
6. 中期融合模型
7. 晚期融合 Stacking

优化目标：最小化验证集 RMSE
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error


# ============================================================
# 评估函数
# ============================================================

def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# 通用调优框架
# ============================================================

class HyperparamTuner:
    """基于 Optuna 的超参数调优器

    使用方式：
    1. 实例化 Tuner 并传入模型类和目标函数
    2. 调用 tune() 执行超参数搜索
    3. 获取最佳参数和模型

    Parameters
    ----------
    model_class : class
        需要调优的模型类（需继承 BaseHousePriceModel）。
    direction : str
        优化方向，"minimize"（默认）表示最小化目标函数。
    n_trials : int
        Optuna 试验次数。
    """

    def __init__(
        self,
        model_class,
        direction: str = "minimize",
        n_trials: int = 50,
        random_state: int = 42,
    ):
        self.model_class = model_class
        self.direction = direction
        self.n_trials = n_trials
        self.random_state = random_state
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_text_train: Optional[Dict[str, np.ndarray]] = None,
        X_text_val: Optional[Dict[str, np.ndarray]] = None,
        objective_fn: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行超参数搜索

        Parameters
        ----------
        X_train, y_train : 训练数据
        X_val, y_val : 验证数据
        X_text_train, X_text_val : 文本特征
        objective_fn : 自定义目标函数（可选），签名需为
                       fn(trial, X_train, y_train, X_val, y_val, ...)

        Returns
        -------
        best_params : dict
            最佳超参数字典。
        """
        if objective_fn is None:
            raise ValueError("需要提供 objective_fn")

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )

        self.study.optimize(
            lambda trial: objective_fn(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                X_text_train,
                X_text_val,
                **kwargs,
            ),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params

    def get_best_model(self):
        """使用最佳参数训练并返回模型"""
        if self.best_params is None:
            raise RuntimeError("请先调用 tune() 完成超参数搜索")
        return self.model_class(config=self.best_params)


# ============================================================
# 各模型的目标函数
# ============================================================

def rf_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """RandomForest 超参数搜索目标函数

    搜索空间：
    - n_estimators: [100, 500]
    - max_depth: [5, 40]
    - min_samples_split: [2, 20]
    - min_samples_leaf: [1, 10]
    """
    from .structured_baseline import RandomForestBaseline

    params = {
        "random_forest": {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
        }
    }
    model = RandomForestBaseline(config=params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return _compute_rmse(y_val, y_pred)


def xgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """XGBoost 超参数搜索目标函数

    搜索空间：
    - n_estimators: [100, 800]
    - max_depth: [3, 15]
    - learning_rate: [0.01, 0.3]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    """
    from .structured_baseline import XGBoostBaseline

    params = {
        "xgboost": {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }
    }
    model = XGBoostBaseline(config=params)
    model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val
    )
    y_pred = model.predict(X_val)
    return _compute_rmse(y_val, y_pred)


def ridge_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """Ridge 超参数搜索目标函数

    搜索空间：
    - alpha: [0.01, 100.0]（对数空间）
    """
    from .text_baseline import TFIDFRidgeBaseline

    params = {
        "ridge_regression": {
            "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
        }
    }
    model = TFIDFRidgeBaseline(config=params)
    model.fit(None, y_train, X_text=X_text_train)
    y_pred = model.predict(None, X_text=X_text_val)
    return _compute_rmse(y_val, y_pred)


def bert_mlp_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """BERT+MLP 超参数搜索目标函数

    搜索空间：
    - learning_rate: [1e-5, 1e-2]（对数空间）
    - dropout: [0.1, 0.5]
    - hidden_dim_0: [64, 512]
    - hidden_dim_1: [32, 256]
    - batch_size: [16, 128]
    """
    from .text_baseline import BERTMLPBaseline

    h0 = trial.suggest_int("hidden_dim_0", 64, 512)
    h1 = trial.suggest_int("hidden_dim_1", 32, min(256, h0))

    params = {
        "bert_mlp": {
            "hidden_dims": [h0, h1],
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "max_epochs": 30,
            "patience": 8,
        }
    }
    model = BERTMLPBaseline(config=params)
    model.fit(
        None,
        y_train,
        X_text=X_text_train,
        X_val=None,
        y_val=y_val,
        X_text_val=X_text_val,
    )
    y_pred = model.predict(None, X_text=X_text_val)
    return _compute_rmse(y_val, y_pred)


def early_fusion_mlp_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """早期融合 MLP 超参数搜索目标函数

    搜索空间：
    - learning_rate: [1e-5, 1e-2]
    - dropout: [0.1, 0.5]
    - hidden_dim_0: [128, 1024]
    - hidden_dim_1: [64, 512]
    - hidden_dim_2: [32, 256]
    - batch_size: [16, 128]
    """
    from .early_fusion import EarlyFusionMLP

    h0 = trial.suggest_int("hidden_dim_0", 128, 1024)
    h1 = trial.suggest_int("hidden_dim_1", 64, 512)
    h2 = trial.suggest_int("hidden_dim_2", 32, 256)

    params = {
        "early_fusion_mlp": {
            "hidden_dims": [h0, h1, h2],
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "max_epochs": 30,
            "patience": 8,
        }
    }
    model = EarlyFusionMLP(config=params)
    model.fit(
        X_train,
        y_train,
        X_text=X_text_train,
        X_val=X_val,
        y_val=y_val,
        X_text_val=X_text_val,
    )
    y_pred = model.predict(X_val, X_text=X_text_val)
    return _compute_rmse(y_val, y_pred)


def mid_fusion_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train=None,
    X_text_val=None,
) -> float:
    """中期融合模型超参数搜索目标函数

    搜索空间：
    - struct_hidden_dim: [32, 256]
    - fusion_dim: [64, 512]
    - num_attention_heads: [2, 8]
    - learning_rate: [1e-5, 1e-2]
    - dropout: [0.1, 0.5]
    """
    from .mid_fusion import MidFusionModel

    params = {
        "mid_fusion_attention": {
            "structured_hidden_dim": trial.suggest_int(
                "struct_hidden_dim", 32, 256
            ),
            "fusion_dim": trial.suggest_int("fusion_dim", 64, 512),
            "num_attention_heads": trial.suggest_int(
                "num_attention_heads", 2, 8
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "max_epochs": 30,
            "patience": 8,
        },
        "text_source": "bert_embeddings",
    }
    model = MidFusionModel(config=params)
    model.fit(
        X_train,
        y_train,
        X_text=X_text_train,
        X_val=X_val,
        y_val=y_val,
        X_text_val=X_text_val,
    )
    y_pred = model.predict(X_val, X_text=X_text_val)
    return _compute_rmse(y_val, y_pred)


# ============================================================
# 快速调优入口
# ============================================================

# 各模型对应的目标函数映射
OBJECTIVE_MAP = {
    "random_forest": rf_objective,
    "xgboost": xgb_objective,
    "ridge": ridge_objective,
    "bert_mlp": bert_mlp_objective,
    "early_fusion_mlp": early_fusion_mlp_objective,
    "mid_fusion": mid_fusion_objective,
}


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_text_train: Optional[Dict[str, np.ndarray]] = None,
    X_text_val: Optional[Dict[str, np.ndarray]] = None,
    n_trials: int = 30,
) -> Dict[str, Any]:
    """快速调优指定模型

    Parameters
    ----------
    model_name : str
        模型名称，可选 "random_forest", "xgboost", "ridge",
        "bert_mlp", "early_fusion_mlp", "mid_fusion"。
    n_trials : int
        Optuna 试验次数。

    Returns
    -------
    best_params : dict
        最佳超参数。
    """
    if model_name not in OBJECTIVE_MAP:
        raise ValueError(
            f"未知模型: {model_name}，可选: {list(OBJECTIVE_MAP.keys())}"
        )

    objective_fn = OBJECTIVE_MAP[model_name]

    # 创建匿名学习器类用于 Tuner
    class _TunedModel:
        def __init__(self, config=None):
            pass

    tuner = HyperparamTuner(
        model_class=_TunedModel,
        direction="minimize",
        n_trials=n_trials,
    )

    return tuner.tune(
        X_train,
        y_train,
        X_val,
        y_val,
        X_text_train,
        X_text_val,
        objective_fn=objective_fn,
    )
