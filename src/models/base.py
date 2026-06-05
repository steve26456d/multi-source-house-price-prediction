"""
房价预测模型基类

负责人：孙钰淼
周次：W14

定义所有模型的统一接口，所有算法模型需继承此基类。
遵循 doc/interface_spec.md 规范。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import joblib


class BaseHousePriceModel(ABC):
    """房价预测模型基类

    所有模型需实现 fit 和 predict 方法。
    支持结构化特征 + 可选文本特征的多模态输入。

    Attributes
    ----------
    config : Dict[str, Any]
        模型配置字典，包含超参数等。
    model : Any
        底层模型实例（sklearn estimator / torch module 等）。
    is_fitted : bool
        标记模型是否已完成训练。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模型

        Parameters
        ----------
        config : dict, optional
            模型配置参数，默认使用空字典。
        """
        self.config = config if config is not None else {}
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "BaseHousePriceModel":
        """训练模型

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            结构化特征矩阵。
        y : np.ndarray, shape (n_samples,)
            目标变量（房价）。
        X_text : dict, optional
            文本特征字典，键为 "tfidf" 或 "bert_embeddings"。
            仅文本模型和融合模型需要。

        Returns
        -------
        self : BaseHousePriceModel
            返回已训练的模型实例。
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """预测房价

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            结构化特征矩阵。
        X_text : dict, optional
            文本特征字典，键为 "tfidf" 或 "bert_embeddings"。

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            预测房价（原始尺度）。
        """
        ...

    def save(self, path: str) -> None:
        """保存模型到指定路径

        使用 joblib 序列化模型对象。

        Parameters
        ----------
        path : str
            模型保存路径。
        """
        joblib.dump(
            {
                "model": self.model,
                "config": self.config,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "BaseHousePriceModel":
        """从路径加载模型

        Parameters
        ----------
        path : str
            模型文件路径。

        Returns
        -------
        model : BaseHousePriceModel
            加载后的模型实例。
        """
        data = joblib.load(path)
        instance = cls(config=data.get("config", {}))
        instance.model = data["model"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
