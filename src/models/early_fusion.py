"""
早期融合模型

负责人：孙钰淼
周次：W15

早期融合策略：在输入层将结构化特征与文本向量直接拼接，
然后送入统一的模型进行训练。

包含两个模型：
1. EarlyFusionXGBoost —— 拼接特征 + XGBoost
2. EarlyFusionMLP —— 拼接特征 + 多层感知机（PyTorch）

优势：实现简单，模型自动学习跨模态交互。
劣势：未考虑模态差异，文本特征可能被高维结构化特征淹没。
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .base import BaseHousePriceModel
from ..features.fusion import concat_features


# ============================================================
# 融合 MLP 网络定义
# ============================================================

class FusionMLP(nn.Module):
    """用于拼接特征的 MLP 回归器

    结构：输入层 → 隐藏层(ReLU+BatchNorm+Dropout) → ... → 输出层(1维)

    使用 BatchNorm 加速训练、稳定梯度。
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x).squeeze(-1)


# ============================================================
# 早期融合 + XGBoost
# ============================================================

class EarlyFusionXGBoost(BaseHousePriceModel):
    """早期融合 XGBoost 模型

    将结构化特征与文本向量（TF-IDF 或 BERT）直接拼接，
    使用 XGBoost 进行训练和预测。

    Parameters
    ----------
    config : dict
        text_source: 文本特征来源，可选 "tfidf" 或 "bert_embeddings"。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed; install xgboost to use EarlyFusionXGBoost."
            )
        xgb_config = self.config.get("early_fusion_xgboost", {})
        self.text_source = self.config.get(
            "text_source", "tfidf"
        )
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

    def _concat(
        self, X: np.ndarray, X_text: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """拼接结构化特征和文本特征"""
        if X_text is None or self.text_source not in X_text:
            raise ValueError(
                f"需要 X_text 中包含 '{self.text_source}' 键"
            )
        return concat_features(X, X_text[self.text_source])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "EarlyFusionXGBoost":
        """训练早期融合 XGBoost 模型

        将结构化特征与文本向量拼接后训练。
        支持验证集早停。
        """
        X_fused = self._concat(X, X_text)

        fit_params = {}
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        X_text_val = kwargs.get("X_text_val", None)

        if X_val is not None and y_val is not None and X_text_val is not None:
            X_val_fused = self._concat(X_val, X_text_val)
            fit_params["eval_set"] = [(X_val_fused, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_fused, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用早期融合 XGBoost 预测房价"""
        self._check_fitted()
        X_fused = self._concat(X, X_text)
        return self.model.predict(X_fused)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        self._check_fitted()
        return self.model.feature_importances_

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")


# ============================================================
# 早期融合 + MLP
# ============================================================

class EarlyFusionMLP(BaseHousePriceModel):
    """早期融合 MLP 模型

    将结构化特征与文本向量（TF-IDF 或 BERT）直接拼接，
    使用多层感知机进行训练和预测。

    相比 XGBoost 版本，MLP 能学习更复杂的非线性交互。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        mlp_config = self.config.get("early_fusion_mlp", {})
        self.hidden_dims = mlp_config.get(
            "hidden_dims", [512, 256, 128]
        )
        self.dropout = mlp_config.get("dropout", 0.3)
        self.lr = mlp_config.get("learning_rate", 1e-4)
        self.batch_size = mlp_config.get("batch_size", 64)
        self.max_epochs = mlp_config.get("max_epochs", 50)
        self.patience = mlp_config.get("patience", 10)
        self.text_source = self.config.get("text_source", "tfidf")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[FusionMLP] = None
        self._y_scaler = None  # 目标标准化器
        self._train_losses: list = []
        self._val_losses: list = []

    def _concat(
        self, X: np.ndarray, X_text: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """拼接结构化特征和文本特征"""
        if X_text is None or self.text_source not in X_text:
            raise ValueError(
                f"需要 X_text 中包含 '{self.text_source}' 键"
            )
        return concat_features(X, X_text[self.text_source])

    def _init_model(self, input_dim: int):
        """初始化 MLP 网络"""
        self.model = FusionMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "EarlyFusionMLP":
        """训练早期融合 MLP 模型"""
        X_fused = self._concat(X, X_text).astype(np.float32)

        # 目标标准化
        from sklearn.preprocessing import StandardScaler
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        self._init_model(X_fused.shape[1])

        # 准备训练数据
        X_tensor = torch.from_numpy(X_fused).to(self.device)
        y_tensor = torch.from_numpy(y_scaled).to(self.device)

        # 验证集
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        X_text_val = kwargs.get("X_text_val", None)
        has_val = (
            X_val is not None
            and y_val is not None
            and X_text_val is not None
        )
        if has_val:
            X_val_fused = self._concat(X_val, X_text_val).astype(np.float32)
            X_val_tensor = torch.from_numpy(X_val_fused).to(self.device)
            y_val_scaled = self._y_scaler.transform(
                y_val.reshape(-1, 1)
            ).ravel().astype(np.float32)
            y_val_tensor = torch.from_numpy(y_val_scaled).to(self.device)

        # 训练配置
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_samples = len(X_fused)

        for epoch in range(self.max_epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            perm = torch.randperm(n_samples, device=self.device)
            n_batches = max(1, n_samples // self.batch_size)

            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batches
            self._train_losses.append(train_loss)

            # 验证
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(
                        val_pred, y_val_tensor
                    ).item()
                self._val_losses.append(val_loss)
                current_loss = val_loss
            else:
                current_loss = train_loss

            scheduler.step(current_loss)

            if current_loss < best_val_loss:
                best_val_loss = current_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用早期融合 MLP 预测房价"""
        self._check_fitted()
        X_fused = self._concat(X, X_text).astype(np.float32)
        X_tensor = torch.from_numpy(X_fused).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()

        # 还原到原始尺度
        predictions = self._y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        return predictions

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
