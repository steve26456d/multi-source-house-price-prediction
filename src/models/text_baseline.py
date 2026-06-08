"""
文本基线模型

负责人：孙钰淼
周次：W14

包含两个仅使用文本特征的基线模型：
1. TFIDFRidgeBaseline —— TF-IDF 特征 + 岭回归
2. BERTMLPBaseline —— BERT 嵌入向量 + 多层感知机（MLP）

所有模型继承 BaseHousePriceModel，实现 fit / predict 统一接口。
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from .base import BaseHousePriceModel


# ============================================================
# PyTorch MLP 网络定义
# ============================================================

class MLPRegressor(nn.Module):
    """用于 BERT 嵌入向量的 MLP 回归器

    结构：输入层 → 隐藏层(ReLU+Dropout) → ... → 输出层(1维)

    Parameters
    ----------
    input_dim : int
        输入特征维度（BERT 嵌入为 768）。
    hidden_dims : list of int
        各隐藏层维度，如 [256, 128]。
    dropout : float
        Dropout 比率。
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # 输出层：1 个神经元（回归输出）
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x).squeeze(-1)


# ============================================================
# TF-IDF + Ridge 基线模型
# ============================================================

class TFIDFRidgeBaseline(BaseHousePriceModel):
    """TF-IDF + 岭回归基线模型

    使用 sklearn 的 Ridge 回归器。
    输入为 TruncatedSVD 降维后的 TF-IDF 特征（128 维）。

    岭回归通过 L2 正则化处理高维稀疏特征，
    是文本回归任务的经典基线。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        ridge_config = self.config.get("ridge_regression", {})
        self.model = Ridge(
            alpha=ridge_config.get("alpha", 1.0),
            random_state=ridge_config.get("random_state", 42),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "TFIDFRidgeBaseline":
        """训练岭回归模型（使用 TF-IDF 文本特征）

        X 参数在此模型中不使用，实际使用 X_text["tfidf"]。

        Parameters
        ----------
        X : np.ndarray
            占位参数，实际不使用。
        y : np.ndarray
            目标变量。
        X_text : dict
            必须包含 "tfidf" 键。
        """
        if X_text is None or "tfidf" not in X_text:
            raise ValueError(
                "TFIDFRidgeBaseline 需要 X_text 中包含 'tfidf' 键"
            )
        self.model.fit(X_text["tfidf"], y)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用岭回归预测房价"""
        self._check_fitted()
        if X_text is None or "tfidf" not in X_text:
            raise ValueError("预测需要 X_text 中包含 'tfidf' 键")
        return self.model.predict(X_text["tfidf"])

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")


# ============================================================
# BERT + MLP 基线模型
# ============================================================

class BERTMLPBaseline(BaseHousePriceModel):
    """BERT 嵌入向量 + MLP 基线模型

    使用 BERT CLS 嵌入（768 维）作为输入，
    通过 MLP 进行房价预测。

    训练使用 PyTorch，支持 GPU 加速。
    内部对目标变量进行标准化处理，
    预测时自动还原到原始尺度。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        bert_config = self.config.get("bert_mlp", {})
        self.hidden_dims = bert_config.get("hidden_dims", [256, 128])
        self.dropout = bert_config.get("dropout", 0.3)
        self.lr = bert_config.get("learning_rate", 1e-4)
        self.batch_size = bert_config.get("batch_size", 32)
        self.max_epochs = bert_config.get("max_epochs", 50)
        self.patience = bert_config.get("patience", 10)

        # 设备：优先 GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 模型在 fit 时根据输入维度创建
        self.model: Optional[MLPRegressor] = None
        # 目标变量标准化器（训练时拟合，预测时还原）
        self._y_scaler = None
        self._train_losses: list = []
        self._val_losses: list = []

    def _init_model(self, input_dim: int):
        """初始化 MLP 网络"""
        self.model = MLPRegressor(
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
    ) -> "BERTMLPBaseline":
        """训练 BERT+MLP 模型

        使用 X_text["bert_embeddings"] 作为输入。
        支持验证集早停。

        Parameters
        ----------
        X : np.ndarray
            占位参数，实际不使用。
        y : np.ndarray
            目标变量。
        X_text : dict
            必须包含 "bert_embeddings" 键。
        """
        if X_text is None or "bert_embeddings" not in X_text:
            raise ValueError(
                "BERTMLPBaseline 需要 X_text 中包含 'bert_embeddings' 键"
            )

        bert_features = X_text["bert_embeddings"].astype(np.float32)

        # 目标变量标准化：y → (y - mean) / std
        from sklearn.preprocessing import StandardScaler
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        # 验证集数据
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        X_text_val = kwargs.get("X_text_val", None)

        # 初始化模型
        self._init_model(bert_features.shape[1])

        # 转换为 Tensor
        X_tensor = torch.from_numpy(bert_features).to(self.device)
        y_tensor = torch.from_numpy(y_scaled).to(self.device)

        has_val = (
            X_val is not None
            and y_val is not None
            and X_text_val is not None
            and "bert_embeddings" in X_text_val
        )
        if has_val:
            X_val_tensor = torch.from_numpy(
                X_text_val["bert_embeddings"].astype(np.float32)
            ).to(self.device)
            y_val_scaled = self._y_scaler.transform(
                y_val.reshape(-1, 1)
            ).ravel().astype(np.float32)
            y_val_tensor = torch.from_numpy(y_val_scaled).to(self.device)

        # 优化器和损失函数
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

        n_samples = len(bert_features)
        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.max_epochs):
            # --- 训练阶段 ---
            self.model.train()
            train_loss = 0.0
            perm = torch.randperm(n_samples, device=self.device)

            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

                train_loss += loss.item()

            train_loss /= n_batches
            self._train_losses.append(train_loss)

            # --- 验证阶段 ---
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(
                        val_pred, y_val_tensor
                    ).item()
                self._val_losses.append(val_loss)

                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                # 无验证集时使用训练损失作为指标
                scheduler.step(train_loss)
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # 恢复最佳模型状态
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
        """使用 BERT+MLP 预测房价"""
        self._check_fitted()
        if X_text is None or "bert_embeddings" not in X_text:
            raise ValueError("预测需要 X_text 中包含 'bert_embeddings' 键")

        bert_features = X_text["bert_embeddings"].astype(np.float32)
        X_tensor = torch.from_numpy(bert_features).to(self.device)

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
