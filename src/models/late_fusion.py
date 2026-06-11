"""
晚期融合模型（Stacking 集成）

负责人：孙钰淼
周次：W15

晚期融合策略：各模态独立训练基础模型进行预测，
然后使用元学习器（Meta-Learner）综合各模态的预测结果。

实现方式（Stacking）：
1. 第一层（Base Learners）：
   - 结构化模型：XGBoost（仅结构化特征）
   - 文本模型：BERT+MLP（仅文本特征）
2. 第二层（Meta Learner）：
   - Ridge 回归：综合两个基模型的预测输出

优势：
- 各模态模型可以独立优化
- 元学习器自动学习最优的模态权重
- 可灵活添加/替换基模型

参考：Wolpert (1992) "Stacked Generalization"
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .base import BaseHousePriceModel
from .text_baseline import MLPRegressor


# ============================================================
# 晚期融合模型
# ============================================================

class LateFusionStacking(BaseHousePriceModel):
    """Stacking 晚期融合模型

    第一层：独立的 XGBoost（结构化）和 BERT+MLP（文本）
    第二层：Ridge 回归作为元学习器

    Parameters
    ----------
    config : dict
        base_models: 基模型列表，如 ["xgboost", "bert_mlp"]
        meta_model: 元学习器类型，默认 "ridge"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        lf_config = self.config.get("late_fusion_stacking", {})
        self.base_model_names = lf_config.get(
            "base_models", ["xgboost", "bert_mlp"]
        )
        self.meta_model_name = lf_config.get("meta_model", "ridge")

        # 子模型将在 fit 时初始化
        self.struct_model: Optional[Any] = None
        self.text_model: Optional[MLPRegressor] = None
        self.meta_model: Optional[Ridge] = None

        # 文本模型配置
        self.text_hidden_dims = self.config.get(
            "bert_mlp", {}
        ).get("hidden_dims", [256, 128])
        self.text_dropout = self.config.get("bert_mlp", {}).get(
            "dropout", 0.3
        )

        # 训练配置
        self.batch_size = self.config.get(
            "late_fusion_stacking", {}
        ).get("batch_size", 64)
        self.max_epochs = self.config.get(
            "late_fusion_stacking", {}
        ).get("max_epochs", 50)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._y_scaler = None  # 文本模型目标标准化器
        self._train_losses: list = []
        self._val_losses: list = []

    def _init_base_models(self, struct_dim: int, text_dim: int):
        """初始化第一层基模型

        Parameters
        ----------
        struct_dim : int
            结构化特征维度。
        text_dim : int
            文本特征维度（BERT 768）。
        """
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed; install xgboost to use LateFusionStacking."
            )

        # 结构化基模型：XGBoost
        xgb_config = self.config.get("xgboost", {})
        self.struct_model = XGBRegressor(
            n_estimators=xgb_config.get("n_estimators", 300),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.05),
            random_state=xgb_config.get("random_state", 42),
            n_jobs=-1,
            verbosity=0,
        )

        # 文本基模型：MLP
        self.text_model = MLPRegressor(
            input_dim=text_dim,
            hidden_dims=self.text_hidden_dims,
            dropout=self.text_dropout,
        ).to(self.device)

        # 元学习器
        self.meta_model = Ridge(alpha=1.0, random_state=42)

    def _train_text_model(
        self,
        X_text: np.ndarray,
        y: np.ndarray,
        X_text_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ):
        """训练文本基模型（BERT+MLP），内部使用目标标准化"""
        # 目标标准化
        from sklearn.preprocessing import StandardScaler
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        X_tensor = torch.from_numpy(X_text.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_scaled).to(self.device)

        has_val = X_text_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.from_numpy(
                X_text_val.astype(np.float32)
            ).to(self.device)
            y_val_scaled = self._y_scaler.transform(
                y_val.reshape(-1, 1)
            ).ravel().astype(np.float32)
            y_val_tensor = torch.from_numpy(y_val_scaled).to(self.device)

        optimizer = torch.optim.AdamW(
            self.text_model.parameters(), lr=1e-4
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_samples = len(X_text)
        patience = 10

        for epoch in range(self.max_epochs):
            self.text_model.train()
            train_loss = 0.0
            perm = torch.randperm(n_samples, device=self.device)
            n_batches = max(1, n_samples // self.batch_size)

            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                pred = self.text_model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.text_model.parameters(), max_norm=1.0
                )
                optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batches
            self._train_losses.append(train_loss)

            if has_val:
                self.text_model.eval()
                with torch.no_grad():
                    val_pred = self.text_model(X_val_tensor)
                    val_loss = criterion(
                        val_pred, y_val_tensor
                    ).item()
                self._val_losses.append(val_loss)
                current_loss = val_loss
            else:
                current_loss = train_loss

            scheduler.step(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.text_model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.text_model.load_state_dict(best_state)
        self.text_model.eval()

    def _get_text_predictions(
        self, X_text: np.ndarray
    ) -> np.ndarray:
        """获取文本模型的预测值（还原到原始尺度）"""
        X_tensor = torch.from_numpy(
            X_text.astype(np.float32)
        ).to(self.device)
        self.text_model.eval()
        with torch.no_grad():
            pred_scaled = self.text_model(X_tensor).cpu().numpy()
        # 还原到原始尺度
        pred = self._y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        return pred

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "LateFusionStacking":
        """训练 Stacking 晚期融合模型

        训练流程：
        1. 训练结构化基模型（XGBoost）
        2. 训练文本基模型（BERT+MLP）
        3. 用基模型的预测作为特征训练元学习器

        Parameters
        ----------
        X : np.ndarray
            结构化特征矩阵。
        y : np.ndarray
            目标变量。
        X_text : dict
            必须包含 "bert_embeddings" 键。
        """
        if X_text is None or "bert_embeddings" not in X_text:
            raise ValueError(
                "LateFusionStacking 需要 X_text 中包含 "
                "'bert_embeddings' 键"
            )

        X_text_arr = X_text["bert_embeddings"]

        # 验证集
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        X_text_val = kwargs.get("X_text_val", None)

        # 初始化模型
        self._init_base_models(X.shape[1], X_text_arr.shape[1])

        # === 第一阶段：训练基模型 ===

        # 1. 训练结构化模型 (XGBoost)
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.struct_model.fit(X, y, **fit_params)

        # 2. 训练文本模型 (BERT+MLP)
        X_text_val_arr = (
            X_text_val["bert_embeddings"]
            if X_text_val is not None and "bert_embeddings" in X_text_val
            else None
        )
        self._train_text_model(
            X_text_arr, y, X_text_val_arr, y_val
        )

        # === 第二阶段：训练元学习器 ===

        # 获取基模型在训练集和验证集上的预测
        struct_pred_train = self.struct_model.predict(X)

        text_pred_train = self._get_text_predictions(X_text_arr)

        # 构建元特征矩阵：(n_samples, 2)
        meta_X_train = np.column_stack(
            [struct_pred_train, text_pred_train]
        )

        # 训练元学习器
        self.meta_model.fit(meta_X_train, y)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """使用 Stacking 模型预测房价

        预测流程：
        1. 各基模型独立预测
        2. 元学习器综合预测结果
        """
        self._check_fitted()
        if X_text is None or "bert_embeddings" not in X_text:
            raise ValueError(
                "预测需要 X_text 中包含 'bert_embeddings' 键"
            )

        # 基模型预测
        struct_pred = self.struct_model.predict(X)
        text_pred = self._get_text_predictions(
            X_text["bert_embeddings"]
        )

        # 元学习器综合
        meta_X = np.column_stack([struct_pred, text_pred])
        final_pred = self.meta_model.predict(meta_X)

        return final_pred

    def save(self, path: str) -> None:
        """保存模型到指定路径（覆盖基类以保存子模型）"""
        import joblib as _joblib
        tcfg = {}
        if self.text_model is not None:
            tcfg = {
                "input_dim": self.text_model.network[0].in_features,
                "hidden_dims": self.text_hidden_dims,
                "dropout": self.text_dropout,
            }
        data = {
            "struct_model": self.struct_model,
            "text_model_state": self.text_model.state_dict() if self.text_model is not None else None,
            "text_model_config": tcfg,
            "meta_model": self.meta_model,
            "config": self.config,
            "is_fitted": self.is_fitted,
            "_y_scaler": self._y_scaler,
        }
        _joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "LateFusionStacking":
        """从路径加载模型（覆盖基类以恢复子模型）"""
        import joblib as _joblib
        data = _joblib.load(path)
        instance = cls(config=data.get("config", {}))
        instance.struct_model = data["struct_model"]
        if data.get("text_model_state") is not None:
            tcfg = data.get("text_model_config", {})
            input_dim = tcfg.get("input_dim", 768)
            instance.text_model = MLPRegressor(
                input_dim=input_dim,
                hidden_dims=tcfg.get("hidden_dims", [256, 128]),
                dropout=tcfg.get("dropout", 0.3),
            ).to(instance.device)
            instance.text_model.load_state_dict(data["text_model_state"])
            instance.text_model.eval()
        instance.meta_model = data["meta_model"]
        instance.is_fitted = data.get("is_fitted", True)
        instance._y_scaler = data.get("_y_scaler", None)
        return instance

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
