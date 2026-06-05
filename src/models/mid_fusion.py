"""
中期融合模型（双塔 + 注意力融合）

负责人：孙钰淼
周次：W15

中期融合策略：结构化特征和文本特征分别通过各自的编码器
（结构化 FC + 文本 BERT 嵌入），提取高层次表示后通过
注意力机制融合，再进行预测。

"双塔"结构：
- 结构化塔：FC 网络将结构化特征映射到低维嵌入空间
- 文本塔：直接使用 BERT CLS 嵌入（或附加轻量投影层）

融合方式：
- 多头部注意力：将两个塔的输出拼接后通过自注意力层，
  让模型学习模态间交互权重。

优势：
- 各模态独立编码，保留模态特有信息
- 注意力机制自适应学习模态重要性
- 优于简单的早期拼接
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseHousePriceModel


# ============================================================
# 双塔 + 注意力融合网络
# ============================================================

class MultiHeadAttentionFusion(nn.Module):
    """多头部注意力融合层

    将结构化特征嵌入和文本特征嵌入通过多头注意力融合。

    原理：
    1. 将两个嵌入堆叠为序列 [struct_emb, text_emb]
    2. 通过多头自注意力学习跨模态交互
    3. 池化后输入预测头

    Parameters
    ----------
    embed_dim : int
        每个模态的嵌入维度（需保持一致）。
    num_heads : int
        注意力头数。
    dropout : float
        Dropout 比率。
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 4, dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, struct_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """注意力融合前向传播

        Parameters
        ----------
        struct_emb : (batch, embed_dim)
            结构化特征嵌入。
        text_emb : (batch, embed_dim)
            文本特征嵌入。

        Returns
        -------
        fused : (batch, embed_dim * 2)
            融合后的特征向量（拼接两个嵌入）。
        """
        # 将两个嵌入堆叠为序列：(batch, 2, embed_dim)
        seq = torch.stack([struct_emb, text_emb], dim=1)

        # 多头自注意力
        attn_out, _ = self.attention(seq, seq, seq)
        seq = seq + self.dropout(attn_out)  # 残差连接
        seq = self.norm(seq)

        # 拼接两个模态的输出
        fused = torch.cat(
            [seq[:, 0, :], seq[:, 1, :]], dim=-1
        )
        return fused


class TwoTowerAttentionNet(nn.Module):
    """双塔注意力融合网络

    结构：
    1. 结构化塔：FC 层将原始特征映射到 embed_dim
    2. 文本塔：FC 投影层（如需要）处理 BERT 嵌入
    3. 注意力融合层
    4. 预测头：FC 层输出房价预测
    """

    def __init__(
        self,
        struct_input_dim: int,
        text_input_dim: int,
        struct_hidden_dim: int = 64,
        text_hidden_dim: int = 768,
        fusion_dim: int = 256,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # === 结构化塔 ===
        self.struct_tower = nn.Sequential(
            nn.Linear(struct_input_dim, struct_hidden_dim * 4),
            nn.BatchNorm1d(struct_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_hidden_dim * 4, struct_hidden_dim * 2),
            nn.BatchNorm1d(struct_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_hidden_dim * 2, struct_hidden_dim),
            nn.BatchNorm1d(struct_hidden_dim),
            nn.ReLU(),
        )

        # === 文本塔（投影层） ===
        # 如果 BERT 维度与目标嵌入维度不同，添加投影
        if text_input_dim != text_hidden_dim:
            self.text_tower = nn.Sequential(
                nn.Linear(text_input_dim, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.text_tower = nn.Identity()

        # === 注意力融合层 ===
        # 使用较小的嵌入维度进行注意力计算
        attn_embed_dim = min(struct_hidden_dim, text_hidden_dim)
        self.struct_proj = nn.Linear(struct_hidden_dim, attn_embed_dim)
        self.text_proj = nn.Linear(text_hidden_dim, attn_embed_dim)

        self.attention_fusion = MultiHeadAttentionFusion(
            embed_dim=attn_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # === 预测头 ===
        # 融合后维度为 attn_embed_dim * 2
        fusion_input_dim = attn_embed_dim * 2
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(
        self, X_struct: torch.Tensor, X_text: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Parameters
        ----------
        X_struct : (batch, struct_input_dim)
            结构化特征。
        X_text : (batch, text_input_dim)
            文本嵌入向量（BERT 或 TF-IDF）。

        Returns
        -------
        pred : (batch,)
            房价预测值。
        """
        # 各塔编码
        struct_emb = self.struct_tower(X_struct)
        text_emb = self.text_tower(X_text)

        # 投影到统一维度
        struct_proj = self.struct_proj(struct_emb)
        text_proj = self.text_proj(text_emb)

        # 注意力融合
        fused = self.attention_fusion(struct_proj, text_proj)

        # 预测
        pred = self.prediction_head(fused).squeeze(-1)
        return pred


# ============================================================
# 中期融合模型
# ============================================================

class MidFusionModel(BaseHousePriceModel):
    """中期融合模型（双塔 + 注意力融合）

    采用双塔架构分别编码结构化特征和文本特征，
    通过多头注意力机制融合两个模态的高层次表示，
    最后使用预测头输出房价。

    这是本项目的核心创新模型。

    Parameters
    ----------
    config : dict
        配置参数，包括各塔维度和注意力参数。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        mf_config = self.config.get("mid_fusion_attention", {})
        self.struct_hidden_dim = mf_config.get("structured_hidden_dim", 64)
        self.text_hidden_dim = mf_config.get("text_hidden_dim", 768)
        self.fusion_dim = mf_config.get("fusion_dim", 256)
        self.num_attention_heads = mf_config.get("num_attention_heads", 4)
        self.dropout = mf_config.get("dropout", 0.3)
        self.lr = mf_config.get("learning_rate", 1e-4)
        self.batch_size = mf_config.get("batch_size", 64)
        self.max_epochs = mf_config.get("max_epochs", 50)
        self.patience = mf_config.get("patience", 10)
        self.text_source = self.config.get("text_source", "bert_embeddings")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[TwoTowerAttentionNet] = None
        self._y_scaler = None  # 目标标准化器
        self._train_losses: list = []
        self._val_losses: list = []

    def _init_model(self, struct_dim: int, text_dim: int):
        """初始化双塔注意力网络"""
        self.model = TwoTowerAttentionNet(
            struct_input_dim=struct_dim,
            text_input_dim=text_dim,
            struct_hidden_dim=self.struct_hidden_dim,
            text_hidden_dim=self.text_hidden_dim,
            fusion_dim=self.fusion_dim,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_text: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> "MidFusionModel":
        """训练中期融合模型

        Parameters
        ----------
        X : np.ndarray
            结构化特征矩阵。
        y : np.ndarray
            目标变量。
        X_text : dict
            必须包含 text_source 指定的文本特征（默认 bert_embeddings）。
        """
        if X_text is None or self.text_source not in X_text:
            raise ValueError(
                f"MidFusionModel 需要 X_text 中包含 "
                f"'{self.text_source}' 键"
            )

        X_struct = X.astype(np.float32)
        X_text_arr = X_text[self.text_source].astype(np.float32)

        # 目标标准化
        from sklearn.preprocessing import StandardScaler
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        self._init_model(X_struct.shape[1], X_text_arr.shape[1])

        # 转换为 Tensor
        X_tensor = torch.from_numpy(X_struct).to(self.device)
        X_text_tensor = torch.from_numpy(X_text_arr).to(self.device)
        y_tensor = torch.from_numpy(y_scaled).to(self.device)

        # 验证集
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)
        X_text_val = kwargs.get("X_text_val", None)
        has_val = (
            X_val is not None
            and y_val is not None
            and X_text_val is not None
            and self.text_source in X_text_val
        )
        if has_val:
            X_val_tensor = torch.from_numpy(
                X_val.astype(np.float32)
            ).to(self.device)
            X_text_val_tensor = torch.from_numpy(
                X_text_val[self.text_source].astype(np.float32)
            ).to(self.device)
            y_val_scaled = self._y_scaler.transform(
                y_val.reshape(-1, 1)
            ).ravel().astype(np.float32)
            y_val_tensor = torch.from_numpy(y_val_scaled).to(self.device)

        # 优化器和损失
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_samples = len(X_struct)

        for epoch in range(self.max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            perm = torch.randperm(n_samples, device=self.device)
            n_batches = max(1, n_samples // self.batch_size)

            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch_X = X_tensor[idx]
                batch_X_text = X_text_tensor[idx]
                batch_y = y_tensor[idx]

                optimizer.zero_grad()
                pred = self.model(batch_X, batch_X_text)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batches
            self._train_losses.append(train_loss)

            # 验证阶段
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(
                        X_val_tensor, X_text_val_tensor
                    )
                    val_loss = criterion(
                        val_pred, y_val_tensor
                    ).item()
                self._val_losses.append(val_loss)
                current_loss = val_loss
            else:
                current_loss = train_loss

            scheduler.step(current_loss)

            # 早停
            if current_loss < best_loss:
                best_loss = current_loss
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
        """使用中期融合模型预测房价"""
        self._check_fitted()
        if X_text is None or self.text_source not in X_text:
            raise ValueError(
                f"预测需要 X_text 中包含 '{self.text_source}' 键"
            )

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X_text_tensor = torch.from_numpy(
            X_text[self.text_source].astype(np.float32)
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(
                X_tensor, X_text_tensor
            ).cpu().numpy()

        # 还原到原始尺度
        predictions = self._y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        return predictions

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
