"""
模型模块

包含所有房价预测模型的实现，遵循统一 BaseHousePriceModel 接口。

模型列表：
- 结构化基线：LinearBaseline, RandomForestBaseline, XGBoostBaseline
- 文本基线：TFIDFRidgeBaseline, BERTMLPBaseline
- 早期融合：EarlyFusionXGBoost, EarlyFusionMLP
- 中期融合：MidFusionModel
- 晚期融合：LateFusionStacking
"""

from .base import BaseHousePriceModel
from .structured_baseline import (
    LinearBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from .text_baseline import (
    TFIDFRidgeBaseline,
    BERTMLPBaseline,
)
from .early_fusion import (
    EarlyFusionXGBoost,
    EarlyFusionMLP,
)
from .mid_fusion import MidFusionModel
from .late_fusion import LateFusionStacking

__all__ = [
    "BaseHousePriceModel",
    "LinearBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "TFIDFRidgeBaseline",
    "BERTMLPBaseline",
    "EarlyFusionXGBoost",
    "EarlyFusionMLP",
    "MidFusionModel",
    "LateFusionStacking",
]
