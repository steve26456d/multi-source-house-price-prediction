# 接口规范文档

> 起草人：孙钰淼（算法负责人）
>
> 最后更新：第 13 周
>
> 状态：待讨论确认

---

## 一、数据接口规范

### 1.1 结构化数据 DataFrame Schema

所有预处理后的结构化数据必须遵循以下 Schema：

| 列名 | 类型 | 说明 | 允许缺失 |
|------|------|------|----------|
| `_id` | `int` | 样本唯一 ID | 否 |
| `_split` | `str` | 数据集划分：`train` / `val` / `test` | 否 |
| `[numerical_features]` | `float64` | 标准化后的数值特征 | 否 |
| `[categorical_features]` | `int` / `float` | 编码后的类别特征 | 否 |

### 1.2 文本数据接口

```python
# 文本预处理后的数据结构
TextFeatures = Dict[str, np.ndarray]
# {
#     "tfidf": np.ndarray,       # shape: (n_samples, 150)
#     "bert_embeddings": np.ndarray,  # shape: (n_samples, 768)
#     "raw_tokens": List[List[str]],  # 分词结果（可选）
# }
```

### 1.3 训练数据接口

```python
@dataclass
class MultiModalData:
    """多模态数据容器"""
    X_structured: np.ndarray          # (n, d_struct)
    X_text: TextFeatures              # 文本特征字典
    y: np.ndarray                     # (n,) 房价标签
    feature_names: List[str]          # 结构化特征名
    split_indices: Dict[str, np.ndarray]  # {train/val/test: indices}
```

---

## 二、模型接口规范

所有模型需继承基类并实现统一接口：

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class BaseHousePriceModel(ABC):
    """房价预测模型基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_text: Optional[Dict[str, np.ndarray]] = None,
            **kwargs) -> "BaseHousePriceModel":
        """
        训练模型。

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
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray,
                X_text: Optional[Dict[str, np.ndarray]] = None
                ) -> np.ndarray:
        """
        预测房价。

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            预测房价（原始尺度）。
        """
        ...

    def save(self, path: str) -> None:
        """保存模型到指定路径（joblib 或 torch.save）"""
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "BaseHousePriceModel":
        """从路径加载模型"""
        import joblib
        instance = cls({})
        instance.model = joblib.load(path)
        instance.is_fitted = True
        return instance
```

### 2.1 模型命名规范

| 模型 | 类名 | 模块 |
|------|------|------|
| Linear Regression | `LinearBaseline` | `src/models/structured_baseline.py` |
| Random Forest | `RandomForestBaseline` | `src/models/structured_baseline.py` |
| XGBoost | `XGBoostBaseline` | `src/models/structured_baseline.py` |
| TF-IDF + Ridge | `TFIDFRidgeBaseline` | `src/models/text_baseline.py` |
| BERT + MLP | `BERTMLPBaseline` | `src/models/text_baseline.py` |
| 早期融合 | `EarlyFusionModel` | `src/models/early_fusion.py` |
| 中期融合 | `MidFusionModel` | `src/models/mid_fusion.py` |
| 晚期融合 | `LateFusionModel` | `src/models/late_fusion.py` |

---

## 三、评估接口规范

```python
from typing import Dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标。

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        真实房价。
    y_pred : np.ndarray, shape (n_samples,)
        预测房价。

    Returns
    -------
    metrics : Dict[str, float]
        包含 rmse, mae, r2, mape 的字典。
    """
    return {
        "rmse": ...,
        "mae": ...,
        "r2": ...,
        "mape": ...,
    }
```

### 3.1 实验记录格式

所有实验结果统一记录为 CSV：

| 列名 | 类型 | 说明 |
|------|------|------|
| `experiment_id` | `str` | 实验唯一 ID |
| `model_name` | `str` | 模型名称 |
| `modality` | `str` | 模态：`structured` / `text` / `fusion_early` / `fusion_mid` / `fusion_late` |
| `split` | `str` | `train` / `val` / `test` |
| `rmse` | `float` | |
| `mae` | `float` | |
| `r2` | `float` | |
| `mape` | `float` | |
| `params` | `str` | 超参数字典（JSON 串） |
| `timestamp` | `str` | 实验时间戳 |

---

## 四、Pipeline 接口规范

```python
def run_pipeline(config_path: str = "config/config.yaml") -> None:
    """
    端到端 Pipeline 入口。

    1. 加载配置
    2. 加载并预处理数据
    3. 特征工程
    4. 训练所有启用的模型
    5. 评估并保存结果
    """
    ...
```

CLI 入口（`src/pipeline/run_pipeline.py`）：

```bash
python src/pipeline/run_pipeline.py --config config/config.yaml
python src/pipeline/run_pipeline.py --models xgboost,bert_mlp,early_fusion  # 仅运行指定模型
python src/pipeline/run_pipeline.py --skip-preprocessing                   # 跳过预处理（使用缓存）
```

---

> **注意**：本文档为初稿，需在团队内讨论后定稿。修改需通过 PR 审阅。
