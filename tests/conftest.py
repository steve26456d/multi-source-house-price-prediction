"""
pytest 公共 fixtures 和配置。

使用方法：
    cd multi-source-house-price-prediction
    pytest tests/ -v
"""

import pytest
import numpy as np


@pytest.fixture
def sample_structured_data():
    """生成模拟结构化数据用于测试。"""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    x = np.random.randn(n_samples, n_features)
    y = 100000 + 50000 * x[:, 0] + np.random.randn(n_samples) * 10000
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return x, y, feature_names


@pytest.fixture
def sample_text_embeddings():
    """生成模拟文本嵌入向量用于测试。"""
    np.random.seed(42)
    n_samples = 100
    return {
        "tfidf": np.random.randn(n_samples, 150),
        "bert_embeddings": np.random.randn(n_samples, 768),
    }


@pytest.fixture
def sample_config():
    """加载测试用配置"""
    return {
        "data": {
            "florida": {
                "target_column": "sale_price",
                "text_column": "text_clean",
            }
        },
        "training": {"random_seed": 42, "test_size": 0.1, "val_size": 0.1},
    }
