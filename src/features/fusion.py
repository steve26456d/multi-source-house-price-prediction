"""
特征融合接口实现

负责人：陈晓英
周次：W15

功能：
1. 早期融合
2. 结构化特征 + 文本向量拼接

产出：
    concat_features 函数
"""

import numpy as np


def concat_features(
    structured_features,
    text_features
):
    """
    将结构化特征与文本向量进行拼接

    参数：
        structured_features:
            结构化特征矩阵

        text_features:
            文本特征矩阵
            可以是 TF-IDF 特征或 BERT 向量

    返回：
        fused_features:
            拼接后的融合特征矩阵
    """

    if structured_features.shape[0] != text_features.shape[0]:
        raise ValueError(
            "结构化特征和文本特征的样本数量不一致"
        )

    fused_features = np.hstack(
        (
            structured_features,
            text_features
        )
    )

    return fused_features