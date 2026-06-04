"""
训练/验证/测试划分

负责人：陈晓英
周次：W15

功能：
1. Florida 数据集随机划分
2. Ames 数据集随机划分

说明：
Florida 数据集原计划采用时序划分。

但根据 W13 EDA 结果，
Florida 数据集不包含 sold_date、
transaction_date 等时间字段。

因此当前阶段采用随机划分：

训练集：80%
验证集：10%
测试集：10%
"""

from sklearn.model_selection import train_test_split


def split_florida(
    X,
    y,
    random_state=42
):
    """
    Florida 数据集随机划分

    划分比例：

        训练集：80%
        验证集：10%
        测试集：10%
    """

    (
        X_train,
        X_temp,
        y_train,
        y_temp
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    (
        X_valid,
        X_test,
        y_valid,
        y_test
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


def split_ames(
    X,
    y,
    random_state=42
):
    """
    Ames 数据集随机划分

    划分比例：

        训练集：80%
        验证集：10%
        测试集：10%
    """

    (
        X_train,
        X_temp,
        y_train,
        y_temp
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    (
        X_valid,
        X_test,
        y_valid,
        y_test
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )