"""
结构化数据预处理 Pipeline

负责人：陈晓英
周次：W14

功能：
1. 缺失值填补
2. IQR异常值处理
3. 类别特征编码
4. 数值特征标准化
"""

from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler


RAW_DATA = Path(
    "data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv"
)
OUTPUT_DATA = Path("data/processed/florida_structured.csv")

def fill_missing_values(df):
    """
    缺失值填补

    数值型：
        使用中位数填补

    分类型：
        使用众数填补
    """

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    return df

def clip_outliers_iqr(df, columns):
    """
    IQR异常值处理

    保留区间：

        [Q1 - 1.5*IQR,
         Q3 + 1.5*IQR]
    """

    for col in columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(
            lower=lower,
            upper=upper
        )

    return df

def encode_categorical(df):
    """
    One-Hot编码
    """

    categorical_cols = [
        "type",
        "sub_type",
        "zip"
    ]

    return pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

def scale_numeric(df):
    """
    StandardScaler标准化

    标准化后：均值 = 0，标准差 = 1
    """

    numeric_cols = [
    "listPrice",
    "sqft",
    "stories",
    "beds",
    "baths",
    "baths_full",
    "baths_full_calc",
    "garage",
    "year_built"
]

    scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(
        df[numeric_cols]
    )

    return df

def main():

    print("开始读取数据...")

    df = pd.read_csv(RAW_DATA)

    print("开始处理缺失值...")

    df = fill_missing_values(df)

    print("开始处理异常值...")

    df = clip_outliers_iqr(
        df,
        [
            "listPrice",
            "lastSoldPrice",
            "sqft"
        ]
    )

    print("开始进行类别编码...")

    df = encode_categorical(df)

    print("开始进行特征标准化...")

    df = scale_numeric(df)

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print(f"预处理完成，文件已保存：{OUTPUT_DATA}")


if __name__ == "__main__":
    main()