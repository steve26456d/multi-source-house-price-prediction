# Data Lineage Document

## 多源数据融合挖掘——房价预测项目

## 数据血缘图文档

负责人：陈晓英
周次：W14
对应任务：多源数据采集与清洗、数据预处理、数据融合对齐
文档路径：`doc/data_lineage.md`

---

## 1. 文档说明

本文档用于记录房价预测项目中数据从原始数据文件到模型输入特征文件的完整流转关系，即数据血缘关系。本文档重点说明：

1. 原始数据来自哪里；
2. 数据经过哪些脚本处理；
3. 每一步产生哪些中间文件或结果；
4. 每个输出文件在后续模型训练中如何使用；
5. 结构化特征与文本特征如何保持样本对齐。

---

## 2. 数据来源层

本项目主要涉及两个数据源。

### 2.1 Ames Housing Dataset

数据用途：

```text
结构化基线模型参考数据
```

主要内容：

```text
房屋面积
房间数量
建造年份
装修质量
地段评级
公共设施
SalePrice
```

数据说明：

Ames 数据集只包含结构化特征，主要用于构建传统房价预测基线模型，不参与 Florida 文本模态实验。

---

### 2.2 Florida Real Estate Sold Dataset 2026

数据用途：

```text
多源数据融合建模核心数据
```

原始数据路径：

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

该数据集同时包含：

```text
结构化房产字段
文本描述字段
成交价格标签
```

因此，Florida 数据集是本项目 W14 数据工程工作的主要处理对象。

---

## 3. 总体数据血缘图

```text
L0 原始数据层
│
├── Ames Housing Dataset
│       │
│       └── 结构化特征
│               │
│               └── 结构化基线模型参考
│
└── Florida Real Estate Sold Dataset 2026
        │
        ├── preprocess_structured.py
        │       │
        │       ├── 缺失值填补
        │       ├── IQR 异常值截尾
        │       ├── One-Hot 编码
        │       ├── StandardScaler 标准化
        │       │
        │       └── data/processed/florida_structured.csv
        │
        └── preprocess_text.py
                │
                ├── sanitized_text 字段验证
                ├── 文本清洗
                ├── 英文分词
                ├── 去停用词
                │
                ├── data/processed/florida_clean_text.csv
                │
                ├── TF-IDF 特征提取
                │       └── data/processed/tfidf_vectorizer.pkl
                │
                ├── TruncatedSVD 降维
                │       ├── data/processed/svd_model.pkl
                │       └── data/processed/florida_tfidf_features.pkl
                │
                └── BERT 向量预计算
                        └── data/processed/florida_bert_embeddings.pkl
```

---

## 4. 分层数据血缘说明

本项目当前数据流转可划分为五层：

```text
L0 原始数据层
L1 数据读取层
L2 数据预处理层
L3 特征输出层
L4 模型输入层
```

---

## 4.1 L0 原始数据层

### 4.1.1 输入文件

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

### 4.1.2 原始字段类型

| 类型   | 字段示例                                                                          | 后续流向   |
| ---- | ----------------------------------------------------------------------------- | ------ |
| 数值字段 | `listPrice`, `lastSoldPrice`, `sqft`, `beds`, `baths`, `garage`, `year_built` | 结构化预处理 |
| 类别字段 | `type`, `sub_type`, `zip`                                                     | 类别编码   |
| 文本字段 | `sanitized_text`                                                              | 文本预处理  |
| 标签字段 | `lastSoldPrice`                                                               | 房价预测目标 |

---

## 4.2 L1 数据读取层

两个预处理脚本都从同一个 Florida 原始 CSV 文件读取数据。

### 4.2.1 结构化数据读取

处理脚本：

```text
src/data/preprocess_structured.py
```

读取方式：

```python
df = pd.read_csv(RAW_DATA)
```

输入：

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

输出：

```text
原始 DataFrame
```

---

### 4.2.2 文本数据读取

处理脚本：

```text
src/data/preprocess_text.py
```

读取方式：

```python
df = pd.read_csv(RAW_DATA)
```

输入：

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

输出：

```text
包含 sanitized_text 字段的原始 DataFrame
```

---

## 4.3 L2 数据预处理层

### 4.3.1 结构化数据处理链路

结构化处理脚本：

```text
src/data/preprocess_structured.py
```

处理顺序如下：

```text
原始 DataFrame
    │
    ▼
fill_missing_values()
    │
    ▼
clip_outliers_iqr()
    │
    ▼
encode_categorical()
    │
    ▼
scale_numeric()
    │
    ▼
florida_structured.csv
```

每一步说明：

| 步骤    | 函数                      | 输入             | 输出                       |
| ----- | ----------------------- | -------------- | ------------------------ |
| 缺失值填补 | `fill_missing_values()` | 原始结构化字段        | 无缺失字段                    |
| 异常值处理 | `clip_outliers_iqr()`   | 填补后的字段         | 截尾后的字段                   |
| 类别编码  | `encode_categorical()`  | 类别字段           | One-Hot 数值字段             |
| 数值标准化 | `scale_numeric()`       | 数值字段           | 标准化后的数值字段                |
| 文件保存  | `to_csv()`              | 预处理后 DataFrame | `florida_structured.csv` |

最终输出：

```text
data/processed/florida_structured.csv
```

---

### 4.3.2 文本数据处理链路

文本处理脚本：

```text
src/data/preprocess_text.py
```

处理顺序如下：

```text
原始 DataFrame
    │
    ▼
检查 sanitized_text 字段
    │
    ▼
clean_text()
    │
    ▼
生成 clean_text 字段
    │
    ├── 保存 florida_clean_text.csv
    │
    ├── TfidfVectorizer()
    │       │
    │       ▼
    │   TF-IDF 矩阵
    │       │
    │       ▼
    │   TruncatedSVD()
    │       │
    │       ▼
    │   florida_tfidf_features.pkl
    │
    └── BERT Tokenizer + BERT Model
            │
            ▼
        florida_bert_embeddings.pkl
```

每一步说明：

| 步骤      | 函数/模型               | 输入               | 输出                                           |
| ------- | ------------------- | ---------------- | -------------------------------------------- |
| 字段验证    | 字段检查                | 原始 DataFrame     | 确认 `sanitized_text` 可用                       |
| 文本清洗    | `clean_text()`      | `sanitized_text` | `clean_text`                                 |
| 清洗文本保存  | `to_csv()`          | `clean_text`     | `florida_clean_text.csv`                     |
| TF-IDF  | `TfidfVectorizer`   | `clean_text`     | TF-IDF 矩阵、`tfidf_vectorizer.pkl`             |
| SVD 降维  | `TruncatedSVD`      | TF-IDF 矩阵        | `florida_tfidf_features.pkl`、`svd_model.pkl` |
| BERT 向量 | `bert-base-uncased` | `clean_text`     | `florida_bert_embeddings.pkl`                |

---

## 4.4 L3 特征输出层

W14 阶段最终生成以下文件：

| 输出文件                                         | 生成脚本                       | 来源字段             | 文件内容                  |
| -------------------------------------------- | -------------------------- | ---------------- | --------------------- |
| `data/processed/florida_structured.csv`      | `preprocess_structured.py` | 结构化字段            | 清洗、编码、标准化后的结构化数据      |
| `data/processed/florida_clean_text.csv`      | `preprocess_text.py`       | `sanitized_text` | 清洗后的文本                |
| `data/processed/florida_tfidf_features.pkl`  | `preprocess_text.py`       | `clean_text`     | 128 维 TF-IDF-SVD 文本特征 |
| `data/processed/tfidf_vectorizer.pkl`        | `preprocess_text.py`       | `clean_text`     | TF-IDF 向量化器           |
| `data/processed/svd_model.pkl`               | `preprocess_text.py`       | TF-IDF 矩阵        | SVD 降维模型              |
| `data/processed/florida_bert_embeddings.pkl` | `preprocess_text.py`       | `clean_text`     | 768 维 BERT 文本向量       |

---

## 4.5 L4 模型输入层

W14 生成的数据文件将进入 W15 建模阶段。

### 4.5.1 结构化模型输入

输入文件：

```text
data/processed/florida_structured.csv
```

对应模型：

```text
Linear Regression
Random Forest
XGBoost
LightGBM
```

主要用途：

```text
构建结构化单模态基线模型。
```

---

### 4.5.2 文本模型输入

输入文件：

```text
data/processed/florida_tfidf_features.pkl
data/processed/florida_bert_embeddings.pkl
```

对应模型：

```text
TF-IDF + Ridge
BERT + MLP
```

主要用途：

```text
构建文本单模态基线模型。
```

---

### 4.5.3 多模态融合模型输入

输入文件：

```text
data/processed/florida_structured.csv
data/processed/florida_tfidf_features.pkl
data/processed/florida_bert_embeddings.pkl
```

对应融合方式：

```text
早期融合：结构化特征与文本特征直接拼接
中期融合：结构化分支和文本分支分别编码后融合
晚期融合：结构化模型与文本模型预测结果集成
```

主要用途：

```text
比较结构化数据、文本数据和融合数据在房价预测任务中的性能差异。
```

---

## 5. 数据对齐关系

Florida 数据集中的结构化字段、文本字段和标签字段来自同一条房源记录，因此天然对齐。

可以表示为：

```text
第 i 条记录：
    结构化字段 X_structured[i]
    文本字段 sanitized_text[i]
    标签 lastSoldPrice[i]
```

经过预处理后，对应关系保持为：

```text
第 i 条结构化特征
对应
第 i 条 TF-IDF-SVD 文本特征
对应
第 i 条 BERT 文本向量
对应
第 i 条房价标签
```

后续进行早期融合时，可以按行拼接：

```text
X_fusion[i] = concat(
    X_structured[i],
    X_text[i]
)
```

该对齐方式避免了跨数据源匹配时可能出现的地址不一致、样本错位和标签不匹配问题。

---

## 6. 数据血缘表

| 编号 | 输入               | 处理脚本/函数                 | 输出                                           | 后续用途        |
| -- | ---------------- | ----------------------- | -------------------------------------------- | ----------- |
| 1  | Florida 原始 CSV   | `pd.read_csv()`         | 原始 DataFrame                                 | 数据预处理       |
| 2  | 原始 DataFrame     | `fill_missing_values()` | 无缺失结构化数据                                     | 异常值处理       |
| 3  | 无缺失结构化数据         | `clip_outliers_iqr()`   | 异常值截尾后的数据                                    | 类别编码        |
| 4  | 截尾后的结构化数据        | `encode_categorical()`  | One-Hot 编码数据                                 | 数值标准化       |
| 5  | 编码后的结构化数据        | `scale_numeric()`       | `florida_structured.csv`                     | 结构化模型、融合模型  |
| 6  | 原始 DataFrame     | 字段验证                    | 可用 `sanitized_text`                          | 文本清洗        |
| 7  | `sanitized_text` | `clean_text()`          | `clean_text`                                 | TF-IDF、BERT |
| 8  | `clean_text`     | `to_csv()`              | `florida_clean_text.csv`                     | 文本检查        |
| 9  | `clean_text`     | `TfidfVectorizer`       | TF-IDF 矩阵、`tfidf_vectorizer.pkl`             | SVD 降维      |
| 10 | TF-IDF 矩阵        | `TruncatedSVD`          | `florida_tfidf_features.pkl`、`svd_model.pkl` | 文本模型、融合模型   |
| 11 | `clean_text`     | `bert-base-uncased`     | `florida_bert_embeddings.pkl`                | 文本模型、融合模型   |

---

## 7. 当前数据产出状态

截至 W14，当前已完成的数据产出如下：

```text
data/processed/
│
├── florida_structured.csv
├── florida_clean_text.csv
├── florida_tfidf_features.pkl
├── tfidf_vectorizer.pkl
├── svd_model.pkl
└── florida_bert_embeddings.pkl
```

这些文件分别对应结构化特征、清洗文本、TF-IDF-SVD 文本特征、TF-IDF 模型、SVD 模型和 BERT 文本向量。

---

## 8. 当前阶段完成情况

| 任务              | 状态  |
| --------------- | --- |
| Florida 原始数据读取  | 已完成 |
| 结构化预处理 Pipeline | 已完成 |
| 文本预处理 Pipeline  | 已完成 |
| 结构化特征文件输出       | 已完成 |
| 清洗文本文件输出        | 已完成 |
| TF-IDF-SVD 特征输出 | 已完成 |
| BERT 向量输出       | 已完成 |
| 多模态样本对齐说明       | 已完成 |
| W15 模型输入准备      | 已完成 |

---

## 9. 小结

本文档梳理了房价预测项目中 Florida 数据从原始 CSV 文件到模型输入特征文件的完整数据流转关系。当前阶段中，结构化字段通过 `preprocess_structured.py` 生成 `florida_structured.csv`；文本字段通过 `preprocess_text.py` 生成清洗文本、TF-IDF-SVD 特征和 BERT 向量。

由于 Florida 数据集中的结构化字段、文本字段和房价标签来自同一条房源记录，因此后续进行多模态融合时可以直接按行对齐和拼接。这些数据产出已经能够支持 W15 阶段的结构化基线模型、文本基线模型和多模态融合模型实验。
