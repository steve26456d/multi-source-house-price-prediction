# Feature Engineering Document

## 多源数据融合挖掘——房价预测项目

## 特征工程文档

负责人：陈晓英
周次：W14
对应任务：结构化数据预处理 Pipeline、文本数据预处理 Pipeline
文档路径：`doc/feature_engineering.md`

---

## 1. 文档说明

本文档用于说明房价预测项目中 W14 阶段的特征工程设计与实现方法，主要包括结构化特征处理和文本特征处理两部分。

本阶段的目标是将 Florida Real Estate Sold Dataset 2026 中的原始结构化字段和文本描述字段转换为后续模型可以直接使用的数值特征，为 W15 阶段的单模态基线模型和多模态融合模型提供输入。

本阶段对应代码文件包括：

```text
src/data/preprocess_structured.py
src/data/preprocess_text.py
```

主要输出包括：

```text
data/processed/florida_structured.csv
data/processed/florida_clean_text.csv
data/processed/florida_tfidf_features.pkl
data/processed/tfidf_vectorizer.pkl
data/processed/svd_model.pkl
data/processed/florida_bert_embeddings.pkl
```

---

## 2. 数据源说明

本项目采用 Ames Housing Dataset 和 Florida Real Estate Sold Dataset 2026 两个数据源开展房价预测实验。

其中：

Ames Housing Dataset 仅包含结构化房屋属性特征，主要用于后续结构化基线模型构建和算法效果对比。
根据项目分工，W13 阶段已完成 Ames 数据集下载、验证和初步 EDA 分析。
由于 Ames 数据集不包含文本描述字段，因此 W14 阶段未针对其开发单独的文本预处理 Pipeline。
Florida 数据集同时包含结构化字段和房源描述文本，是本项目多模态融合实验的核心数据集。
W14 阶段完成的结构化预处理、文本预处理、TF-IDF 特征提取、TruncatedSVD 降维以及 BERT 嵌入向量预计算工作均基于 Florida 数据集实现。
因此，本特征工程文档后续章节主要围绕 Florida 数据集展开说明。

### 2.1 结构化字段

结构化字段主要描述房屋的基本属性、价格属性和地理属性。当前代码中重点处理的字段包括：

| 字段                | 含义          | 类型  | 处理方式            |
| ----------------- | ----------- | --- | --------------- |
| `listPrice`       | 挂牌价格        | 数值型 | 缺失值填补、异常值处理、标准化 |
| `lastSoldPrice`   | 最终成交价格      | 数值型 | 缺失值填补、异常值处理     |
| `sqft`            | 房屋面积        | 数值型 | 缺失值填补、异常值处理、标准化 |
| `stories`         | 楼层数         | 数值型 | 缺失值填补、标准化       |
| `beds`            | 卧室数量        | 数值型 | 缺失值填补、标准化       |
| `baths`           | 卫生间数量       | 数值型 | 缺失值填补、标准化       |
| `baths_full`      | 完整卫生间数量     | 数值型 | 缺失值填补、标准化       |
| `baths_full_calc` | 计算后的完整卫生间数量 | 数值型 | 缺失值填补、标准化       |
| `garage`          | 车库数量        | 数值型 | 缺失值填补、标准化       |
| `year_built`      | 建造年份        | 数值型 | 缺失值填补、标准化       |
| `type`            | 房产类型        | 类别型 | One-Hot 编码      |
| `sub_type`        | 房产子类型       | 类别型 | One-Hot 编码      |
| `zip`             | 邮政编码        | 类别型 | One-Hot 编码      |

其中，`lastSoldPrice` 是房屋最终成交价格，可作为后续模型训练的预测目标；其余字段可作为结构化输入特征。

---

### 2.2 文本字段

文本特征主要来自 Florida 数据集中的：

```text
sanitized_text
```

该字段来源于房源描述文本，包含房屋装修、地理环境、设施情况、房型特点等非结构化信息。

例如文本中可能出现：

```text
updated kitchen
granite countertops
screened pool
waterfront
garage
new roof
```

这些描述可能与房屋售价存在关联，因此需要通过 NLP 方法转换为模型可使用的数值特征。

---

## 3. 结构化特征工程

结构化特征工程对应代码文件：

```text
src/data/preprocess_structured.py
```

输入数据：

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

输出数据：

```text
data/processed/florida_structured.csv
```

处理流程包括：

```text
缺失值填补 → IQR 异常值处理 → 类别特征编码 → 数值特征标准化
```

---

## 3.1 缺失值填补

### 3.1.1 处理目的

原始房产数据中可能存在部分字段为空，例如面积缺失、车库数量缺失、房产类型缺失等。如果直接输入模型，可能导致训练失败或模型效果下降。因此需要对缺失值进行统一处理。

### 3.1.2 处理方法

当前采用两类填补策略：

| 字段类型  | 填补方法  | 原因                        |
| ----- | ----- | ------------------------- |
| 数值型字段 | 中位数填补 | 中位数对极端值不敏感，适合房价、面积等偏态分布字段 |
| 类别型字段 | 众数填补  | 众数表示最常见类别，适合补充缺失的类别信息     |

代码逻辑：

```python
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
```

### 3.1.3 处理效果

完成后，结构化字段中的空值被填补，数据可以继续进入异常值处理、编码和标准化步骤。

---

## 3.2 IQR 异常值处理

### 3.2.1 处理目的

房产数据中容易存在极端值，例如：

* 面积特别大的豪宅；
* 挂牌价格异常高或异常低；
* 成交价格存在录入错误；
* 面积字段出现不合理数值。

这些极端值可能影响模型对普通样本的学习效果，因此需要进行异常值处理。

### 3.2.2 处理字段

当前代码中进行 IQR 异常值截尾的字段包括：

```text
listPrice
lastSoldPrice
sqft
```

这些字段与房价预测关系较强，同时也最容易出现极端值。

### 3.2.3 处理方法

采用 IQR 方法计算上下界：

```text
Q1 = 第一四分位数
Q3 = 第三四分位数
IQR = Q3 - Q1

下界 = Q1 - 1.5 × IQR
上界 = Q3 + 1.5 × IQR
```

代码中使用 `clip()` 进行截尾：

```python
df[col] = df[col].clip(
    lower=lower,
    upper=upper
)
```

### 3.2.4 处理说明

本项目采用“截尾”而不是“删除异常样本”的方式，主要原因是：

1. 保留原始样本数量，避免数据量减少。
2. 降低极端值对模型训练的干扰。
3. 避免直接删除高价房、大面积房屋这类可能真实存在的样本。
4. 保留房屋价格和面积的相对分布信息。

---

## 3.3 类别特征编码

### 3.3.1 处理目的

机器学习模型通常无法直接处理字符串类别字段，因此需要将类别字段转换为数值形式。

### 3.3.2 处理字段

当前采用 One-Hot Encoding 处理以下类别字段：

```text
type
sub_type
zip
```

其中：

| 字段         | 说明              |
| ---------- | --------------- |
| `type`     | 房产类型            |
| `sub_type` | 房产子类型           |
| `zip`      | 邮政编码，可表示粗粒度地理位置 |

### 3.3.3 处理方法

代码中使用：

```python
pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)
```

参数说明：

| 参数                         | 含义                    |
| -------------------------- | --------------------- |
| `columns=categorical_cols` | 指定需要编码的类别字段           |
| `drop_first=True`          | 删除每个类别变量的第一个虚拟变量，减少冗余 |

### 3.3.4 处理效果

One-Hot 编码后，类别字段会被转换为多个 0/1 数值字段。例如：

```text
type_Single Family
type_Condo
type_Townhouse
zip_33101
zip_33102
```

这样处理后，模型可以直接使用类别信息。

---

## 3.4 数值特征标准化

### 3.4.1 处理目的

不同数值字段的量纲差异较大。例如：

* `listPrice` 可能达到几十万或几百万；
* `sqft` 可能是几百到上万；
* `beds`、`baths` 通常是个位数；
* `year_built` 是年份数值。

如果不进行标准化，数值范围较大的字段可能在模型训练中产生更强影响。因此，本项目使用 StandardScaler 对数值字段进行标准化。

### 3.4.2 处理字段

当前标准化字段包括：

```text
listPrice
sqft
stories
beds
baths
baths_full
baths_full_calc
garage
year_built
```

### 3.4.3 处理方法

标准化公式为：

```text
z = (x - μ) / σ
```

其中：

| 符号  | 含义       |
| --- | -------- |
| `x` | 原始特征值    |
| `μ` | 字段均值     |
| `σ` | 字段标准差    |
| `z` | 标准化后的特征值 |

代码实现：

```python
scaler = StandardScaler()

df[numeric_cols] = scaler.fit_transform(
    df[numeric_cols]
)
```

### 3.4.4 处理效果

标准化后，数值字段整体均值接近 0，标准差接近 1，有利于后续线性模型、神经网络模型和距离相关模型训练。

---

## 4. 文本特征工程

文本特征工程对应代码文件：

```text
src/data/preprocess_text.py
```

输入数据：

```text
data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv
```

输出数据：

```text
data/processed/florida_clean_text.csv
data/processed/florida_tfidf_features.pkl
data/processed/tfidf_vectorizer.pkl
data/processed/svd_model.pkl
data/processed/florida_bert_embeddings.pkl
```

处理流程包括：

```text
文本字段验证 → 文本清洗 → 英文分词 → 去停用词 → TF-IDF → TruncatedSVD → BERT 向量预计算
```

---

## 4.1 文本字段验证

### 4.1.1 处理目的

为了保证文本处理 Pipeline 能够稳定运行，需要先检查数据集中是否存在指定文本字段。

当前代码检查字段为：

```text
sanitized_text
```

代码实现：

```python
if "sanitized_text" not in df.columns:
    raise ValueError(
        "数据集中缺少 sanitized_text 字段"
    )
```

### 4.1.2 说明

项目计划书中提到的文本字段为 `text_clean` 或已清洗文本字段。实际数据集中当前使用的是 `sanitized_text` 字段。该字段同样属于预清洗和脱敏后的房源描述文本，因此可作为文本特征来源。

为了保持代码与数据一致，当前程序以 `sanitized_text` 作为输入，并在处理后生成新的：

```text
clean_text
```

字段。

---

## 4.2 文本清洗

### 4.2.1 处理目的

原始房源描述文本通常包含大小写混合、标点、数字、缩写、停用词等内容。直接使用原始文本会增加噪声，因此需要进行清洗。

### 4.2.2 处理步骤

当前 `clean_text()` 函数完成以下处理：

1. 缺失文本处理
   如果文本为空，则返回空字符串。

2. 转小写
   将文本统一转换为小写，避免同一词因大小写不同被视为不同特征。

3. 英文分词
   使用 NLTK 的 `word_tokenize()` 对英文文本进行分词。

4. 去停用词
   使用 NLTK 英文停用词表，去除 `the`、`is`、`and`、`with` 等语义区分度较低的词。

5. 去除非字母 token
   仅保留纯英文单词，去除数字、标点符号和特殊字符。

代码核心逻辑：

```python
tokens = word_tokenize(
    text
)

stop_words = set(
    stopwords.words(
        "english"
    )
)

tokens = [
    token
    for token in tokens
    if token.isalpha()
    and token not in stop_words
]
```

### 4.2.3 处理效果示例

清洗前：

```text
Beautifully updated 3BR/2BA home with new kitchen, granite countertops!
```

清洗后：

```text
beautifully updated home new kitchen granite countertops
```

该步骤可以减少文本噪声，使后续 TF-IDF 和 BERT 特征提取更加稳定。

---

## 4.3 TF-IDF 特征提取

### 4.3.1 处理目的

TF-IDF 用于衡量词语在文本中的重要程度。对于房源描述文本，TF-IDF 可以突出一些与房价相关的关键词，例如：

```text
pool
waterfront
renovated
garage
kitchen
granite
new
luxury
```

这些关键词可能反映房屋设施、装修情况、地理环境和房屋品质。

### 4.3.2 参数设置

当前代码使用：

```python
TfidfVectorizer(
    max_features=5000
)
```

参数说明：

| 参数             | 设置     | 说明              |
| -------------- | ------ | --------------- |
| `max_features` | `5000` | 最多保留 5000 个重要词项 |

### 4.3.3 处理结果

TF-IDF 会将每条文本转换为一个高维稀疏向量。每一维对应一个词项，每个数值表示该词在当前文本中的重要程度。

由于 TF-IDF 特征维度较高，后续需要使用 TruncatedSVD 进行降维。

---

## 4.4 TruncatedSVD 降维

### 4.4.1 处理目的

TF-IDF 特征通常具有高维、稀疏的特点。如果直接用于模型训练，可能出现以下问题：

1. 特征维度过高；
2. 训练速度较慢；
3. 数据稀疏；
4. 容易产生过拟合；
5. 不便于与结构化特征进行拼接。

因此，本项目使用 TruncatedSVD 对 TF-IDF 特征进行降维。

### 4.4.2 参数设置

当前代码使用：

```python
TruncatedSVD(
    n_components=128,
    random_state=42
)
```

参数说明：

| 参数             | 设置    | 说明                   |
| -------------- | ----- | -------------------- |
| `n_components` | `128` | 将 TF-IDF 特征压缩为 128 维 |
| `random_state` | `42`  | 保证实验结果可复现            |

### 4.4.3 维度说明

项目计划中要求将 TF-IDF 特征降维至 100—200 维。当前设置为：

```text
128维
```

符合计划要求。

### 4.4.4 输出文件

```text
data/processed/florida_tfidf_features.pkl
data/processed/svd_model.pkl
```

其中：

| 文件                           | 说明                      |
| ---------------------------- | ----------------------- |
| `florida_tfidf_features.pkl` | 降维后的 128 维文本特征          |
| `svd_model.pkl`              | 已训练的 SVD 降维模型，可用于后续数据转换 |

---

## 4.5 BERT 嵌入向量预计算

### 4.5.1 处理目的

TF-IDF 主要关注词频信息，难以充分表达上下文语义。因此，本项目额外使用 BERT 提取文本语义向量。

BERT 可以捕捉文本中更深层的上下文信息，例如房源描述中“new kitchen”“screened pool”“waterfront view”等短语的语义特征。

### 4.5.2 模型设置

当前使用模型：

```text
bert-base-uncased
```

该模型适合英文文本处理，并且输出维度固定，便于后续建模。

### 4.5.3 参数设置

当前代码设置：

```text
batch_size = 32
max_length = 128
truncation = True
padding = True
```

参数说明：

| 参数           | 设置     | 说明                    |
| ------------ | ------ | --------------------- |
| `batch_size` | `32`   | 每批处理 32 条文本，提高处理效率    |
| `max_length` | `128`  | 文本最大长度限制为 128 个 token |
| `truncation` | `True` | 超长文本自动截断              |
| `padding`    | `True` | 短文本自动补齐，便于批量输入        |

### 4.5.4 向量提取方式

当前代码使用 BERT 最后一层隐藏状态中的 `[CLS]` 向量作为整条文本的表示：

```python
outputs.last_hidden_state[:, 0, :]
```

输出维度为：

```text
768维
```

### 4.5.5 输出文件

```text
data/processed/florida_bert_embeddings.pkl
```

该文件保存所有样本对应的 BERT 文本嵌入向量，可用于文本基线模型或多模态融合模型。

---

## 5. 最终特征输出说明

W14 阶段特征工程最终输出如下：

| 输出文件                          | 特征类型              | 维度/形式 | 用途              |
| ----------------------------- | ----------------- | ----- | --------------- |
| `florida_structured.csv`      | 结构化特征             | 表格型特征 | 结构化基线模型、多模态融合   |
| `florida_clean_text.csv`      | 清洗文本              | 单列文本  | 文本清洗结果检查        |
| `florida_tfidf_features.pkl`  | TF-IDF + SVD 文本特征 | 128 维 | 文本基线模型、早期融合     |
| `tfidf_vectorizer.pkl`        | TF-IDF 模型         | 模型对象  | 后续新文本转换         |
| `svd_model.pkl`               | SVD 模型            | 模型对象  | 保持文本特征降维一致      |
| `florida_bert_embeddings.pkl` | BERT 文本特征         | 768 维 | BERT 文本模型、多模态融合 |

---

## 6. 后续使用方式

本阶段输出的特征将用于 W15 阶段的模型训练和融合实验。

### 6.1 结构化基线模型

输入：

```text
florida_structured.csv
```

可用于：

```text
Linear Regression
Random Forest
XGBoost
LightGBM
```

目的：

```text
评估结构化字段对房价预测的基础效果。
```

---

### 6.2 文本基线模型

输入：

```text
florida_tfidf_features.pkl
florida_bert_embeddings.pkl
```

可用于：

```text
TF-IDF + Ridge
BERT + MLP
```

目的：

```text
评估房源描述文本对房价预测的独立贡献。
```

---

### 6.3 多模态融合模型

输入：

```text
结构化特征 + TF-IDF-SVD 特征
结构化特征 + BERT 特征
```

融合方式：

```text
早期融合：特征直接拼接
中期融合：结构化分支与文本分支分别编码后融合
晚期融合：不同模型预测结果加权或 Stacking
```

目的：

```text
比较单一模态和多模态融合在房价预测任务中的性能差异。
```

---

## 7. 当前完成情况

截至 W14，特征工程部分完成情况如下：

| 任务                      | 状态  |
| ----------------------- | --- |
| 结构化数据读取                 | 已完成 |
| 结构化缺失值填补                | 已完成 |
| IQR 异常值处理               | 已完成 |
| 类别特征 One-Hot 编码         | 已完成 |
| 数值特征 StandardScaler 标准化 | 已完成 |
| 文本字段验证                  | 已完成 |
| 文本清洗                    | 已完成 |
| 英文分词                    | 已完成 |
| 去停用词                    | 已完成 |
| TF-IDF 特征提取             | 已完成 |
| TruncatedSVD 降维         | 已完成 |
| BERT 向量预计算              | 已完成 |
| 特征文件保存                  | 已完成 |

---

## 8. 小结

本阶段完成了房价预测项目中结构化特征和文本特征的工程化处理。结构化数据方面，完成了缺失值填补、异常值截尾、类别编码和数值标准化；文本数据方面，完成了字段验证、文本清洗、英文分词、去停用词、TF-IDF 特征提取、SVD 降维和 BERT 向量预计算。

最终生成的结构化特征、128 维 TF-IDF-SVD 文本特征和 768 维 BERT 文本特征，可以直接支持后续的单模态基线模型、多模态融合模型和消融实验。
