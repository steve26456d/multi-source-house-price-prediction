"""
文本数据预处理 Pipeline

负责人：陈晓英
周次：W14

功能：
1. 文本字段验证
2. 分词
3. 去停用词
4. TF-IDF特征提取
5. TruncatedSVD降维
"""

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModel
)

import joblib
import nltk
import pandas as pd
import torch
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.decomposition import (
    TruncatedSVD
)


# ==========================
# 文件路径
# ==========================

RAW_DATA = Path(
    "data/raw/Florida/florida_real_estate_sold_properties_ultimate.csv"
)

OUTPUT_TEXT = Path(
    "data/processed/florida_clean_text.csv"
)

OUTPUT_FEATURE = Path(
    "data/processed/florida_tfidf_features.pkl"
)

OUTPUT_VECTORIZER = Path(
    "data/processed/tfidf_vectorizer.pkl"
)

OUTPUT_SVD = Path(
    "data/processed/svd_model.pkl"
)

OUTPUT_BERT = Path(
    "data/processed/florida_bert_embeddings.pkl"
)

# ==========================
# 检查NLTK资源
# ==========================

try:
    nltk.data.find(
        "tokenizers/punkt"
    )
except LookupError:
    nltk.download(
        "punkt"
    )

try:
    nltk.data.find(
        "tokenizers/punkt_tab"
    )
except LookupError:
    nltk.download(
        "punkt_tab"
    )

try:
    nltk.data.find(
        "corpora/stopwords"
    )
except LookupError:
    nltk.download(
        "stopwords"
    )


# ==========================
# 文本清洗
# ==========================
def clean_text(text):
    """
    文本清洗

    处理内容：

    1. 转小写
    2. 分词
    3. 去停用词
    4. 去除非字母字符
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

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

    return " ".join(tokens)


def generate_bert_embeddings(texts):
    """
    BERT嵌入向量预计算

    模型：
        bert-base-uncased

    输出：
        768维向量
    """

    print("加载BERT模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "bert-base-uncased"
    )

    model.eval()

    embeddings = []

    batch_size = 32

    with torch.no_grad():

        for i in range(
            0,
            len(texts),
            batch_size
        ):

            batch_texts = texts[
                i:i + batch_size
            ].tolist()

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            outputs = model(
                **inputs
            )

            cls_embeddings = (
                outputs
                .last_hidden_state[:, 0, :]
                .cpu()
                .numpy()
            )

            embeddings.append(
                cls_embeddings
            )

            print(
                f"BERT进度: "
                f"{min(i + batch_size, len(texts))}"
                f"/{len(texts)}"
            )

    return np.vstack(
        embeddings
    )

# ==========================
# 主程序
# ==========================

def main():

    print("开始读取数据...")

    df = pd.read_csv(
        RAW_DATA
    )

    if "sanitized_text" not in df.columns:
        raise ValueError(
            "数据集中缺少 sanitized_text 字段"
        )

    print("开始清洗文本...")

    df["clean_text"] = (
        df["sanitized_text"]
        .fillna("")
        .apply(clean_text)
    )

    print("开始提取TF-IDF特征...")

    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            df["clean_text"]
        )
    )

    print("开始执行SVD降维...")

    svd = TruncatedSVD(
        n_components=128,
        random_state=42
    )

    text_features = (
        svd.fit_transform(
            tfidf_matrix
        )
    )

    print("开始生成BERT向量...")

    bert_features = (
        generate_bert_embeddings(
            df["clean_text"]
        )
    )

    OUTPUT_TEXT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("保存清洗后的文本...")

    df[
        ["clean_text"]
    ].to_csv(
        OUTPUT_TEXT,
        index=False
    )

    print("保存文本特征矩阵...")

    joblib.dump(
        text_features,
        OUTPUT_FEATURE
    )

    print("保存TF-IDF模型...")

    joblib.dump(
        vectorizer,
        OUTPUT_VECTORIZER
    )

    print("保存SVD模型...")

    joblib.dump(
        svd,
        OUTPUT_SVD
    )

    print("保存BERT向量...")

    joblib.dump(
        bert_features,
        OUTPUT_BERT
    )

    print("文本预处理完成")

    print(
        f"清洗文本：{OUTPUT_TEXT}"
    )

    print(
        f"文本特征：{OUTPUT_FEATURE}"
    )

    print(
        f"TF-IDF模型：{OUTPUT_VECTORIZER}"
    )

    print(
        f"SVD模型：{OUTPUT_SVD}"
    )

    print(
        f"BERT向量：{OUTPUT_BERT}"
    )


if __name__ == "__main__":
    main()