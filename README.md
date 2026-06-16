# 多源数据融合挖掘——房价预测

> **同济大学 计算机科学与技术学院 · 数据分析与挖掘 期末项目**
>
> 选项 A：多源数据融合挖掘

---

## 项目简介

本项目旨在探索**多源异构数据融合**在房价预测任务中的有效性。通过整合结构化房屋特征数据与非结构化的房源描述文本数据，系统性地比较单一数据源模型与多源融合模型的性能差异，验证多模态融合策略的实际价值。

### 核心问题

- 传统房价预测仅依赖结构化数据（表格式房屋特征），忽视文本描述中蕴含的丰富信息
- 如何有效融合数值型结构化特征与文本语义特征？早期/中期/晚期融合哪种最优？
- 引入文本模态相比纯结构化基线能带来多少精度提升（消融实验）？

### 技术路线

```
数据采集 → 多源预处理 → 特征工程 → 多模态融合建模 → 模型评估 → 结果分析
```

---

## 数据来源

| 数据集 | 样本数 | 模态 | 用途 |
|--------|--------|------|------|
| [Florida Real Estate Sold 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) | 10,893 | 结构化 + 文本 | 多模态融合实验 |

---

## 融合策略

| 策略 | 方法 | 说明 |
|------|------|------|
| **早期融合** | 拼接 + XGBoost/MLP | 结构化特征与文本向量直接拼接 |
| **中期融合** | 双塔 + 注意力机制 | 结构化编码与 BERT 编码通过注意力融合 |
| **晚期融合** | Stacking 集成 | 各模态独立预测后加权集成 |

---

## 项目结构

```
multi-source-house-price-prediction/
├── README.md                       # 项目说明（本文件）
├── requirements.txt                # pip 依赖
├── environment.yaml                # conda 环境配置（如本地使用 conda）
├── .gitignore                      # Git 忽略规则
├── .env.example                    # 环境变量模板
├── config/
│   └── config.yaml                 # 项目配置文件
├── data/
│   ├── raw/                        # 原始数据（不入 git）
│   └── processed/                  # 预处理后数据
├── doc/
│   └── TODO.md                     # 团队分工 TODO
├── notebooks/                      # Jupyter 探索性分析与评估分析
├── results/                        # 实验日志、评估图表和汇总报告
├── src/
│   ├── __init__.py
│   ├── data/                       # 数据加载与预处理
│   ├── features/                   # 特征工程（TF-IDF, BERT）
│   ├── models/                     # 模型实现（基线/融合）
│   ├── evaluation/                 # 评估指标与可视化
│   ├── pipeline/                   # 端到端 Pipeline
│   └── utils/                      # 通用工具函数
└── tests/                          # 单元测试
```

---

## 团队分工

| 角色 | 成员 | 核心职责 |
|------|------|----------|
| 数据工程负责人 | 陈晓英 | 多源数据采集清洗、特征工程、数据融合对齐 |
| 算法负责人 | 孙钰淼 | 多模态融合模型设计、模型实现与调优、对比实验 |
| 评估与分析负责人 | 石韫嘉 | 评估体系设计、消融实验、可视化与错误分析 |
| 工程与交付负责人 | 潘志强 | 代码整合与模块化、Pipeline 构建、报告/答辩材料 |

### 指导教师

- 赵钦佩
- 饶卫雄

---

## 环境配置

### 方式一：Conda（推荐）

```bash
conda env create -f environment.yaml
conda activate house-price-fusion
```

### 方式二：pip

```bash
pip install -r requirements.txt
```

---
## 快速开始

### 1. 激活环境

```bash
# 配置数据路径
cp .env.example .env
# 编辑 .env 填入 Kaggle 数据集保存路径

# 激活环境
conda activate house-price-fusion
```

### 2. 准备数据

将处理后的数据文件放入：

```text
data/processed/
```

必须包含：

```text
florida_structured.csv
florida_tfidf_features.pkl
florida_bert_embeddings.pkl
```

### 3. 检查项目结构和数据

```bash
python src/pipeline/run_pipeline.py --check-only
```

通过后会显示：

```text
项目结构检查通过。
真实数据文件检查通过。
```

### 4. 运行 Smoke 测试

Smoke 模式使用小规模模拟数据，用于快速检查 Pipeline 是否能完整跑通：

```bash
python src/pipeline/run_pipeline.py --smoke
```

### 5. 运行完整实验

```bash
python src/pipeline/run_pipeline.py
```

Pipeline 会自动完成：

```text
1. 加载结构化特征、TF-IDF 特征和 BERT 文本特征
2. 划分训练集、验证集和测试集
3. 训练结构化模型、文本模型和多源融合模型
4. 计算 RMSE、MAE、R2、MAPE 等评估指标
5. 将实验结果保存到 results/experiment_log.csv
6. 检查评估分析图表并生成 results/evaluation_summary.md
```

### 6. 运行测试

```bash
python -m pytest
```

当前测试结果：

```text
14 passed
```

## Pipeline 入口

本项目的统一运行入口为：

```text
src/pipeline/run_pipeline.py
```

支持以下运行方式：

```bash
python src/pipeline/run_pipeline.py --check-only
python src/pipeline/run_pipeline.py --smoke
python src/pipeline/run_pipeline.py --analysis-only
python src/pipeline/run_pipeline.py
```

| 命令 | 作用 |
| --- | --- |
| `--check-only` | 只检查项目结构和真实数据文件，不训练模型 |
| `--smoke` | 使用模拟数据快速测试完整训练、评估、结果写入流程 |
| `--analysis-only` | 不重新训练模型，只汇总 `results/` 中已有的实验日志和评估分析图 |
| `--require-figures` | 与默认运行或 `--analysis-only` 搭配使用；若评估图缺失则报错 |
| `--skip-analysis` | 默认运行后不生成 `results/evaluation_summary.md` |
| 默认运行 | 使用真实处理后数据运行完整实验，并在结束后生成评估分析汇总 |

## 实验结果与评估分析输出

真实数据实验已成功跑通，共训练并评估 9 个模型：

```text
LinearBaseline
RandomForestBaseline
XGBoostBaseline
TFIDFRidgeBaseline
BERTMLPBaseline
EarlyFusionXGBoost
EarlyFusionMLP
MidFusionModel
LateFusionStacking
```

实验数据规模：

```text
样本数量：10893
结构化特征：882
TF-IDF 特征：128
BERT 特征：768
训练集：8714
验证集：1089
测试集：1090
```

当前 `results/experiment_log.csv` 中的测试集指标如下：

| Model | Modality | Test RMSE | Test MAE | Test R² | Test MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| EarlyFusionMLP | fusion_early | 107350.09 | 71126.47 | 0.8458 | 203.70% |
| MidFusionModel | fusion_mid | 119933.99 | 79803.65 | 0.8075 | 262.54% |
| EarlyFusionXGBoost | fusion_early | 128379.50 | 89086.50 | 0.7794 | 264.31% |
| LateFusionStacking | fusion_late | 137709.85 | 95652.37 | 0.7462 | 340.50% |
| LinearBaseline | structured | 138423.67 | 98183.64 | 0.7436 | 404.47% |
| XGBoostBaseline | structured | 143004.62 | 99944.31 | 0.7263 | 321.95% |
| RandomForestBaseline | structured | 164139.78 | 110770.61 | 0.6394 | 333.82% |
| TFIDFRidgeBaseline | text | 175620.82 | 132936.36 | 0.5872 | 501.76% |
| BERTMLPBaseline | text | 207498.53 | 153415.72 | 0.4238 | 711.95% |

最佳模型为：

```text
模型：EarlyFusionMLP
Test RMSE：107350.09
Test MAE：71126.47
Test R²：0.8458
```

结果表明，融合结构化房屋属性和文本描述特征后，模型预测效果优于单一数据源模型。文本单独建模效果有限，但作为结构化特征的补充可以显著提升预测效果。

### 评估分析图表

评估分析图统一保存在 `results/` 目录下，并可通过以下命令生成汇总清单：

```bash
python src/pipeline/run_pipeline.py --analysis-only
```

该命令会读取 `results/experiment_log.csv`，检查评估图是否齐全，并生成：

```text
results/evaluation_summary.md
```

主要评估图表包括：

| 类别 | 文件 | 说明 |
| --- | --- | --- |
| 基线性能比较 | `fig_baseline_metrics.png` | 对比 9 个模型的 RMSE、MAE、R²、MAPE |
| 模态差距分析 | `fig_baseline_modality_gap.png` | 对比结构化、文本和融合模型的整体差距 |
| 预测散点图 | `fig_baseline_pred_vs_true.png` | 展示预测值与真实值的贴合程度 |
| 残差分析 | `fig_baseline_residuals.png` | 比较不同模型的误差分布 |
| 特征重要性 | `fig_baseline_feature_importance.png` | 分析结构化模型中关键房屋属性贡献 |
| 错误分析 | `fig_error_*.png` | 从箱线图、价格区间、CDF、相对误差等角度分析误差 |
| 消融实验 | `fig_ablation_*.png` | 分析结构化模态、文本模态和三类融合策略的增量贡献 |
| 显著性检验 | `fig_ablation_significance_matrix.png` | 展示模型性能差异的统计显著性 |

对应的分析 Notebook 位于：

```text
notebooks/04_evaluation_baseline.ipynb
notebooks/05_ablation_study.ipynb
notebooks/06_error_analysis.ipynb
```

这些图表可直接用于报告和答辩 PPT 中的“评估分析”部分。若需要强制检查图表是否齐全，可以运行：

```bash
python src/pipeline/run_pipeline.py --analysis-only --require-figures
```

---

## 评估指标

| 指标 | 说明 |
|------|------|
| **RMSE** | 均方根误差，核心回归指标 |
| **MAE** | 平均绝对误差 |
| **R² Score** | 模型对房价方差的解释程度 |
| **MAPE** | 平均绝对百分比误差 |

---

## 实验设计

- **单模态基线**：仅结构化 / 仅文本
- **融合实验**：早期融合 / 中期融合 / 晚期融合
- **消融实验**：隔离各模态增量贡献
- **显著性检验**：Wilcoxon 检验，用于判断模型误差差异是否具有统计显著性
- **评估图表**：基线对比、预测散点图、残差分析、错误分析、消融实验和显著性矩阵

---

## 时间表

| 周次 | 阶段 | 主要任务 |
|------|------|----------|
| W13 | 项目启动 & 数据准备 | 选题确认、数据下载、EDA、计划书 |
| W14 | 特征工程 & 基线模型 | 数据清洗、特征提取、单模态基线 |
| W15 | 融合建模 & 深度实验 | 三种融合策略、消融实验、对比分析 |
| W16 | 报告撰写 & 代码完善 | 报告初稿、Pipeline 整合、README |
| W17 | 终稿交付 & 答辩 | 报告终稿、PPT、演示视频 |

---

## 主要参考资料

1. Kaggle Florida Real Estate Sold 2026: [Florida Real Estate Sold Dataset 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026)
2. Ahmed, E. & Moustafa, M. (2016). House price estimation from visual and textual features. *IJCNN 2016*.
3. Poursaeed, O., Matera, T., & Belongie, S. (2018). Vision-based real estate price estimation. *Machine Vision and Applications*, 29(4), 667–676.
4. Law, S., Paige, B., & Russell, C. (2019). Take a look around: Using street view and satellite images to predict house prices. *ACM TIST*, 10(5), 1–19.
5. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*.
6. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD 2016*.

---

## 协作规范

- **GitHub 分支管理**：每人在独立分支开发，通过 PR 合并到 main 分支
- **每周例会**：同步进度，讨论技术方案与数据对齐问题
- **接口规范**：第 13 周确定 DataFrame Schema 和 fit/predict 签名
- **代码风格**：遵循 PEP 8，使用 type hints；提交信息使用中文
