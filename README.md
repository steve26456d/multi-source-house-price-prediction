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
| [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | 2,930 | 结构化（79 特征） | 结构化基线 |
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
├── environment.yaml                # conda 环境配置
├── .gitignore                      # Git 忽略规则
├── .env.example                    # 环境变量模板
├── config/
│   └── config.yaml                 # 项目配置文件
├── data/
│   ├── raw/                        # 原始数据（不入 git）
│   └── processed/                  # 预处理后数据
├── doc/
│   └── TODO.md                     # 团队分工 TODO
├── notebooks/                      # Jupyter 探索性分析
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

```bash
# 1. 配置数据路径
cp .env.example .env
# 编辑 .env 填入 Kaggle 数据集保存路径

# 2. 运行完整 Pipeline
python src/pipeline/run_pipeline.py
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
- **显著性检验**：配对 t-test 或 Wilcoxon 检验

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

1. Kaggle Ames Housing Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
2. Kaggle Florida Real Estate Sold 2026: [Florida Real Estate Sold Dataset 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026)
3. Ahmed, E. & Moustafa, M. (2016). House price estimation from visual and textual features. *IJCNN 2016*.
4. Poursaeed, O., Matera, T., & Belongie, S. (2018). Vision-based real estate price estimation. *Machine Vision and Applications*, 29(4), 667–676.
5. Law, S., Paige, B., & Russell, C. (2019). Take a look around: Using street view and satellite images to predict house prices. *ACM TIST*, 10(5), 1–19.
6. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*.
7. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD 2016*.

---

## 协作规范

- **GitHub 分支管理**：每人在独立分支开发，通过 PR 合并到 main 分支
- **每周例会**：同步进度，讨论技术方案与数据对齐问题
- **接口规范**：第 13 周确定 DataFrame Schema 和 fit/predict 签名
- **代码风格**：遵循 PEP 8，使用 type hints；提交信息使用中文
