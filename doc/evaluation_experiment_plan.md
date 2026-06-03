# 评估实验方案

> 负责人：石韫嘉
>
> 目标：规范单模态基线、多模态融合、消融实验、显著性检验和错误分析的评估流程

---

## 一、实验对象

本项目需要评估的模型分为三类：

| 类别 | 模型 | 模态 |
|------|------|------|
| 结构化基线 | Linear Regression | structured |
| 结构化基线 | Random Forest | structured |
| 结构化基线 | XGBoost | structured |
| 文本基线 | TF-IDF + Ridge | text |
| 文本基线 | BERT Embedding + MLP | text |
| 早期融合 | 拼接特征 + XGBoost / MLP | fusion_early |
| 中期融合 | 双塔结构 + 注意力融合 | fusion_mid |
| 晚期融合 | Stacking 集成 | fusion_late |

所有模型需要在一致的数据划分、指标体系和实验记录格式下比较。

---

## 二、评估流程

### 2.1 单模型评估

每个模型训练完成后，需要输出：

1. train / val / test 三个 split 的指标
2. test split 的逐样本预测结果
3. 模型主要参数和实验时间

指标包括：

- RMSE
- MAE
- R²
- MAPE

---

### 2.2 模型对比

模型对比表建议按以下格式组织：

| model_name | modality | split | rmse | mae | r2 | mape |
|------------|----------|-------|------|-----|----|------|
| xgboost | structured | test | 待填 | 待填 | 待填 | 待填 |
| bert_mlp | text | test | 待填 | 待填 | 待填 | 待填 |
| early_fusion_xgboost | fusion_early | test | 待填 | 待填 | 待填 | 待填 |
| mid_fusion_attention | fusion_mid | test | 待填 | 待填 | 待填 | 待填 |
| late_fusion_stacking | fusion_late | test | 待填 | 待填 | 待填 | 待填 |

排序原则：

1. 优先按 test RMSE 从低到高排序。
2. RMSE 接近时比较 MAE。
3. 若 RMSE 和 MAE 结论冲突，需要结合残差分布分析。

---

## 三、消融实验设计

### 3.1 消融目标

消融实验用于验证：

- 文本模态是否提供了结构化数据之外的信息。
- 多模态融合是否优于单模态模型。
- 不同融合策略的收益是否稳定。

### 3.2 对比组

建议设置以下实验组：

| 组别 | 输入 | 目的 |
|------|------|------|
| Structured Only | 结构化特征 | 结构化基线 |
| Text Only | 文本特征 | 文本基线 |
| Early Fusion | 结构化 + 文本拼接 | 验证简单融合效果 |
| Mid Fusion | 结构化编码 + 文本编码 + 注意力 | 验证交互式融合效果 |
| Late Fusion | 单模态预测结果集成 | 验证决策层融合效果 |

### 3.3 增益计算

以 RMSE 为例，融合模型相对结构化基线的改进比例：

```text
Improvement = (RMSE_structured - RMSE_fusion) / RMSE_structured * 100%
```

如果改进比例为正，说明融合模型降低了误差。

建议同时计算：

- 相对最佳结构化模型的改进
- 相对最佳文本模型的改进
- 相对最佳单模态模型的改进

---

## 四、显著性检验设计

### 4.1 检验目的

显著性检验用于判断模型之间的性能差异是否可靠，而不是由随机波动造成。

推荐比较：

1. 最佳结构化模型 vs 最佳融合模型
2. 最佳文本模型 vs 最佳融合模型
3. 早期融合 vs 中期融合
4. 中期融合 vs 晚期融合

### 4.2 检验数据

对每个 test 样本计算模型误差：

```text
abs_error_i = |y_i - ŷ_i|
```

两个模型在同一批样本上的 `abs_error` 构成配对样本。

### 4.3 检验方法

优先使用 Wilcoxon signed-rank test：

- 不要求误差差值严格服从正态分布。
- 适合比较两个模型在同一测试集上的逐样本误差。

可补充配对 t-test：

- 当前后模型误差差值近似正态时使用。
- 作为 Wilcoxon 的辅助结果。

显著性水平：

```text
alpha = 0.05
```

解释规则：

- `p < 0.05`：差异显著。
- `p >= 0.05`：不能认为差异显著。

---

## 五、错误分析设计

### 5.1 高误差样本分析

选择 test split 中绝对误差最高的 Top K 样本，例如 Top 20。

记录字段：

| 字段 | 说明 |
|------|------|
| sample_id | 样本 ID |
| y_true | 真实房价 |
| y_pred | 预测房价 |
| abs_error | 绝对误差 |
| ape | 相对误差 |
| text_length | 文本长度 |
| property_type | 房屋类型 |
| price_bin | 价格区间 |

分析目标：

- 是否集中在高价房？
- 是否集中在文本描述较短或异常的样本？
- 是否集中在某些房屋类型或地区？

### 5.2 分价格区间误差分析

建议按真实房价分位数划分区间：

- Q1：低价
- Q2：中低价
- Q3：中高价
- Q4：高价

每个区间计算：

- RMSE
- MAE
- MAPE
- 样本数

目的：

- 判断模型是否对高价房更不稳定。
- 判断融合模型是否改善某些价格区间的预测。

### 5.3 残差分析

残差定义：

```text
residual = y_pred - y_true
```

需要观察：

- 残差是否以 0 为中心。
- 是否存在系统性高估或低估。
- 高价区间是否出现更大的残差波动。

---

## 六、可视化清单

最终报告建议包含以下图表：

| 图表 | 作用 |
|------|------|
| 模型指标对比柱状图 | 展示不同模型 RMSE / MAE 对比 |
| 预测值 vs 真实值散点图 | 观察整体拟合情况 |
| 残差分布图 | 观察误差分布和偏态 |
| 分价格区间误差图 | 分析不同价格段稳定性 |
| 消融实验增益图 | 展示各模态和融合策略贡献 |
| 显著性检验结果表 | 说明提升是否可靠 |

---

## 七、交付物清单

| 阶段 | 交付物 | 说明 |
|------|--------|------|
| W13 | `doc/evaluation_metrics.md` | 指标体系说明 |
| W14 | `notebooks/04_evaluation_baseline.ipynb` | 单模态基线评估 |
| W14 | `src/evaluation/visualization.py` | 可视化函数 |
| W15 | `notebooks/05_ablation_study.ipynb` | 消融实验 |
| W15 | `src/evaluation/significance_test.py` | 显著性检验 |
| W15 | `notebooks/06_error_analysis.ipynb` | 错误分析 |
| W16 | `doc/analysis_report.md` | 分析报告章节 |

说明：当前文档提交在 `docs` 分支，只包含文档规划和规范，不提交代码文件。

