# 石韫嘉个人任务说明

> 角色：评估与分析负责人
>
> 来源：`README.md`、`doc/TODO.md`、`doc/interface_spec.md`

---

## 一、你的核心职责

你负责整个项目的评估与分析部分，重点不是训练模型本身，而是回答：

- 模型到底好不好？
- 文本模态是否真的带来提升？
- 早期/中期/晚期融合的差异是否显著？
- 哪些样本预测失败，失败原因是什么？
- 最终报告和答辩中如何把实验结果讲清楚？

对应产出包括评估代码、实验 notebook、消融实验、显著性检验、错误分析图表和分析报告章节。

---

## 二、近期最高优先级

### W13：先把评估标准定下来

1. 编写评估指标体系文档
   - 产出：`doc/evaluation_metrics.md`
   - 内容：RMSE、MAE、R²、MAPE 的公式、适用场景、解释方式。
   - 目标：让所有同学用同一套指标评价模型，避免后期结果不可比。

2. 搭建评估代码框架
   - 产出：`src/evaluation/metrics.py`
   - 需要实现：
     - `compute_metrics(y_true, y_pred) -> Dict[str, float]`
     - RMSE
     - MAE
     - R²
     - MAPE
   - 要和 `doc/interface_spec.md` 中的接口保持一致。

### W14：评估单模态基线

1. 做结构化模型和文本模型的基线评估
   - 产出：`notebooks/04_evaluation_baseline.ipynb`
   - 对比对象：
     - Linear Regression
     - Random Forest
     - XGBoost
     - TF-IDF + Ridge
     - BERT Embedding + MLP

2. 编写评估可视化函数
   - 产出：`src/evaluation/visualization.py`
   - 需要包含：
     - 预测值 vs 真实值散点图
     - 残差分布图
     - 树模型特征重要性图

### W15：做项目最关键的分析

1. 消融实验
   - 产出：`notebooks/05_ablation_study.ipynb`
   - 对比：
     - 仅结构化
     - 仅文本
     - 早期融合
     - 中期融合
     - 晚期融合
   - 重点回答：文本模态和融合策略分别带来了多少增益。

2. 显著性检验
   - 产出：`src/evaluation/significance_test.py`
   - 方法：
     - 配对 t-test
     - Wilcoxon signed-rank test
   - 重点回答：模型之间的指标差异是否只是随机波动。

3. 错误分析
   - 产出：`notebooks/06_error_analysis.ipynb`
   - 分析方向：
     - 高误差样本
     - 不同价格区间的误差
     - 文本长度、房屋类型、地区等因素对误差的影响

### W16：产出报告材料

1. 撰写分析报告章节
   - 产出：`doc/analysis_report.md`
   - 内容：
     - 实验设置
     - 指标结果表
     - 消融实验结论
     - 显著性检验结论
     - 错误分析

2. 制作最终图表
   - 用于最终报告和答辩 PPT。

---

## 三、你需要依赖其他同学的内容

你的工作需要等待或对接以下产出：

| 依赖对象 | 负责人 | 对你的影响 |
|---------|--------|------------|
| 数据预处理结果 | 陈晓英 | 没有干净的 `X/y/split`，无法做正式评估 |
| 模型预测结果 | 孙钰淼 | 没有各模型 `y_pred`，无法做对比、消融和显著性检验 |
| Pipeline 输出格式 | 潘志强 | 最好统一保存预测结果、指标表和图表路径 |
| 接口规范 | 孙钰淼 | 你的 `compute_metrics`、实验记录 CSV 要和规范一致 |

建议你尽早和他们约定预测结果保存格式，例如：

```text
results/predictions/{model_name}_{split}.csv
```

字段建议：

```text
sample_id,y_true,y_pred,split,model_name,modality
```

---

## 四、你现在可以马上做的事

即使其他同学还没完成模型，你也可以先做这些：

1. 写 `doc/evaluation_metrics.md`
2. 实现 `src/evaluation/metrics.py`
3. 设计 `results/experiment_log.csv` 的字段规范
4. 先用随机数组或 toy data 测试评估函数
5. 写好可视化函数的接口，等真实预测结果出来后直接接入

---

## 五、最终你需要在答辩中讲清楚的话

你负责的部分要把项目从“我们训练了很多模型”提升成“我们严谨验证了模型有效性”。

答辩中建议围绕三句话展开：

1. 我们使用 RMSE、MAE、R²、MAPE 从绝对误差、相对误差和解释能力三个角度评价模型。
2. 消融实验表明，文本模态相较纯结构化基线是否带来了稳定增益。
3. 显著性检验和错误分析说明，融合模型的提升是否可靠，以及模型在哪些样本上仍然失败。
