# Evaluation Summary

该文件由 `src/pipeline/run_pipeline.py` 自动生成，用于汇总模型指标和评估分析图表。

## 1. Experiment log

- Source: `results/experiment_log.csv`
- Successful models: 9 / 9
- Best test model: `EarlyFusionMLP` (RMSE=109152.57, MAE=73233.57, R²=0.8405)

## 2. Test-set metrics

| Model | Modality | Test RMSE | Test MAE | Test R² | Test MAPE | Train Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EarlyFusionMLP | fusion_early | 109152.57 | 73233.57 | 0.8405 | 178.44% | 50.21s |
| MidFusionModel | fusion_mid | 117920.69 | 78979.83 | 0.8139 | 299.72% | 99.21s |
| EarlyFusionXGBoost | fusion_early | 128379.50 | 89086.50 | 0.7794 | 264.31% | 18.67s |
| LateFusionStacking | fusion_late | 138278.74 | 95443.23 | 0.7441 | 329.93% | 27.57s |
| LinearBaseline | structured | 138423.67 | 98183.64 | 0.7436 | 404.47% | 8.17s |
| XGBoostBaseline | structured | 143004.62 | 99944.31 | 0.7263 | 321.95% | 3.75s |
| RandomForestBaseline | structured | 164139.78 | 110770.61 | 0.6394 | 333.82% | 16.33s |
| TFIDFRidgeBaseline | text | 175620.82 | 132936.36 | 0.5872 | 501.76% | 0.15s |
| BERTMLPBaseline | text | 205190.80 | 153250.29 | 0.4365 | 655.23% | 64.22s |

## 3. Evaluation-analysis figures

- Available figures: 17 / 17

### baseline_evaluation

- [OK] `results/fig_baseline_metrics.png` - 单模态与融合模型的核心指标对比
- [OK] `results/fig_baseline_modality_gap.png` - 结构化、文本、融合三类模态的性能差距
- [OK] `results/fig_baseline_pred_vs_true.png` - 预测值与真实值散点图
- [OK] `results/fig_baseline_residuals.png` - 模型残差分布对比
- [OK] `results/fig_baseline_feature_importance.png` - 结构化基线模型特征重要性

### error_analysis

- [OK] `results/fig_error_boxplot.png` - 不同模型绝对误差/残差箱线图
- [OK] `results/fig_error_by_price_bin.png` - 分价格区间误差分析
- [OK] `results/fig_error_cdf.png` - 误差累计分布曲线
- [OK] `results/fig_error_correlation.png` - 模型误差相关性分析
- [OK] `results/fig_error_over_under.png` - 高估与低估情况分析
- [OK] `results/fig_error_relative_by_price.png` - 相对误差随价格变化分析
- [OK] `results/fig_error_residual_vs_price.png` - 残差与真实价格关系图

### ablation_study

- [OK] `results/fig_ablation_modality.png` - 结构化、文本和融合模态消融对比
- [OK] `results/fig_ablation_waterfall.png` - 融合模型相对基线的增量提升
- [OK] `results/fig_ablation_heatmap.png` - 消融实验指标热力图
- [OK] `results/fig_ablation_significance_matrix.png` - 模型差异显著性检验矩阵
- [OK] `results/fig_ablation_learning_curves.png` - 学习曲线与训练稳定性分析

## 5. Notes for report/PPT

- `fig_baseline_metrics.png` 和 `fig_baseline_modality_gap.png` 适合用于总体模型性能比较。
- `fig_baseline_pred_vs_true.png`、`fig_baseline_residuals.png` 和错误分析图适合解释模型误差来源。
- `fig_ablation_*` 系列图适合支撑“文本模态是否带来增量价值”和“三种融合策略对比”的结论。
- MAPE 在房价长尾分布和低价异常样本下容易被放大，报告分析应优先参考 RMSE、MAE 和 R²。
