# 石韫嘉评估结果提交

负责人：石韫嘉

分支：`shiyunjia-evaluation-on-algorithm`

运行命令：

```bash
conda run -n text004 python src/experiment_runner.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `experiment_log.csv` | 完整实验日志，包含 train/val/test 的 RMSE、MAE、R2、MAPE 和训练耗时 |
| `model_ranking.csv` | 按 test RMSE 升序排列的模型结果表，适合直接放入报告 |

## 数据说明

本次结果基于群里提供的 processed 数据运行：

```text
data/processed/florida_structured.csv
data/processed/florida_tfidf_features.pkl
data/processed/florida_bert_embeddings.pkl
```

数据未打包进本提交包，避免重复发送大文件。

## 实验规模

```text
Total samples: 10893
Train: 8714
Validation: 1089
Test: 1090
Models: 9
Status: all success
```

## 最佳模型

Test RMSE 最优模型：

```text
EarlyFusionMLP
RMSE = 110786.79
MAE  = 73940.12
R2   = 0.8357
MAPE = 232.12%
```

## 简要结论

从 test RMSE 和 R2 看，早期融合 MLP 是本次实验中表现最好的模型，优于结构化单模态基线和文本单模态基线。中期融合模型排名第二，说明融合文本模态后整体预测性能有提升。

MAPE 数值偏高，可能受低价样本或价格分布长尾影响。正式报告中建议重点使用 RMSE、MAE、R2，并在错误分析部分单独解释 MAPE。
