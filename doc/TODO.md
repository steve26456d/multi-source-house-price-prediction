# 团队分工 TODO 清单

> 项目：多源数据融合挖掘——房价预测
>
> 最后更新：第 13 周

---

## 一、全局协作 TODO（第 13 周，全员）

### 1.1 项目启动

- [ ] **选题确认**：全员讨论确认最终选题方向
  - 负责人：全员
  - 截止：W13
  - 产出：项目计划书 ✅（已完成）
- [ ] **GitHub 仓库就绪**：创建仓库、配置分支保护、添加协作者
  - 负责人：潘志强
  - 截止：W13
- [ ] **开发环境配置**：确保全员可运行 `conda env create -f environment.yaml`
  - 负责人：全员
  - 截止：W13
- [ ] **接口规范制定**：确定 DataFrame Schema、模型 fit/predict 签名
  - 负责人：孙钰淼（主持）、全员参与
  - 截止：W13
  - 产出：`doc/interface_spec.md`

### 1.2 数据探索

- [ ] **Ames 数据集下载与 EDA**
  - 负责人：陈晓英
  - 截止：W13
  - 产出：`notebooks/01_ames_eda.ipynb`
- [ ] **Florida 数据集下载与 EDA**
  - 负责人：陈晓英
  - 截止：W13
  - 产出：`notebooks/02_florida_eda.ipynb`
- [ ] **数据对齐方案设计**
  - 负责人：陈晓英（主）、孙钰淼（审阅）
  - 截止：W13
  - 产出：`doc/data_alignment.md`

---

## 二、数据工程负责人 —— 陈晓英

### 核心职责

多源数据采集与清洗、结构化/文本数据预处理、数据融合对齐与特征工程

### W13 任务

- [ ] **数据下载与验证**：下载 Ames 和 Florida 数据集，验证数据完整性
  - 产出：原始数据存放于 `data/raw/`
- [ ] **初步 EDA 报告**：缺失值统计、特征分布、文本长度分布等
  - 产出：`notebooks/01_ames_eda.ipynb`、`notebooks/02_florida_eda.ipynb`
- [ ] **数据对齐方案文档**：
  - 产出：`doc/data_alignment.md`

### W14 任务

- [ ] **结构化数据预处理 Pipeline**
  - 缺失值填补（数值型→中位数、分类型→众数）
  - 异常值检测与截尾（IQR 方法）
  - 分类特征 One-Hot / Label Encoding
  - 数值特征 StandardScaler 标准化
  - 产出：`src/data/preprocess_structured.py`
- [ ] **文本数据预处理 Pipeline**
  - `text_clean` 字段加载与验证
  - 英文分词（spaCy / NLTK）+ 去停用词
  - TF-IDF 特征提取 + TruncatedSVD 降维（100–200 维）
  - BERT 嵌入向量预计算
  - 产出：`src/data/preprocess_text.py`
- [ ] **特征工程文档**
  - 产出：`doc/feature_engineering.md`
- [ ] **数据血缘图**
  - 产出：`doc/data_lineage.md`

### W15 任务

- [ ] **特征融合接口实现**
  - 早期融合：结构化 + 文本向量拼接
  - 产出：`src/features/fusion.py`（`concat_features` 函数）
- [ ] **训练/验证/测试划分**
  - Florida：按时序划分（80%/10%/10%）
  - Ames：随机划分
  - 产出：`src/data/split.py`

### W16 任务

- [ ] **数据模块文档完善**
- [ ] **数据预处理代码 review 与整合**

---

## 三、算法负责人（建模）—— 孙钰淼

### 核心职责

多模态融合模型设计、各模型实现与调优、对比实验设计

### W13 任务

- [ ] **接口规范文档**：定义所有模型的统一接口
  - 产出：`doc/interface_spec.md`
- [ ] **文献调研**：BERT-fusion、双塔模型、注意力融合相关论文
  - 产出：`doc/literature_notes.md`

### W14 任务

- [ ] **结构化基线模型**
  - Linear Regression（性能下界）
  - Random Forest（强基线）
  - XGBoost（结构化 SOTA 基线）
  - 产出：`src/models/structured_baseline.py`
- [ ] **文本基线模型**
  - TF-IDF + Ridge Regression
  - BERT 嵌入 + MLP
  - 产出：`src/models/text_baseline.py`
- [ ] **单模态基线实验结果**
  - 产出：`notebooks/03_baseline_results.ipynb`

### W15 任务

- [ ] **早期融合模型**
  - XGBoost（拼接特征）
  - MLP（拼接特征）
  - 产出：`src/models/early_fusion.py`
- [ ] **中期融合模型**
  - 双塔结构：结构化 FC 编码 + 文本 BERT 编码
  - 注意力融合层
  - 产出：`src/models/mid_fusion.py`
- [ ] **晚期融合模型**
  - Stacking 集成（各模态独立预测 → 元学习器加权）
  - 产出：`src/models/late_fusion.py`
- [ ] **超参数调优**
  - 使用 Optuna 或 GridSearchCV
  - 产出：`src/models/hyperparam_tuning.py`
- [ ] **完整实验记录表**
  - 所有模型在各指标上的对比
  - 产出：`results/experiment_log.csv`

### W16 任务

- [ ] **模型对比分析报告**
- [ ] **模型代码 module 化整理**

---

## 四、评估与分析负责人 —— 石韫嘉

### 核心职责

评估指标体系设计、消融实验与显著性检验、结果可视化与错误分析

### W13 任务

- [x] **评估指标体系设计文档**
  - 明确 RMSE、MAE、R²、MAPE 的计算方式
  - 产出：`doc/evaluation_metrics.md`
- [ ] **评估代码框架搭建**
  - 产出：`src/evaluation/metrics.py`
  - 说明：`docs` 分支不提交代码，该项需在代码开发分支完成

### W14 任务

- [ ] **单模态基线评估**
  - 结构化模型 vs. 文本模型的预测性能对比
  - 产出：`notebooks/04_evaluation_baseline.ipynb`
- [ ] **评估可视化函数**
  - 预测值 vs 真实值散点图
  - 残差分布图
  - 特征重要性图（树模型）
  - 产出：`src/evaluation/visualization.py`

### W15 任务

- [ ] **消融实验**
  - 仅结构化 vs. 仅文本 vs. 融合
  - 各模态增量贡献量化
  - 产出：`notebooks/05_ablation_study.ipynb`
- [ ] **显著性检验**
  - 配对 t-test / Wilcoxon 检验各模型差异是否显著
  - 产出：`src/evaluation/significance_test.py`
- [ ] **错误分析**
  - 高误差样本分析
  - 按价格区间分层的误差分析
  - 产出：`notebooks/06_error_analysis.ipynb`

### W16 任务

- [ ] **分析报告章节撰写**
  - 产出：`doc/analysis_report.md`
  - 说明：已提交章节草稿，待真实实验结果产出后补充表格、图表和结论
- [ ] **最终图表制作**（用于报告和答辩 PPT）

---

## 五、工程与交付负责人 —— 潘志强

### 核心职责

代码整合与模块化、Pipeline 构建、报告/演示材料制作

### W13 任务

- [ ] **GitHub 仓库创建与配置**
  - 创建仓库、分支规则、CI 配置
  - 产出：GitHub 仓库就绪
- [ ] **项目框架搭建**
  - 目录结构、配置文件、依赖管理
  - 产出：`requirements.txt`、`environment.yaml`、`.gitignore` 等
- [ ] **配置文件模板**
  - 产出：`config/config.yaml`、`.env.example`

### W14 任务

- [ ] **代码风格规范文档**
  - 产出：`doc/coding_standard.md`
- [ ] **Git 工作流文档**
  - 分支策略、PR 模板、commit 规范
  - 产出：`doc/git_workflow.md`

### W15 任务

- [ ] **端到端 Pipeline 脚本**
  - 一键运行：数据预处理 → 特征工程 → 模型训练 → 评估
  - 产出：`src/pipeline/run_pipeline.py`
- [ ] **配置文件加载模块**
  - 产出：`src/utils/config.py`

### W16 任务

- [ ] **代码模块化整合**
  - 确保所有模块接口一致、可独立运行
  - 产出：`src/pipeline/` 下的集成代码
- [ ] **README 文档完善**
  - 使用说明、API 文档、贡献指南
- [ ] **单元测试**
  - 产出：`tests/` 下的测试用例

### W17 任务

- [ ] **答辩 PPT 制作**
  - 负责人：潘志强（主）、全员提供素材
  - 产出：`presentation/` 目录
- [ ] **最终报告排版定稿**
- [ ] **演示视频录制**

---

## 六、每周甘特式进度一览

```
                         W13    W14    W15    W16    W17
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
多源数据采集与预处理       ██     ██
特征工程与基线模型                ██
融合策略实现与调优                       ██
评估分析与可视化                ██     ██     ██
代码整合与 Pipeline                    ██     ██
报告撰写与答辩准备                                  ██     ██
```

---

## 七、里程碑检查清单

- [ ] **W13 里程碑**：项目计划书 ✅ | 数据 EDA 报告 | GitHub 仓库就绪
- [ ] **W14 里程碑**：多源数据预处理 Pipeline | 单模态基线结果
- [ ] **W15 里程碑**：融合模型代码 | 完整实验记录 | 消融实验结果
- [ ] **W16 里程碑**：报告初稿 | 可运行 Pipeline | README 文档
- [ ] **W17 里程碑**：最终报告 | 答辩 PPT | 演示视频

---

## 八、接口规范待办

> 详细信息见 `doc/interface_spec.md`（孙钰淼 负责起草）

- 数据接口：统一 DataFrame Schema（列名、类型、缺失值约定）
- 模型接口：
  ```python
  class BaseModel:
      def fit(self, X, y, **kwargs) -> "BaseModel": ...
      def predict(self, X) -> np.ndarray: ...
      def save(self, path: str) -> None: ...
      @classmethod
      def load(cls, path: str) -> "BaseModel": ...
  ```
- 评估接口：
  ```python
  def evaluate(y_true, y_pred) -> Dict[str, float]: ...
  ```

---

> 本 TODO 由全体团队成员共同讨论制定，随项目进展持续更新。
>
> 开始日期：第 13 周 | 最终交付：第 17 周
