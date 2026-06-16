"""End-to-end pipeline entry point for the house price fusion project.

工程与交付入口：
    python src/pipeline/run_pipeline.py

常用检查：
    python src/pipeline/run_pipeline.py --check-only
    python src/pipeline/run_pipeline.py --smoke
    python src/pipeline/run_pipeline.py --analysis-only

说明：
- 默认模式会调用 src/experiment_runner.py 中已有的实验流程，并使用真实
  data/processed 下的预处理结果。
- smoke 模式使用合成小样本数据，用来验证代码链路、模型接口、结果写入是否通。
- analysis-only 模式不重新训练模型，只汇总 results/ 中已有的实验日志和评估分析图。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.md"

REQUIRED_REAL_DATA_FILES = [
    PROCESSED_DIR / "florida_structured.csv",
    PROCESSED_DIR / "florida_tfidf_features.pkl",
    PROCESSED_DIR / "florida_bert_embeddings.pkl",
]

# 评估分析负责人新增的图表输出。Pipeline 不直接重画 Notebook 图，
# 但会在完整实验或 --analysis-only 后统一检查并生成汇总清单。
EVALUATION_FIGURE_GROUPS: dict[str, list[tuple[str, str]]] = {
    "baseline_evaluation": [
        ("fig_baseline_metrics.png", "单模态与融合模型的核心指标对比"),
        ("fig_baseline_modality_gap.png", "结构化、文本、融合三类模态的性能差距"),
        ("fig_baseline_pred_vs_true.png", "预测值与真实值散点图"),
        ("fig_baseline_residuals.png", "模型残差分布对比"),
        ("fig_baseline_feature_importance.png", "结构化基线模型特征重要性"),
    ],
    "error_analysis": [
        ("fig_error_boxplot.png", "不同模型绝对误差/残差箱线图"),
        ("fig_error_by_price_bin.png", "分价格区间误差分析"),
        ("fig_error_cdf.png", "误差累计分布曲线"),
        ("fig_error_correlation.png", "模型误差相关性分析"),
        ("fig_error_over_under.png", "高估与低估情况分析"),
        ("fig_error_relative_by_price.png", "相对误差随价格变化分析"),
        ("fig_error_residual_vs_price.png", "残差与真实价格关系图"),
    ],
    "ablation_study": [
        ("fig_ablation_modality.png", "结构化、文本和融合模态消融对比"),
        ("fig_ablation_waterfall.png", "融合模型相对基线的增量提升"),
        ("fig_ablation_heatmap.png", "消融实验指标热力图"),
        ("fig_ablation_significance_matrix.png", "模型差异显著性检验矩阵"),
        ("fig_ablation_learning_curves.png", "学习曲线与训练稳定性分析"),
    ],
}


def _ensure_project_on_path() -> None:
    """Make `src` importable when this file is executed as a script."""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def _relative(path: Path) -> str:
    """Return a POSIX-style path relative to the project root when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _format_missing(paths: Iterable[Path]) -> str:
    return "\n".join(f"  - {_relative(path)}" for path in paths)


def validate_layout(smoke: bool = False, analysis_only: bool = False) -> None:
    """Validate project folders and required input files before running."""
    required_dirs = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "src" / "data",
        PROJECT_ROOT / "src" / "features",
        PROJECT_ROOT / "src" / "models",
        PROJECT_ROOT / "src" / "evaluation",
        PROJECT_ROOT / "src" / "pipeline",
        PROJECT_ROOT / "config",
    ]
    missing_dirs = [path for path in required_dirs if not path.exists()]
    if missing_dirs:
        raise SystemExit(
            "项目结构不完整，缺少以下目录：\n"
            f"{_format_missing(missing_dirs)}"
        )

    if not CONFIG_PATH.exists():
        raise SystemExit(
            "缺少配置文件：config/config.yaml\n"
            "请确认当前分支是否基于最新整合分支。"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Smoke 和 analysis-only 不依赖真实 data/processed 大文件。
    if smoke or analysis_only:
        return

    missing_files = [path for path in REQUIRED_REAL_DATA_FILES if not path.exists()]
    if missing_files:
        raise SystemExit(
            "真实数据模式需要先准备 data/processed 下的预处理结果，当前缺少：\n"
            f"{_format_missing(missing_files)}\n\n"
            "临时验证工程链路可以先运行：\n"
            "  python src/pipeline/run_pipeline.py --smoke\n\n"
            "如果只是查看已生成的评估图和实验日志，可以运行：\n"
            "  python src/pipeline/run_pipeline.py --analysis-only\n\n"
            "如果要跑完整实验，请让数据工程同学提供上述文件，或先运行数据预处理脚本。"
        )


def _safe_float(value: str | float | int | None) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result


def _format_number(value: str | float | int | None, digits: int = 2) -> str:
    number = _safe_float(value)
    if math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def _load_experiment_log(path: Path = EXPERIMENT_LOG_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _successful_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status", "").lower() == "success"]


def _find_best_test_model(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    successful = _successful_rows(rows)
    if not successful:
        return None
    return min(successful, key=lambda row: _safe_float(row.get("test_rmse")))


def _build_metrics_table(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return "未找到 `results/experiment_log.csv`，暂无法生成模型指标表。"

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("status", "").lower() != "success",
            _safe_float(row.get("test_rmse")),
        ),
    )
    lines = [
        "| Model | Modality | Test RMSE | Test MAE | Test R² | Test MAPE | Train Time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| {model} | {modality} | {rmse} | {mae} | {r2} | {mape}% | {time}s |".format(
                model=row.get("model_name", "-"),
                modality=row.get("modality", "-"),
                rmse=_format_number(row.get("test_rmse"), 2),
                mae=_format_number(row.get("test_mae"), 2),
                r2=_format_number(row.get("test_r2"), 4),
                mape=_format_number(row.get("test_mape"), 2),
                time=_format_number(row.get("train_time_sec"), 2),
            )
        )
    return "\n".join(lines)


def _collect_evaluation_figures() -> tuple[list[Path], list[Path]]:
    present: list[Path] = []
    missing: list[Path] = []
    for items in EVALUATION_FIGURE_GROUPS.values():
        for filename, _description in items:
            path = RESULTS_DIR / filename
            if path.exists():
                present.append(path)
            else:
                missing.append(path)
    return present, missing


def build_evaluation_summary() -> str:
    """Build a Markdown summary for metrics and generated analysis artifacts."""
    rows = _load_experiment_log()
    successful = _successful_rows(rows)
    best = _find_best_test_model(rows)
    present, missing = _collect_evaluation_figures()

    lines: list[str] = [
        "# Evaluation Summary",
        "",
        "该文件由 `src/pipeline/run_pipeline.py` 自动生成，用于汇总模型指标和评估分析图表。",
        "",
        "## 1. Experiment log",
        "",
        f"- Source: `{_relative(EXPERIMENT_LOG_PATH)}`",
        f"- Successful models: {len(successful)} / {len(rows)}",
    ]
    if best is not None:
        lines.extend(
            [
                "- Best test model: "
                f"`{best.get('model_name')}` "
                f"(RMSE={_format_number(best.get('test_rmse'), 2)}, "
                f"MAE={_format_number(best.get('test_mae'), 2)}, "
                f"R²={_format_number(best.get('test_r2'), 4)})",
            ]
        )
    else:
        lines.append("- Best test model: not available")

    lines.extend(
        [
            "",
            "## 2. Test-set metrics",
            "",
            _build_metrics_table(rows),
            "",
            "## 3. Evaluation-analysis figures",
            "",
            f"- Available figures: {len(present)} / {len(present) + len(missing)}",
            "",
        ]
    )

    for group_name, items in EVALUATION_FIGURE_GROUPS.items():
        lines.append(f"### {group_name}")
        lines.append("")
        for filename, description in items:
            path = RESULTS_DIR / filename
            mark = "OK" if path.exists() else "MISSING"
            lines.append(f"- [{mark}] `{_relative(path)}` - {description}")
        lines.append("")

    if missing:
        lines.extend(
            [
                "## 4. Missing artifacts",
                "",
                "以下图表当前未在 `results/` 中找到。若需要完整复现 PPT/报告中的评估分析，"
                "请先运行对应 Notebook 或评估脚本生成这些图表：",
                "",
            ]
        )
        for path in missing:
            lines.append(f"- `{_relative(path)}`")
        lines.append("")

    lines.extend(
        [
            "## 5. Notes for report/PPT",
            "",
            "- `fig_baseline_metrics.png` 和 `fig_baseline_modality_gap.png` 适合用于总体模型性能比较。",
            "- `fig_baseline_pred_vs_true.png`、`fig_baseline_residuals.png` 和错误分析图适合解释模型误差来源。",
            "- `fig_ablation_*` 系列图适合支撑“文本模态是否带来增量价值”和“三种融合策略对比”的结论。",
            "- MAPE 在房价长尾分布和低价异常样本下容易被放大，报告分析应优先参考 RMSE、MAE 和 R²。",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_summary(require_all_figures: bool = False) -> Path:
    """Write results/evaluation_summary.md and print an artifact overview."""
    summary = build_evaluation_summary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_SUMMARY_PATH.write_text(summary, encoding="utf-8")

    present, missing = _collect_evaluation_figures()
    print("\n" + "=" * 72)
    print("Evaluation analysis summary")
    print("=" * 72)
    if EXPERIMENT_LOG_PATH.exists():
        print(f"Experiment log: {_relative(EXPERIMENT_LOG_PATH)}")
    else:
        print("Experiment log: MISSING")
    print(f"Figures available: {len(present)} / {len(present) + len(missing)}")
    print(f"Summary saved: {_relative(EVALUATION_SUMMARY_PATH)}")

    if missing:
        print("\nMissing evaluation figures:")
        for path in missing:
            print(f"  - {_relative(path)}")
        if require_all_figures:
            raise SystemExit(
                "评估分析图不完整。请先生成缺失图表，或去掉 --require-figures。"
            )
    return EVALUATION_SUMMARY_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-source house price prediction pipeline."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用合成小样本数据跑通端到端流程，不依赖 Kaggle 原始/处理后数据。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查项目结构和必要数据文件，不训练模型。",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="不重新训练模型，只汇总 results/ 中已有实验日志和评估分析图。",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="训练结束后不生成 results/evaluation_summary.md。",
    )
    parser.add_argument(
        "--require-figures",
        action="store_true",
        help="生成评估汇总时，如果评估分析图缺失则报错退出。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.smoke and args.analysis_only:
        raise SystemExit("--smoke 和 --analysis-only 不能同时使用。")
    if args.check_only and args.analysis_only:
        raise SystemExit("--check-only 和 --analysis-only 不能同时使用。")

    print("=" * 72)
    print("Multi-source House Price Prediction Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    if args.analysis_only:
        mode = "analysis only"
    else:
        mode = "smoke" if args.smoke else "real data"
    print(f"Mode: {mode}")
    print("=" * 72)

    validate_layout(smoke=args.smoke, analysis_only=args.analysis_only)

    if args.check_only:
        print("项目结构检查通过。")
        if args.smoke:
            print("Smoke 模式不需要真实数据文件。")
        else:
            print("真实数据文件检查通过。")
        return None

    if args.analysis_only:
        return write_evaluation_summary(require_all_figures=args.require_figures)

    _ensure_project_on_path()

    # 复用算法/评估同学已经实现的统一实验流程，避免复制训练逻辑。
    from src.experiment_runner import main as run_experiments

    experiment_args = ["--smoke"] if args.smoke else []
    results = run_experiments(experiment_args)

    if not args.skip_analysis:
        write_evaluation_summary(require_all_figures=args.require_figures)

    return results


if __name__ == "__main__":
    main()
