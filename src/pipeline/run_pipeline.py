"""End-to-end pipeline entry point for the house price fusion project.

工程与交付入口：
    python src/pipeline/run_pipeline.py

常用检查：
    python src/pipeline/run_pipeline.py --check-only
    python src/pipeline/run_pipeline.py --smoke

说明：
- 默认模式会调用 src/experiment_runner.py 中已有的实验流程，并使用真实
  data/processed 下的预处理结果。
- smoke 模式使用合成小样本数据，用来验证代码链路、模型接口、结果写入是否通。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_REAL_DATA_FILES = [
    PROCESSED_DIR / "florida_structured.csv",
    PROCESSED_DIR / "florida_tfidf_features.pkl",
    PROCESSED_DIR / "florida_bert_embeddings.pkl",
]


def _ensure_project_on_path() -> None:
    """Make `src` importable when this file is executed as a script."""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def _format_missing(paths: Iterable[Path]) -> str:
    return "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in paths)


def validate_layout(smoke: bool = False) -> None:
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

    if smoke:
        return

    missing_files = [path for path in REQUIRED_REAL_DATA_FILES if not path.exists()]
    if missing_files:
        raise SystemExit(
            "真实数据模式需要先准备 data/processed 下的预处理结果，当前缺少：\n"
            f"{_format_missing(missing_files)}\n\n"
            "临时验证工程链路可以先运行：\n"
            "  python src/pipeline/run_pipeline.py --smoke\n\n"
            "如果要跑完整实验，请让数据工程同学提供上述文件，或先运行数据预处理脚本。"
        )


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    print("=" * 72)
    print("Multi-source House Price Prediction Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {'smoke' if args.smoke else 'real data'}")
    print("=" * 72)

    validate_layout(smoke=args.smoke)

    if args.check_only:
        print("项目结构检查通过。")
        if args.smoke:
            print("Smoke 模式不需要真实数据文件。")
        else:
            print("真实数据文件检查通过。")
        return None

    _ensure_project_on_path()

    # 复用算法/评估同学已经实现的统一实验流程，避免复制训练逻辑。
    from src.experiment_runner import main as run_experiments

    experiment_args = ["--smoke"] if args.smoke else []
    return run_experiments(experiment_args)


if __name__ == "__main__":
    main()
