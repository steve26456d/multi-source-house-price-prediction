"""
Experiment Runner

Author: Sun Yumiao
Week: W14-W15

Functionality:
1. Load preprocessed data
2. Split into train/val/test
3. Train all models
4. Compute evaluation metrics
5. Log results to results/experiment_log.csv
"""

import csv
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_metrics
from src.models.structured_baseline import (
    LinearBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from src.models.text_baseline import (
    TFIDFRidgeBaseline,
    BERTMLPBaseline,
)
from src.models.early_fusion import (
    EarlyFusionXGBoost,
    EarlyFusionMLP,
)
from src.models.mid_fusion import MidFusionModel
from src.models.late_fusion import LateFusionStacking


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STRUCTURED_PATH = DATA_PROCESSED / "florida_structured.csv"
TFIDF_PATH = DATA_PROCESSED / "florida_tfidf_features.pkl"
BERT_PATH = DATA_PROCESSED / "florida_bert_embeddings.pkl"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"


# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load all preprocessed data

    Returns
    -------
    X_struct : np.ndarray
        Structured feature matrix.
    y : np.ndarray
        Target variable (lastSoldPrice).
    X_text : dict
        Text features containing "tfidf" and "bert_embeddings".
    """
    print("=" * 60)
    print("Loading preprocessed data...")
    print("=" * 60)

    # Load structured data
    df = pd.read_csv(STRUCTURED_PATH)
    print(f"Structured data shape: {df.shape}")

    # Target variable: lastSoldPrice
    target_col = "lastSoldPrice"
    if target_col not in df.columns:
        for col in ["listPrice", "sale_price", "SalePrice"]:
            if col in df.columns:
                target_col = col
                break
    print(f"Target: {target_col}")

    y = df[target_col].values.astype(np.float64)

    # Remove target, listPrice (leakage), and non-feature columns
    drop_cols = [target_col, "listPrice"]
    for col in ["_id", "_split", "sanitized_text", "clean_text", "type",
                "sub_type", "zip", "address", "description"]:
        if col in df.columns:
            drop_cols.append(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_struct = df[feature_cols].values.astype(np.float64)

    # Handle NaN/Inf
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=np.nanmedian(y))

    print(f"Structured features: {X_struct.shape}")

    # Load text features
    tfidf = joblib.load(TFIDF_PATH)
    bert = joblib.load(BERT_PATH)
    print(f"TF-IDF features: {tfidf.shape}")
    print(f"BERT features: {bert.shape}")

    X_text = {
        "tfidf": tfidf.astype(np.float64),
        "bert_embeddings": bert.astype(np.float64),
    }

    return X_struct, y, X_text


def load_smoke_data(
    n_samples: int = 80,
    n_struct_features: int = 12,
    n_tfidf_features: int = 32,
    n_bert_features: int = 64,
    random_state: int = 42,
):
    """Generate a small deterministic dataset for end-to-end smoke runs.

    This does not replace the real Florida/Ames experiment. It verifies that
    the data/model/evaluation/result-writing pipeline is executable when the
    large Kaggle files are not present locally.
    """
    rng = np.random.default_rng(random_state)
    X_struct = rng.normal(size=(n_samples, n_struct_features)).astype(np.float64)
    tfidf = rng.normal(size=(n_samples, n_tfidf_features)).astype(np.float64)
    bert = rng.normal(size=(n_samples, n_bert_features)).astype(np.float64)
    y = (
        250000
        + X_struct[:, 0] * 60000
        + X_struct[:, 1] * -25000
        + tfidf[:, 0] * 15000
        + bert[:, 0] * 10000
        + rng.normal(scale=12000, size=n_samples)
    ).astype(np.float64)
    return X_struct, y, {"tfidf": tfidf, "bert_embeddings": bert}


# ============================================================
# Model Creation
# ============================================================

def create_models(smoke: bool = False) -> List[Dict[str, Any]]:
    """Create all models for evaluation

    Returns
    -------
    models : list of dict
        Each dict contains model_name, model_instance, modality.
    """
    config = {
        "random_forest": {
            "n_estimators": 20 if smoke else 200,
            "max_depth": 6 if smoke else 20,
        },
        "xgboost": {
            "n_estimators": 20 if smoke else 500,
            "max_depth": 3 if smoke else 8,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "ridge_regression": {"alpha": 1.0},
        "bert_mlp": {
            "hidden_dims": [32] if smoke else [256, 128],
            "dropout": 0.3,
            "learning_rate": 1e-3 if smoke else 1e-4,
            "batch_size": 16 if smoke else 32,
            "max_epochs": 3 if smoke else 50,
            "patience": 2 if smoke else 10,
        },
        "early_fusion_xgboost": {
            "n_estimators": 20 if smoke else 500,
            "max_depth": 3 if smoke else 8,
            "learning_rate": 0.05,
        },
        "early_fusion_mlp": {
            "hidden_dims": [64] if smoke else [512, 256, 128],
            "dropout": 0.3,
            "learning_rate": 1e-3 if smoke else 1e-4,
            "batch_size": 16 if smoke else 64,
            "max_epochs": 3 if smoke else 50,
            "patience": 2 if smoke else 10,
        },
        "mid_fusion_attention": {
            "structured_hidden_dim": 32 if smoke else 64,
            "text_hidden_dim": 64 if smoke else 768,
            "fusion_dim": 32 if smoke else 256,
            "num_attention_heads": 4,
            "dropout": 0.3,
            "learning_rate": 1e-3 if smoke else 1e-4,
            "batch_size": 16 if smoke else 64,
            "max_epochs": 3 if smoke else 50,
            "patience": 2 if smoke else 10,
        },
        "late_fusion_stacking": {
            "base_models": ["xgboost", "bert_mlp"],
            "meta_model": "ridge",
            "batch_size": 16 if smoke else 64,
            "max_epochs": 3 if smoke else 50,
        },
    }

    specs = [
        ("LinearBaseline", LinearBaseline, config, "structured"),
        ("RandomForestBaseline", RandomForestBaseline, config, "structured"),
        ("XGBoostBaseline", XGBoostBaseline, config, "structured"),
        ("TFIDFRidgeBaseline", TFIDFRidgeBaseline, config, "text"),
        ("BERTMLPBaseline", BERTMLPBaseline, config, "text"),
        (
            "EarlyFusionXGBoost",
            EarlyFusionXGBoost,
            {**config, "text_source": "tfidf"},
            "fusion_early",
        ),
        (
            "EarlyFusionMLP",
            EarlyFusionMLP,
            {**config, "text_source": "tfidf"},
            "fusion_early",
        ),
        (
            "MidFusionModel",
            MidFusionModel,
            {**config, "text_source": "bert_embeddings"},
            "fusion_mid",
        ),
        ("LateFusionStacking", LateFusionStacking, config, "fusion_late"),
    ]

    models = []
    for model_name, model_cls, model_config, modality in specs:
        try:
            models.append(
                {
                    "model_name": model_name,
                    "instance": model_cls(config=model_config),
                    "modality": modality,
                }
            )
        except ImportError as exc:
            print(f"[SKIP] {model_name}: {exc}")

    return models


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(
    model_info: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_text_train: Dict[str, np.ndarray],
    X_text_val: Dict[str, np.ndarray],
    X_text_test: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Train a single model and evaluate

    Parameters
    ----------
    model_info : dict
        Contains model_name, instance, modality.
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Train/val/test data.
    X_text_train, X_text_val, X_text_test : dict
        Text feature data.

    Returns
    -------
    result : dict
        Result dictionary with metrics and metadata.
    """
    model = model_info["instance"]
    model_name = model_info["model_name"]
    modality = model_info["modality"]

    print(f"\n{'='*50}")
    print(f"Training: {model_name} ({modality})")
    print(f"{'='*50}")

    start_time = time.time()

    try:
        # Fit based on modality
        if modality == "structured":
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
            )
        elif modality == "text":
            model.fit(
                None, y_train,
                X_text=X_text_train,
                X_val=None, y_val=y_val,
                X_text_val=X_text_val,
            )
        else:
            model.fit(
                X_train, y_train,
                X_text=X_text_train,
                X_val=X_val, y_val=y_val,
                X_text_val=X_text_val,
            )

        train_time = time.time() - start_time

        # Predict
        if modality == "structured":
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
        elif modality == "text":
            y_train_pred = model.predict(None, X_text=X_text_train)
            y_val_pred = model.predict(None, X_text=X_text_val)
            y_test_pred = model.predict(None, X_text=X_text_test)
        else:
            y_train_pred = model.predict(X_train, X_text=X_text_train)
            y_val_pred = model.predict(X_val, X_text=X_text_val)
            y_test_pred = model.predict(X_test, X_text=X_text_test)

        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        # Save model
        model_path = MODELS_DIR / f"{model_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        # Print results
        print(f"  Train: RMSE={train_metrics['rmse']:.2f}, "
              f"R2={train_metrics['r2']:.4f}")
        print(f"  Valid: RMSE={val_metrics['rmse']:.2f}, "
              f"R2={val_metrics['r2']:.4f}")
        print(f"  Test:  RMSE={test_metrics['rmse']:.2f}, "
              f"R2={test_metrics['r2']:.4f}")
        print(f"  Time: {train_time:.1f}s")

        status = "success"

    except Exception as e:
        print(f"  [FAILED] Training error: {str(e)}")
        train_time = time.time() - start_time
        nan_metrics = {"rmse": float("nan"), "mae": float("nan"),
                       "r2": float("nan"), "mape": float("nan")}
        train_metrics = val_metrics = test_metrics = nan_metrics
        status = "failed"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "model_name": model_name,
        "modality": modality,
        "status": status,
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "train_mape": train_metrics["mape"],
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "val_mape": val_metrics["mape"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "train_time_sec": round(train_time, 2),
    }


# ============================================================
# Results Saving
# ============================================================

def save_results(results: List[Dict[str, Any]]):
    """Save experiment results to CSV

    Parameters
    ----------
    results : list of dict
        Experiment results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp", "model_name", "modality", "status",
        "train_rmse", "train_mae", "train_r2", "train_mape",
        "val_rmse", "val_mae", "val_r2", "val_mape",
        "test_rmse", "test_mae", "test_r2", "test_mape",
        "train_time_sec",
    ]

    with open(EXPERIMENT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExperiment log saved: {EXPERIMENT_LOG_PATH}")


def print_summary(results: List[Dict[str, Any]]):
    """Print experiment summary table"""
    print("\n" + "=" * 90)
    print("Experiment Summary (Test Set)")
    print("=" * 90)
    header = (f"{'Model':<25} {'Modality':<15} {'RMSE':>10} "
              f"{'MAE':>10} {'R2':>10} {'MAPE':>10} {'Status':>8}")
    print(header)
    print("-" * 90)

    for r in results:
        status = r["status"]
        if status == "success":
            print(
                f"{r['model_name']:<25} {r['modality']:<15} "
                f"{r['test_rmse']:>10.2f} {r['test_mae']:>10.2f} "
                f"{r['test_r2']:>10.4f} {r['test_mape']:>10.2f}% "
                f"{status:>8}"
            )
        else:
            print(
                f"{r['model_name']:<25} {r['modality']:<15} "
                f"{'FAILED':>10} {'FAILED':>10} {'FAILED':>10} "
                f"{'FAILED':>10} {status:>8}"
            )

    print("-" * 90)

    successful = [r for r in results if r["status"] == "success"]
    if successful:
        best_val = min(successful, key=lambda x: x["val_rmse"])
        best_test = min(successful, key=lambda x: x["test_rmse"])
        print(f"\nBest Val Model: {best_val['model_name']} "
              f"(RMSE={best_val['val_rmse']:.2f})")
        print(f"Best Test Model: {best_test['model_name']} "
              f"(RMSE={best_test['test_rmse']:.2f})")


# ============================================================
# Main Entry
# ============================================================

def main(argv: Optional[List[str]] = None):
    """Run all experiments"""
    parser = argparse.ArgumentParser(description="Run house price experiments.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small synthetic end-to-end smoke experiment.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Multi-source Data Fusion -- House Price Prediction")
    print("Experiment Runner")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load data
    if args.smoke:
        print("Smoke mode enabled: using synthetic data.")
        X_struct, y, X_text = load_smoke_data()
    else:
        X_struct, y, X_text = load_data()

    # 2. Split data consistently
    print("\nSplitting data...")
    n_total = len(y)
    indices = np.arange(n_total)

    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42
    )

    X_train, X_val, X_test = (
        X_struct[idx_train], X_struct[idx_val], X_struct[idx_test]
    )
    y_train, y_val, y_test = (
        y[idx_train], y[idx_val], y[idx_test]
    )

    X_text_train = {
        "tfidf": X_text["tfidf"][idx_train].astype(np.float64),
        "bert_embeddings": X_text["bert_embeddings"][idx_train].astype(np.float64),
    }
    X_text_val = {
        "tfidf": X_text["tfidf"][idx_val].astype(np.float64),
        "bert_embeddings": X_text["bert_embeddings"][idx_val].astype(np.float64),
    }
    X_text_test = {
        "tfidf": X_text["tfidf"][idx_test].astype(np.float64),
        "bert_embeddings": X_text["bert_embeddings"][idx_test].astype(np.float64),
    }

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Valid: {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    # 3. Create models
    models = create_models(smoke=args.smoke)
    print(f"\nTotal {len(models)} models to train")

    # 4. Train and evaluate
    results = []
    for model_info in models:
        result = run_experiment(
            model_info,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            X_text_train, X_text_val, X_text_test,
        )
        results.append(result)

    # 5. Save results
    save_results(results)

    # 6. Print summary
    print_summary(results)

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
