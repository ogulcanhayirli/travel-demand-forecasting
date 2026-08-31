"""LightGBM trainer with lag and calendar features.

Reads the same weekly_demand.csv as Prophet, builds lag features via
build_features, and trains a gradient boosted regressor. Holds out the
last TEST_SIZE_WEEKS weeks for evaluation.

Run locally:
    python src/models/train_lightgbm.py

The same function is called by the SageMaker training script in sagemaker/.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import create_lag_features, get_feature_columns
from src.models.metrics import mape, mae, rmse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_SIZE_WEEKS = 12


def train_lightgbm(
    data_path: str,
    model_output_path: str,
    metrics_output_path: str,
    test_size: int = TEST_SIZE_WEEKS,
) -> dict:
    """Train LightGBM on engineered lag and calendar features.

    Parameters
    ----------
    data_path : Path to weekly_demand.csv
    model_output_path : Where to pickle the fitted LightGBM booster
    metrics_output_path : Where to write the metrics JSON
    test_size : Number of weeks to hold out for evaluation

    Returns
    -------
    dict of evaluation metrics
    """
    # --- Load and prepare data ---
    log.info(f"Loading data from {data_path}")
    raw = pd.read_csv(data_path, parse_dates=["week_start"])
    raw = raw.rename(columns={"week_start": "ds", "bookings": "y"})
    raw = raw.sort_values("ds").reset_index(drop=True)
    log.info(f"Loaded {len(raw)} weeks of data")

    # --- Build features ---
    log.info("Engineering features...")
    df = create_lag_features(raw)
    feature_cols = get_feature_columns()
    log.info(f"Feature matrix shape after lag drop: {df.shape}")
    log.info(f"Features: {feature_cols}")

    # --- Time-based train/test split ---
    # Split AFTER feature engineering to avoid leakage from rolling windows
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    log.info(f"Train: {len(train)} rows | Test: {len(test)} rows")

    X_train = train[feature_cols]
    y_train = train["y"]
    X_test = test[feature_cols]
    y_test = test["y"]

    # --- Train LightGBM ---
    log.info("Training LightGBM model...")
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 5,
        "n_estimators": 500,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info(f"Best iteration: {model.best_iteration_}")

    # --- Evaluate ---
    y_pred = np.maximum(model.predict(X_test), 0)
    y_true = y_test.values

    metrics = {
        "model": "lightgbm",
        "test_size_weeks": test_size,
        "mape": round(mape(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "best_iteration": int(model.best_iteration_),
        "n_features": len(feature_cols),
        "train_weeks": len(train),
        "test_weeks": len(test),
        "train_start": str(train["ds"].min().date()),
        "train_end": str(train["ds"].max().date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
    }
    log.info(f"LightGBM metrics: MAPE={metrics['mape']:.2f}%  MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}")

    # --- Feature importance (top 10) ---
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    log.info(f"Top 10 features:\n{importance.head(10)}")
    metrics["top_features"] = importance.head(10).to_dict()

    # --- Save model ---
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved to {model_output_path}")

    # --- Save metrics ---
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_output_path}")

    return metrics


if __name__ == "__main__":
    metrics = train_lightgbm(
        data_path="data/processed/weekly_demand.csv",
        model_output_path="models/lgbm_model.pkl",
        metrics_output_path="models/lgbm_metrics.json",
    )
    print("\n=== LightGBM Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
