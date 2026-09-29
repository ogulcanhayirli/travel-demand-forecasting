"""LightGBM trainer with lag and calendar features.

Reads the same weekly_demand.csv as Prophet, builds lag features via
build_features, and trains a gradient boosted regressor. Holds out the
last TEST_SIZE_WEEKS weeks for evaluation. The number of trees is chosen by
early stopping on the last VAL_SIZE_WEEKS weeks of the training period, then
the model is refit on the whole training period. Naive baselines are scored
on the same test weeks.

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
VAL_SIZE_WEEKS = 12  # tail of the training period used only for early stopping
MAX_TREES = 500


def naive_baselines(history: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Score naive forecasts on the same test weeks as the model.

    Both are one-step-ahead, like the model evaluation, which builds its
    lag features from actuals:
      naive_lag1   : this week = last week's actual
      naive_lag52  : this week = the same week last year
    """
    actuals = history.set_index("ds")["y"]
    y_true = test["y"].values
    preds = {
        "naive_lag1": actuals.reindex(test["ds"] - pd.Timedelta(weeks=1)).values,
        "naive_lag52": actuals.reindex(test["ds"] - pd.Timedelta(weeks=52)).values,
    }
    out = {}
    for name, y_pred in preds.items():
        if np.isnan(y_pred).any():
            raise ValueError(f"{name}: history does not cover every test week")
        out[name] = {
            "mape": round(mape(y_true, y_pred), 4),
            "mae": round(mae(y_true, y_pred), 4),
            "rmse": round(rmse(y_true, y_pred), 4),
        }
    return out


def train_lightgbm(
    data_path: str,
    model_output_path: str,
    metrics_output_path: str,
    test_size: int = TEST_SIZE_WEEKS,
    val_size: int = VAL_SIZE_WEEKS,
) -> dict:
    """Train LightGBM on engineered lag and calendar features.

    Parameters
    ----------
    data_path : Path to weekly_demand.csv
    model_output_path : Where to pickle the fitted LightGBM booster
    metrics_output_path : Where to write the metrics JSON
    test_size : Number of weeks to hold out for evaluation
    val_size : Weeks at the end of the training period used for early stopping

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

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 5,
        "random_state": 42,
        "verbose": -1,
    }

    # --- Choose the number of trees without looking at the test set ---
    # Early stopping monitors the last `val_size` weeks of the training
    # period. The test weeks are never used for model selection.
    fit = train.iloc[:-val_size]
    val = train.iloc[-val_size:]
    log.info(f"Early stopping: fit on {len(fit)} rows, validate on {len(val)} rows "
             f"({val['ds'].min().date()} to {val['ds'].max().date()})")
    booster = lgb.train(
        params,
        lgb.Dataset(fit[feature_cols], fit["y"]),
        num_boost_round=MAX_TREES,
        valid_sets=[lgb.Dataset(val[feature_cols], val["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iteration = int(booster.best_iteration)
    log.info(f"Best iteration on validation window: {best_iteration}")

    # --- Refit on the full training period with the chosen tree count ---
    log.info("Training LightGBM model...")
    model = lgb.LGBMRegressor(n_estimators=best_iteration, **params)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = np.maximum(model.predict(X_test), 0)
    y_true = y_test.values

    metrics = {
        "model": "lightgbm",
        "test_size_weeks": test_size,
        "mape": round(mape(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "best_iteration": best_iteration,
        "early_stopping_val_weeks": val_size,
        "n_features": len(feature_cols),
        "train_weeks": len(train),
        "test_weeks": len(test),
        "train_start": str(train["ds"].min().date()),
        "train_end": str(train["ds"].max().date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
        "baselines": naive_baselines(raw, test),
    }
    log.info(f"LightGBM metrics: MAPE={metrics['mape']:.2f}%  MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}")
    for name, b in metrics["baselines"].items():
        log.info(f"Baseline {name}: MAPE={b['mape']:.2f}%  MAE={b['mae']:.1f}  RMSE={b['rmse']:.1f}")

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
