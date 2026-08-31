"""Prophet baseline trainer.

Reads weekly demand, fits a Prophet model with yearly seasonality,
holds out the last TEST_SIZE_WEEKS weeks for evaluation, and writes
both the pickled model and a metrics JSON.

Run locally:
    python src/models/train_prophet.py

The same function is called by the SageMaker training script in sagemaker/.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.metrics import mape, mae, rmse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_SIZE_WEEKS = 12


def train_prophet(
    data_path: str,
    model_output_path: str,
    metrics_output_path: str,
    test_size: int = TEST_SIZE_WEEKS,
) -> dict:
    """Train Prophet on weekly demand and evaluate on the held-out test set.

    Parameters
    ----------
    data_path : Path to weekly_demand.csv (columns: week_start, bookings)
    model_output_path : Where to pickle the fitted Prophet model
    metrics_output_path : Where to write the metrics JSON
    test_size : Number of weeks to hold out for evaluation

    Returns
    -------
    dict of evaluation metrics
    """
    # --- Load data ---
    log.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["week_start"])
    df = df.rename(columns={"week_start": "ds", "bookings": "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    log.info(f"Loaded {len(df)} weeks of data ({df['ds'].min()} to {df['ds'].max()})")

    # --- Time-based train/test split (never random split for time series) ---
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    log.info(f"Train: {len(train)} weeks | Test: {len(test)} weeks")

    # --- Fit Prophet ---
    log.info("Fitting Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # data is already weekly aggregated
        daily_seasonality=False,
        seasonality_mode="additive", # EDA showed additive seasonality
        changepoint_prior_scale=0.05, # regularise trend changepoints
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    model.fit(train)

    # --- Forecast on test period ---
    future = model.make_future_dataframe(periods=test_size, freq="W")
    forecast = model.predict(future)
    test_forecast = forecast.tail(test_size)

    y_true = test["y"].values
    y_pred = test_forecast["yhat"].values
    y_pred = np.maximum(y_pred, 0)  # clip negative forecasts to zero

    # --- Compute metrics ---
    metrics = {
        "model": "prophet",
        "test_size_weeks": test_size,
        "mape": round(mape(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "train_weeks": len(train),
        "test_weeks": len(test),
        "train_start": str(train["ds"].min().date()),
        "train_end": str(train["ds"].max().date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
    }
    log.info(f"Prophet metrics: MAPE={metrics['mape']:.2f}%  MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}")

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
    metrics = train_prophet(
        data_path="data/processed/weekly_demand.csv",
        model_output_path="models/prophet_model.pkl",
        metrics_output_path="models/prophet_metrics.json",
    )
    print("\n=== Prophet Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
