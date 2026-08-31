"""Scenario forecast generator.

Wraps a trained LightGBM model and applies pessimistic, baseline, and
optimistic multipliers to produce three parallel forecast tracks.

The scenarios are deliberately simple and transparent: finance teams
understand and trust a clearly-labelled percentage adjustment far more
than opaque distributional sampling. The 80/100/120 multipliers are
calibrated against the historical demand range seen in the EDA.

Design decisions
----------------
* LightGBM is used here (not Prophet) because it runs locally without
  CmdStan. When Prophet is available (SageMaker), it should be preferred
  for the baseline because its uncertainty intervals (yhat_lower/yhat_upper)
  naturally express scenario width. We apply the same multiplier logic.
* Recursive one-step-ahead forecasting: each predicted week is fed back
  as a lag feature for the next week. This is standard for multi-step
  ahead inference with lag-based models.
* We clip negative predictions to zero. Negative bookings are impossible.
* Output is a CSV with columns: week_start, pessimistic, baseline, optimistic.
  This format is what the Streamlit dashboard reads directly.

Run locally:
    python src/scenarios/scenario_generator.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import create_lag_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCENARIOS = {
    "pessimistic": 0.80,
    "baseline": 1.00,
    "optimistic": 1.20,
}

FORECAST_WEEKS = 26  # 6-month forward look


def _make_future_features(history: pd.DataFrame, forecast_weeks: int) -> pd.DataFrame:
    """Extend the historical feature frame by rolling one-step-ahead predictions.

    Parameters
    ----------
    history : Feature-engineered DataFrame including all lag and calendar cols.
              Must already have the 'y' column with actuals.
    forecast_weeks : How many weeks to forecast forward.

    Returns
    -------
    DataFrame of length forecast_weeks with feature columns populated.
    Each row's lag features are built from the predicted (or actual) values
    that came before it.
    """
    feature_cols = get_feature_columns()
    # Start from last known date
    last_date = history["ds"].max()

    future_rows = []
    # Rolling buffer of y values (actuals then predictions) used to build lags
    y_buffer = list(history["y"].values)

    for week_offset in range(1, forecast_weeks + 1):
        future_date = last_date + pd.Timedelta(weeks=week_offset)
        dt = pd.Timestamp(future_date)

        # Build a minimal row dict with calendar features
        row = {
            "ds": future_date,
            "week_of_year": int(dt.isocalendar().week),
            "month": dt.month,
            "quarter": dt.quarter,
            "year": dt.year,
            "is_peak_season": int(dt.month in [6, 7, 8, 9]),
            "week_sin": np.sin(2 * np.pi * int(dt.isocalendar().week) / 52),
            "week_cos": np.cos(2 * np.pi * int(dt.isocalendar().week) / 52),
        }

        # Lag features from y_buffer (most recent first via negative indexing)
        lag_map = {1: -1, 2: -2, 4: -4, 8: -8, 12: -12}
        for lag, idx in lag_map.items():
            row[f"lag_{lag}w"] = y_buffer[idx] if len(y_buffer) >= abs(idx) else np.nan

        # Rolling stats (over y_buffer, shifted 1 to avoid using the future)
        for window in (4, 8, 12):
            vals = y_buffer[-window - 1:-1] if len(y_buffer) > window else y_buffer[:-1]
            row[f"rolling_mean_{window}w"] = float(np.mean(vals)) if vals else np.nan

        vals_4 = y_buffer[-5:-1] if len(y_buffer) > 4 else y_buffer[:-1]
        row["rolling_std_4w"] = float(np.std(vals_4, ddof=1)) if len(vals_4) > 1 else 0.0
        row["trend_signal"] = row["rolling_mean_4w"] - row["rolling_mean_12w"]

        future_rows.append(row)
        # Placeholder y (will be overwritten in forecast loop); use last known
        y_buffer.append(y_buffer[-1])

    return pd.DataFrame(future_rows)


def generate_scenarios(
    model_path: str,
    data_path: str,
    output_path: str,
    forecast_weeks: int = FORECAST_WEEKS,
) -> pd.DataFrame:
    """Generate pessimistic, baseline, and optimistic scenario forecasts.

    Parameters
    ----------
    model_path : Path to pickled LightGBM model
    data_path : Path to weekly_demand.csv
    output_path : Where to write scenarios CSV
    forecast_weeks : Number of weeks to forecast forward

    Returns
    -------
    DataFrame with columns: week_start, pessimistic, baseline, optimistic
    """
    log.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    log.info(f"Loading historical data from {data_path}")
    raw = pd.read_csv(data_path, parse_dates=["week_start"])
    raw = raw.rename(columns={"week_start": "ds", "bookings": "y"})
    raw = raw.sort_values("ds").reset_index(drop=True)
    log.info(f"History: {len(raw)} weeks ({raw['ds'].min().date()} to {raw['ds'].max().date()})")

    # Build full feature frame from actuals (for lag buffer initialisation)
    feature_cols = get_feature_columns()
    history = create_lag_features(raw)

    # --- Recursive one-step-ahead forecast ---
    log.info(f"Forecasting {forecast_weeks} weeks ahead (recursive)...")
    forecast_results = []

    # Use full history as the rolling y_buffer for lag construction
    y_buffer = list(raw["y"].values)

    last_date = raw["ds"].max()

    for week_offset in range(1, forecast_weeks + 1):
        future_date = last_date + pd.Timedelta(weeks=week_offset)
        dt = pd.Timestamp(future_date)

        row = {
            "week_of_year": int(dt.isocalendar().week),
            "month": dt.month,
            "quarter": dt.quarter,
            "year": dt.year,
            "is_peak_season": int(dt.month in [6, 7, 8, 9]),
            "week_sin": np.sin(2 * np.pi * int(dt.isocalendar().week) / 52),
            "week_cos": np.cos(2 * np.pi * int(dt.isocalendar().week) / 52),
        }

        for lag in (1, 2, 4, 8, 12):
            row[f"lag_{lag}w"] = y_buffer[-lag] if len(y_buffer) >= lag else np.nan

        for window in (4, 8, 12):
            vals = y_buffer[-window - 1:-1] if len(y_buffer) > window else y_buffer[:-1]
            row[f"rolling_mean_{window}w"] = float(np.mean(vals)) if vals else np.nan

        vals_4 = y_buffer[-5:-1] if len(y_buffer) > 4 else y_buffer[:-1]
        row["rolling_std_4w"] = float(np.std(vals_4, ddof=1)) if len(vals_4) > 1 else 0.0
        row["trend_signal"] = row.get("rolling_mean_4w", 0) - row.get("rolling_mean_12w", 0)

        X = pd.DataFrame([row])[feature_cols]
        baseline_pred = float(max(model.predict(X)[0], 0))

        forecast_results.append({
            "week_start": future_date,
            "pessimistic": round(baseline_pred * SCENARIOS["pessimistic"], 1),
            "baseline": round(baseline_pred * SCENARIOS["baseline"], 1),
            "optimistic": round(baseline_pred * SCENARIOS["optimistic"], 1),
        })

        # Feed baseline prediction back into buffer for next week's lags
        y_buffer.append(baseline_pred)

    scenarios_df = pd.DataFrame(forecast_results)
    log.info(f"Forecast range: {scenarios_df['week_start'].min().date()} "
             f"to {scenarios_df['week_start'].max().date()}")

    # --- Save ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scenarios_df.to_csv(output_path, index=False)
    log.info(f"Scenarios saved to {output_path}")

    # Print summary table
    log.info("\nScenario forecast preview (first 8 weeks):")
    log.info(scenarios_df.head(8).to_string(index=False))

    return scenarios_df


if __name__ == "__main__":
    result = generate_scenarios(
        model_path="models/lgbm_model.pkl",
        data_path="data/processed/weekly_demand.csv",
        output_path="data/processed/scenarios.csv",
    )
    print("\n=== Scenario Forecasts Generated ===")
    print(result.to_string(index=False))
