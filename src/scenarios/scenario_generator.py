"""Scenario forecast generator.

Wraps a trained LightGBM model and applies pessimistic, baseline, and
optimistic multipliers to produce three parallel forecast tracks.

The scenarios are deliberately simple and transparent: a fixed 0.80x /
1.00x / 1.20x adjustment to the baseline forecast. They are planning
assumptions, not statistically derived intervals.

Design decisions
----------------
* LightGBM is used because it is the current champion (see
  models/promotion_decision.json). Prophet was evaluated as the challenger
  and did not clear the promotion threshold.
* Recursive one-step-ahead forecasting: each predicted week is fed back
  as a lag feature for the next week. This is standard for multi-step
  ahead inference with lag-based models. The forecast starts after the
  last week in weekly_demand.csv, which only contains complete weeks.
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
from src.features.build_features import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCENARIOS = {
    "pessimistic": 0.80,
    "baseline": 1.00,
    "optimistic": 1.20,
}

FORECAST_WEEKS = 26  # 6-month forward look
LAGS = (1, 2, 4, 8, 12)  # must match get_feature_columns


def next_week_features(y_history: list[float], week_start: pd.Timestamp) -> dict:
    """Build one feature row for `week_start` from the weeks before it.

    `y_history` ends with the week immediately before `week_start`. The
    values match what create_lag_features computes for the same week in
    training (tests/test_scenarios.py checks this), so the model sees the
    same inputs at inference as it was trained on.
    """
    week = int(week_start.isocalendar().week)
    row = {
        "week_of_year": week,
        "month": week_start.month,
        "quarter": week_start.quarter,
        "year": week_start.year,
        "is_peak_season": int(week_start.month in [6, 7, 8, 9]),
        "week_sin": np.sin(2 * np.pi * week / 52),
        "week_cos": np.cos(2 * np.pi * week / 52),
    }
    for lag in LAGS:
        row[f"lag_{lag}w"] = y_history[-lag]
    for window in (4, 8, 12):
        row[f"rolling_mean_{window}w"] = float(np.mean(y_history[-window:]))
    row["rolling_std_4w"] = float(np.std(y_history[-4:], ddof=1))
    row["trend_signal"] = row["rolling_mean_4w"] - row["rolling_mean_12w"]
    return row


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

    feature_cols = get_feature_columns()

    # --- Recursive one-step-ahead forecast ---
    log.info(f"Forecasting {forecast_weeks} weeks ahead (recursive)...")
    forecast_results = []

    # Actuals first, then each prediction is appended to feed the next week's lags
    y_buffer = list(raw["y"].astype(float).values)
    last_date = raw["ds"].max()

    for week_offset in range(1, forecast_weeks + 1):
        future_date = last_date + pd.Timedelta(weeks=week_offset)
        X = pd.DataFrame([next_week_features(y_buffer, future_date)])[feature_cols]
        baseline_pred = float(max(model.predict(X)[0], 0))

        forecast_results.append({
            "week_start": future_date,
            "pessimistic": round(baseline_pred * SCENARIOS["pessimistic"], 1),
            "baseline": round(baseline_pred * SCENARIOS["baseline"], 1),
            "optimistic": round(baseline_pred * SCENARIOS["optimistic"], 1),
        })
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
