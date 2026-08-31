"""Feature engineering for the travel demand forecasting pipeline.

Builds lag windows, rolling statistics, and calendar features that the
LightGBM model consumes. Prophet uses the raw ds/y series directly.

Design decisions documented in notebooks/01_eda.ipynb:
- Lag features at 1, 2, 4, 8, 12, 52 weeks capture short and long-range autocorrelation
- Rolling stats capture trend direction and volatility
- Calendar features encode seasonality without requiring the model to learn it from lags alone
- No leaking features: all features use only information available at prediction time T
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def create_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 4, 8, 12),
    target_col: str = "y",
    date_col: str = "ds",
) -> pd.DataFrame:
    """Build lag, rolling, and calendar features for the LightGBM model.

    Parameters
    ----------
    df : DataFrame with at minimum columns `date_col` and `target_col`.
         Must be sorted by date ascending with no gaps.
    lags : Week offsets to create lag features for.
    target_col : Name of the target column (default 'y').
    date_col : Name of the date column (default 'ds').

    Returns
    -------
    DataFrame with added feature columns. Rows with NaN lag values
    (the first max(lags) rows) are dropped.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    # --- Lag features ---
    for lag in lags:
        df[f"lag_{lag}w"] = df[target_col].shift(lag)

    # --- Rolling statistics (computed on past data only to avoid leakage) ---
    for window in (4, 8, 12):
        df[f"rolling_mean_{window}w"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
    df["rolling_std_4w"] = df[target_col].shift(1).rolling(window=4).std()

    # --- Trend feature: difference between 4w mean and 12w mean ---
    df["trend_signal"] = df["rolling_mean_4w"] - df["rolling_mean_12w"]

    # --- Calendar features ---
    dt = pd.to_datetime(df[date_col])
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year

    # Peak season flag (Jun-Sep = summer peak based on EDA)
    df["is_peak_season"] = dt.dt.month.isin([6, 7, 8, 9]).astype(int)

    # Sine/cosine encoding of week_of_year for cyclical continuity
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # --- Drop rows with NaN lags (first max(lags) rows) ---
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used by the LightGBM model.

    Keep this in sync with create_lag_features so training and inference
    always use the same feature set.
    """
    lag_cols = [f"lag_{lag}w" for lag in (1, 2, 4, 8, 12)]
    rolling_cols = [
        "rolling_mean_4w",
        "rolling_mean_8w",
        "rolling_mean_12w",
        "rolling_std_4w",
        "trend_signal",
    ]
    calendar_cols = [
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_peak_season",
        "week_sin",
        "week_cos",
    ]
    return lag_cols + rolling_cols + calendar_cols


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/weekly_demand.csv"
    raw = pd.read_csv(data_path, parse_dates=["week_start"])
    raw = raw.rename(columns={"week_start": "ds", "bookings": "y"})
    features = create_lag_features(raw)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(features.head())
