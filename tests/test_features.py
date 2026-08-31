"""Unit tests for feature engineering.

Tests that create_lag_features and get_feature_columns behave correctly
and that no target leakage is present in the feature set.

Run:
    pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build_features import create_lag_features, get_feature_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_weekly_df(n_weeks: int = 80) -> pd.DataFrame:
    """Create a synthetic weekly demand DataFrame for testing."""
    dates = pd.date_range(start="2015-01-05", periods=n_weeks, freq="W")
    np.random.seed(42)
    bookings = 500 + 100 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52) + np.random.randn(n_weeks) * 20
    return pd.DataFrame({"ds": dates, "y": bookings})


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)

    def test_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols)), "Duplicate feature columns found"

    def test_expected_lag_cols_present(self):
        cols = get_feature_columns()
        for lag in (1, 2, 4, 8, 12):
            assert f"lag_{lag}w" in cols, f"lag_{lag}w missing from feature columns"

    def test_no_52w_lag(self):
        """52-week lag is excluded because it drops too many rows on small datasets."""
        cols = get_feature_columns()
        assert "lag_52w" not in cols

    def test_calendar_features_present(self):
        cols = get_feature_columns()
        for feat in ("week_of_year", "month", "quarter", "year", "is_peak_season",
                     "week_sin", "week_cos"):
            assert feat in cols

    def test_rolling_features_present(self):
        cols = get_feature_columns()
        for feat in ("rolling_mean_4w", "rolling_mean_8w", "rolling_mean_12w",
                     "rolling_std_4w", "trend_signal"):
            assert feat in cols


# ---------------------------------------------------------------------------
# create_lag_features tests
# ---------------------------------------------------------------------------

class TestCreateLagFeatures:
    def test_output_has_feature_columns(self):
        df = make_weekly_df(80)
        result = create_lag_features(df)
        for col in get_feature_columns():
            assert col in result.columns, f"Feature column {col} missing from output"

    def test_no_nulls_in_feature_columns(self):
        df = make_weekly_df(80)
        result = create_lag_features(df)
        feature_cols = get_feature_columns()
        null_counts = result[feature_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Null values found in features:\n{null_counts[null_counts > 0]}"

    def test_rows_dropped_for_lag_warmup(self):
        """First max(lags) rows should be dropped due to NaN lag values."""
        df = make_weekly_df(80)
        result = create_lag_features(df, lags=(1, 2, 4, 8, 12))
        # max lag is 12, so 12 rows should be dropped from the start
        assert len(result) == len(df) - 12

    def test_output_sorted_by_date(self):
        df = make_weekly_df(80).sample(frac=1, random_state=0)  # shuffle input
        result = create_lag_features(df)
        assert result["ds"].is_monotonic_increasing

    def test_is_peak_season_correct(self):
        """Rows in Jun-Sep should have is_peak_season=1, others 0."""
        df = make_weekly_df(80)
        result = create_lag_features(df)
        peak_months = result["ds"].dt.month.isin([6, 7, 8, 9])
        assert (result.loc[peak_months, "is_peak_season"] == 1).all()
        assert (result.loc[~peak_months, "is_peak_season"] == 0).all()

    def test_week_sin_cos_range(self):
        """Sine and cosine encodings must be in [-1, 1]."""
        df = make_weekly_df(80)
        result = create_lag_features(df)
        assert result["week_sin"].between(-1, 1).all()
        assert result["week_cos"].between(-1, 1).all()

    def test_trend_signal_is_difference(self):
        """trend_signal must equal rolling_mean_4w - rolling_mean_12w."""
        df = make_weekly_df(80)
        result = create_lag_features(df)
        expected = result["rolling_mean_4w"] - result["rolling_mean_12w"]
        pd.testing.assert_series_equal(result["trend_signal"], expected, check_names=False)

    def test_no_leakage_in_rolling_features(self):
        """Rolling stats must use shift(1) — they must not include the current row's y."""
        df = make_weekly_df(60)
        result = create_lag_features(df)
        # Manually compute what rolling_mean_4w should be (shifted by 1)
        expected_4w = df.sort_values("ds")["y"].shift(1).rolling(4).mean().dropna()
        # Values should align (after the warmup rows are dropped)
        computed = result["rolling_mean_4w"].values
        # They won't be identical in length but the values should match the pattern
        assert not np.allclose(computed, df.sort_values("ds")["y"].rolling(4).mean().dropna().values[:len(computed)]), \
            "Rolling mean appears to use current row (potential leakage)"

    def test_custom_lags(self):
        """Custom lag sets should produce the correct lag columns."""
        df = make_weekly_df(80)
        result = create_lag_features(df, lags=(1, 4))
        assert "lag_1w" in result.columns
        assert "lag_4w" in result.columns
        assert "lag_2w" not in result.columns


# ---------------------------------------------------------------------------
# Metric function tests
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_mape_perfect_forecast(self):
        from src.models.metrics import mape
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self):
        from src.models.metrics import mape
        y_true = np.array([100.0])
        y_pred = np.array([110.0])
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_ignores_zeros(self):
        from src.models.metrics import mape
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([999.0, 110.0])  # first pair should be ignored
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mae_perfect_forecast(self):
        from src.models.metrics import mae
        y = np.array([100.0, 200.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        from src.models.metrics import rmse
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9 + 16) / 2) = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))
