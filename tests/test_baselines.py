"""Tests for the naive baselines reported next to the model metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train_lightgbm import naive_baselines


def make_history(n_weeks: int = 60) -> pd.DataFrame:
    ds = pd.date_range("2016-01-04", periods=n_weeks, freq="W-MON")
    return pd.DataFrame({"ds": ds, "y": np.arange(100.0, 100.0 + n_weeks)})


def test_lag1_and_lag52_use_the_right_weeks() -> None:
    history = make_history()
    test = history.tail(4)

    result = naive_baselines(history, test)

    # y rises by exactly 1 per week, so lag 1 is off by 1 and lag 52 by 52.
    assert result["naive_lag1"]["mae"] == pytest.approx(1.0)
    assert result["naive_lag52"]["mae"] == pytest.approx(52.0)


def test_missing_history_raises() -> None:
    history = make_history(40)  # too short for a 52-week lag
    with pytest.raises(ValueError, match="naive_lag52"):
        naive_baselines(history, history.tail(4))
