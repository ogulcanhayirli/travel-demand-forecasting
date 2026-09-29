"""The scenario generator must build the same features the model trained on.

Inference builds each week's features by hand from a buffer of past values,
while training uses create_lag_features on a DataFrame. A silent mismatch
between the two (an earlier version dropped the most recent week from every
rolling window) degrades every forecast without failing anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build_features import create_lag_features, get_feature_columns
from src.scenarios.scenario_generator import next_week_features


def test_inference_features_match_training_features() -> None:
    ds = pd.date_range("2015-07-06", periods=80, freq="W-MON")
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"ds": ds, "y": 600 + rng.normal(0, 50, len(ds)).round()})

    training = create_lag_features(raw).set_index("ds")
    y = raw["y"].tolist()

    for i, week_start in enumerate(ds):
        if week_start not in training.index:
            continue  # warmup rows dropped by create_lag_features
        built = next_week_features(y[:i], week_start)
        for col in get_feature_columns():
            assert built[col] == pytest.approx(training.loc[week_start, col]), (
                f"{col} differs for {week_start.date()}"
            )
