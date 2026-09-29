"""Tests for the PSI drift detector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring.drift_detector import calculate_psi, check_drift_and_alert


def test_same_distribution_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(700, 50, 5000)
    current = rng.normal(700, 50, 5000)

    assert calculate_psi(reference, current) < 0.02


def test_identical_samples_score_zero() -> None:
    values = np.arange(100.0)
    assert calculate_psi(values, values) == pytest.approx(0.0)


def test_shifted_distribution_crosses_the_threshold() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(700, 50, 5000)
    current = rng.normal(800, 50, 5000)  # two standard deviations higher

    assert calculate_psi(reference, current) > 0.2


def test_values_outside_the_reference_range_are_counted() -> None:
    """A shift entirely above the reference range is the clearest drift of all."""
    reference = np.arange(100.0)
    current = np.full(50, 1000.0)

    assert calculate_psi(reference, current) > 1.0


def test_constant_reference_does_not_crash() -> None:
    assert calculate_psi(np.full(20, 5.0), np.full(20, 5.0)) == pytest.approx(0.0)
    assert calculate_psi(np.full(20, 5.0), np.full(20, 9.0)) > 0.2


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        calculate_psi([], [1.0, 2.0])


def _write(path: Path, bookings) -> str:
    pd.DataFrame({"week_start": range(len(bookings)), "bookings": bookings}).to_csv(path, index=False)
    return str(path)


def test_check_drift_flags_a_shift(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    ref = _write(tmp_path / "ref.csv", rng.normal(700, 50, 500))
    new = _write(tmp_path / "new.csv", rng.normal(850, 50, 500))

    assert check_drift_and_alert(ref, new) is True


def test_check_drift_passes_a_stable_sample(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    ref = _write(tmp_path / "ref.csv", rng.normal(700, 50, 500))
    new = _write(tmp_path / "new.csv", rng.normal(700, 50, 500))

    assert check_drift_and_alert(ref, new) is False


def test_threshold_is_respected(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    ref = _write(tmp_path / "ref.csv", rng.normal(700, 50, 500))
    new = _write(tmp_path / "new.csv", rng.normal(720, 50, 500))
    psi = calculate_psi(pd.read_csv(ref)["bookings"], pd.read_csv(new)["bookings"])

    assert check_drift_and_alert(ref, new, threshold=psi + 0.01) is False
    assert check_drift_and_alert(ref, new, threshold=psi - 0.01) is True
