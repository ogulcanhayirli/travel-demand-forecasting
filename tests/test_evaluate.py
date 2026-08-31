"""Tests for the champion-challenger promotion rule.

The promotion rule is the piece of this pipeline that decides what reaches
production, so it is tested against synthetic metric pairs rather than waiting
on a real training run. The case that matters most is a challenger that is
better but not better *enough*: promoting on noise is the failure this rule
exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import MIN_IMPROVEMENT_PCT, evaluate_champion_challenger


def write_metrics(path: Path, *, model: str, mape: float, mae: float = 50.0,
                  rmse: float = 100.0) -> Path:
    """Write a synthetic metrics JSON of the shape the trainers emit."""
    path.write_text(json.dumps({"model": model, "mape": mape, "mae": mae, "rmse": rmse}))
    return path


def run(tmp_path: Path, champion_mape: float, challenger_mape: float,
        **kwargs) -> dict:
    champion = write_metrics(tmp_path / "champ.json", model="lightgbm",
                             mape=champion_mape, rmse=kwargs.pop("champion_rmse", 100.0))
    challenger = write_metrics(tmp_path / "chall.json", model="prophet",
                               mape=challenger_mape,
                               rmse=kwargs.pop("challenger_rmse", 100.0))
    return evaluate_champion_challenger(
        str(champion), str(challenger), str(tmp_path / "decision.json"), **kwargs
    )


def test_clearly_better_challenger_is_promoted(tmp_path: Path) -> None:
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=8.0)

    assert decision["promote"] is True
    assert decision["delta_mape_pp"] == pytest.approx(4.0)


def test_marginal_improvement_is_not_promoted(tmp_path: Path) -> None:
    """A 1pp gain is inside the noise band the threshold exists to reject."""
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=11.0)

    assert decision["promote"] is False
    assert "does not meet" in decision["reason"]


def test_worse_challenger_is_not_promoted(tmp_path: Path) -> None:
    decision = run(tmp_path, champion_mape=10.0, challenger_mape=13.0)

    assert decision["promote"] is False
    assert "worse" in decision["reason"]


def test_improvement_exactly_at_the_threshold_is_promoted(tmp_path: Path) -> None:
    """Boundary case: the comparison is >=, so exactly 2.00pp promotes."""
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=12.0 - MIN_IMPROVEMENT_PCT)

    assert decision["delta_mape_pp"] == pytest.approx(MIN_IMPROVEMENT_PCT)
    assert decision["promote"] is True


def test_just_below_the_threshold_is_not_promoted(tmp_path: Path) -> None:
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=10.01)

    assert decision["promote"] is False


def test_custom_threshold_is_respected(tmp_path: Path) -> None:
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=11.0,
                   min_improvement_pct=0.5)

    assert decision["promote"] is True


def test_rmse_regression_raises_a_warning_on_an_otherwise_promoted_model(
    tmp_path: Path,
) -> None:
    """Better MAPE with much worse RMSE is the near-zero-actuals trap."""
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=8.0,
                   champion_rmse=100.0, challenger_rmse=125.0)

    assert decision["promote"] is True
    assert decision["secondary_warning"] is not None
    assert "RMSE" in decision["secondary_warning"]


def test_rmse_within_tolerance_produces_no_warning(tmp_path: Path) -> None:
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=8.0,
                   champion_rmse=100.0, challenger_rmse=105.0)

    assert decision["secondary_warning"] is None


def test_decision_is_written_to_disk(tmp_path: Path) -> None:
    """Airflow branches on the written file, not on the return value."""
    decision = run(tmp_path, champion_mape=12.0, challenger_mape=8.0)

    written = json.loads((tmp_path / "decision.json").read_text())

    assert written == decision
