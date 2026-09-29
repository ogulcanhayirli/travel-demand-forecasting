"""Tests for the weekly aggregation's complete-week rule.

The raw extract ends on Thursday 31 August 2017, so the final calendar week
held four days of arrivals and showed up as a fake demand collapse. These
tests pin the rule that replaced it: a week is kept only if all seven of its
days fall inside the source data range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_weekly_demand import aggregate_weekly


def make_bookings(start: str, end: str, per_day: int = 10) -> pd.DataFrame:
    """One non-cancelled and one cancelled booking pattern per day in [start, end]."""
    days = pd.date_range(start, end, freq="D")
    arrivals = pd.DataFrame({
        "arrival_date": days.repeat(per_day),
        "is_canceled": 0,
    })
    cancelled = pd.DataFrame({"arrival_date": days, "is_canceled": 1})
    return pd.concat([arrivals, cancelled], ignore_index=True)


def test_incomplete_trailing_week_is_dropped() -> None:
    # Mon 2017-08-07 to Thu 2017-08-31: the week of Mon 2017-08-28 has 4 days.
    weekly = aggregate_weekly(make_bookings("2017-08-07", "2017-08-31"))

    assert weekly["week_start"].max() == pd.Timestamp("2017-08-21")
    assert pd.Timestamp("2017-08-28") not in set(weekly["week_start"])


def test_incomplete_leading_week_is_dropped() -> None:
    # Starts Wed 2015-07-01, so the week of Mon 2015-06-29 has 5 days.
    weekly = aggregate_weekly(make_bookings("2015-07-01", "2015-07-26"))

    assert weekly["week_start"].min() == pd.Timestamp("2015-07-06")


def test_range_on_week_boundaries_keeps_every_week() -> None:
    # Mon 2017-08-07 to Sun 2017-08-27: three full weeks.
    weekly = aggregate_weekly(make_bookings("2017-08-07", "2017-08-27", per_day=10))

    assert list(weekly["week_start"]) == list(
        pd.to_datetime(["2017-08-07", "2017-08-14", "2017-08-21"])
    )
    assert (weekly["bookings"] == 70).all()


def test_cancelled_bookings_set_coverage_but_are_not_counted() -> None:
    """A final day with only cancellations still counts as covered."""
    bookings = pd.concat([
        make_bookings("2017-08-07", "2017-08-26"),
        pd.DataFrame({"arrival_date": [pd.Timestamp("2017-08-27")], "is_canceled": [1]}),
    ], ignore_index=True)

    weekly = aggregate_weekly(bookings)

    assert weekly["week_start"].max() == pd.Timestamp("2017-08-21")
    assert weekly["bookings"].iloc[-1] == 60
