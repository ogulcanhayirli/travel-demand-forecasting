"""Aggregate raw hotel bookings into the weekly demand series.

Reads the Kaggle Hotel Booking Demand CSV, applies the cleaning steps from
notebooks/01_eda.ipynb, counts non-cancelled arrivals per Monday-start week,
and writes data/processed/weekly_demand.csv.

Only weeks fully covered by the source data are kept. The dataset starts on
Wednesday 1 July 2015 and ends on Thursday 31 August 2017, so the first and
last calendar weeks contain only a few days of arrivals. Left in, the last
one looks like a demand collapse, lands in the test set, and seeds the
recursive scenario forecast.

Run locally:
    python src/data/build_weekly_demand.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = "data/raw/hotel_bookings.csv"
OUTPUT_PATH = "data/processed/weekly_demand.csv"

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Build arrival_date and drop the records the EDA flagged as invalid."""
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["children"] = df["children"].fillna(0)
    zero_guests = (df["adults"] + df["children"] + df["babies"]) == 0
    extreme_adr = df["adr"] > 5000
    return df[~zero_guests & ~extreme_adr].reset_index(drop=True)


def drop_incomplete_weeks(
    weekly: pd.DataFrame,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only weeks whose seven days all fall inside the source data range.

    Parameters
    ----------
    weekly : DataFrame with a Monday `week_start` column.
    coverage_start, coverage_end : First and last calendar day the source
        data covers (inclusive).
    """
    week_end = weekly["week_start"] + pd.Timedelta(days=6)
    complete = (weekly["week_start"] >= coverage_start) & (week_end <= coverage_end)
    for ws in weekly.loc[~complete, "week_start"]:
        log.info(f"Dropping incomplete week starting {ws.date()}")
    return weekly[complete].reset_index(drop=True)


def aggregate_weekly(bookings: pd.DataFrame) -> pd.DataFrame:
    """Count non-cancelled arrivals per Monday-start week, full weeks only.

    Coverage is taken from all bookings (cancelled included), since the
    source range is a property of the extract, not of the arrivals.
    """
    arrivals = bookings[bookings["is_canceled"] == 0]
    week_start = arrivals["arrival_date"] - pd.to_timedelta(
        arrivals["arrival_date"].dt.dayofweek, unit="D"
    )
    weekly = (
        week_start.value_counts()
        .rename_axis("week_start")
        .rename("bookings")
        .sort_index()
        .reset_index()
    )
    return drop_incomplete_weeks(
        weekly,
        coverage_start=bookings["arrival_date"].min(),
        coverage_end=bookings["arrival_date"].max(),
    )


def build_weekly_demand(raw_path: str = RAW_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    log.info(f"Loaded {len(raw):,} bookings from {raw_path}")
    bookings = clean_bookings(raw)
    log.info(f"{len(bookings):,} bookings after cleaning")
    weekly = aggregate_weekly(bookings)
    log.info(
        f"Weekly series: {len(weekly)} weeks, "
        f"{weekly['week_start'].min().date()} to {weekly['week_start'].max().date()}"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    log.info(f"Saved to {output_path}")
    return weekly


if __name__ == "__main__":
    build_weekly_demand()
