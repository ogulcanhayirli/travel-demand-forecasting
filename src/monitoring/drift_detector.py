"""Population Stability Index drift detector.

PSI measures how far a new sample's distribution has moved from a reference
sample. Thresholds commonly used in practice:
  PSI < 0.1        stable
  PSI 0.1 to 0.2   monitor
  PSI > 0.2        significant shift, retrain trigger

Bins are the reference sample's quantiles (right-closed, like pandas.qcut),
and the outermost bins are open ended, so new values below or above anything
seen in the reference still count (a shift out of the reference range is
exactly what this should catch).
Empty bins are floored at a small proportion so the log term stays finite.

Run locally:
    python src/monitoring/drift_detector.py \
        --reference data/processed/weekly_demand.csv \
        --current path/to/new_weekly_demand.csv
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", 0.2))
MIN_PROPORTION = 1e-4  # floor for empty bins


def calculate_psi(expected, actual, buckets: int = 10) -> float:
    """Population Stability Index of `actual` against the reference `expected`.

    Parameters
    ----------
    expected : Reference sample (e.g. the training period).
    actual : New sample to compare against the reference.
    buckets : Number of quantile bins taken from `expected`. Fewer bins are
        used when the reference has repeated values at the quantile edges.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        raise ValueError("expected and actual must each contain at least one value")
    if buckets < 1:
        raise ValueError("buckets must be at least 1")

    # Right-closed bins (a, b], open at both ends. np.histogram would put a
    # value equal to an edge in the bin above, which merges "equal to" and
    # "above" whenever the reference has ties at a quantile.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)[1:-1]))
    n_bins = edges.size + 1
    expected_pct = np.bincount(np.searchsorted(edges, expected), minlength=n_bins) / expected.size
    actual_pct = np.bincount(np.searchsorted(edges, actual), minlength=n_bins) / actual.size
    expected_pct = np.maximum(expected_pct, MIN_PROPORTION)
    actual_pct = np.maximum(actual_pct, MIN_PROPORTION)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def check_drift_and_alert(
    training_data_path: str,
    new_data_path: str,
    threshold: float = PSI_THRESHOLD,
    column: str = "bookings",
    buckets: int = 10,
) -> bool:
    """Compare `column` in two CSVs and log an alert if PSI exceeds `threshold`.

    Returns True when drift is detected (PSI > threshold).
    """
    reference = pd.read_csv(training_data_path)[column]
    current = pd.read_csv(new_data_path)[column]
    if len(current) < buckets:
        log.warning(
            f"Only {len(current)} new values for {buckets} bins; PSI will be noisy."
        )

    psi = calculate_psi(reference, current, buckets=buckets)
    drift = psi > threshold
    if drift:
        log.warning(f"DRIFT: PSI({column}) = {psi:.4f} > {threshold}. Consider retraining.")
    else:
        log.info(f"No drift: PSI({column}) = {psi:.4f} <= {threshold}.")
    return drift


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSI drift check between two CSVs")
    parser.add_argument("--reference", required=True, help="CSV with the reference data")
    parser.add_argument("--current", required=True, help="CSV with the new data")
    parser.add_argument("--column", default="bookings")
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    drifted = check_drift_and_alert(
        args.reference, args.current, args.threshold, args.column, args.buckets
    )
    raise SystemExit(1 if drifted else 0)
