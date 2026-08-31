"""Population Stability Index drift detector.

PSI is the standard distribution drift metric used by financial
forecasting teams. Thresholds:
  PSI < 0.1   stable
  PSI 0.1 to 0.2   monitor
  PSI > 0.2   retrain trigger

Phase 5 fills in the body.
"""
from __future__ import annotations


def calculate_psi(expected, actual, buckets: int = 10) -> float:
    raise NotImplementedError("Implemented in Phase 5.")


def check_drift_and_alert(training_data_path: str, new_data_path: str, threshold: float = 0.2) -> bool:
    raise NotImplementedError("Implemented in Phase 5.")
