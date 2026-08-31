"""SageMaker training entry point for LightGBM.

SageMaker calls this script when a training job starts. It reads data from
the SageMaker-mounted input channel, calls the same train_lightgbm function
used locally, and writes the model artifact to the SageMaker model directory.

SageMaker directory contract
----------------------------
  /opt/ml/input/data/train/   <- S3 training data is mounted here
  /opt/ml/model/              <- anything saved here gets packaged as model.tar.gz
  /opt/ml/output/             <- for non-model outputs (metrics, logs)

Hyperparameters are passed as environment variables prefixed SM_HP_ or read
from /opt/ml/input/config/hyperparameters.json. We read them via os.environ
so the same script works for both local SageMaker mode and cloud jobs.

To launch this job from Python, see sagemaker/launch_training_job.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# SageMaker injects /opt/ml paths; locally we fall back to project paths
SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "data/processed")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "models")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Make project src importable (handles both SageMaker and local execution)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train_lightgbm import train_lightgbm


def parse_hyperparameters() -> dict:
    """Read hyperparameters from SageMaker environment or config file.

    SageMaker passes hyperparameters as env vars like SM_HP_NUM_LEAVES=31,
    and also writes them to /opt/ml/input/config/hyperparameters.json.
    We prefer the JSON file when it exists (cloud), fall back to defaults.
    """
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            raw = json.load(f)
        log.info(f"Loaded hyperparameters from {hp_path}: {raw}")
        return {k: _cast(v) for k, v in raw.items()}

    # Local mode: read from SM_HP_* env vars
    hps = {}
    for key, val in os.environ.items():
        if key.startswith("SM_HP_"):
            hps[key[6:].lower()] = _cast(val)
    if hps:
        log.info(f"Loaded hyperparameters from environment: {hps}")
    return hps


def _cast(value: str):
    """Try to cast a hyperparameter string to int or float."""
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


if __name__ == "__main__":
    log.info("=== SageMaker LightGBM Training Job Starting ===")
    log.info(f"Input channel : {SM_CHANNEL_TRAIN}")
    log.info(f"Model dir     : {SM_MODEL_DIR}")

    data_path = str(Path(SM_CHANNEL_TRAIN) / "weekly_demand.csv")
    model_output_path = str(Path(SM_MODEL_DIR) / "lgbm_model.pkl")
    metrics_output_path = str(Path(SM_OUTPUT_DATA_DIR) / "lgbm_metrics.json")

    hps = parse_hyperparameters()
    test_size = int(hps.get("test_size", 12))

    metrics = train_lightgbm(
        data_path=data_path,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        test_size=test_size,
    )

    log.info("=== Training Complete ===")
    log.info(f"MAPE: {metrics['mape']:.2f}%  MAE: {metrics['mae']:.1f}  RMSE: {metrics['rmse']:.1f}")

    # SageMaker picks up metrics from stdout in this format for CloudWatch
    print(f"[LightGBM] mape={metrics['mape']:.4f};mae={metrics['mae']:.4f};rmse={metrics['rmse']:.4f}")
