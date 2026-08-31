"""SageMaker training entry point for Prophet.

Identical contract to train_lgbm.py but calls train_prophet instead.
Prophet is trained on SageMaker (Amazon Linux 2) rather than locally
because CmdStan compiles cleanly on that platform.

Instance recommendation: ml.m5.large (2 vCPU, 8 GB RAM).
Prophet training on ~100 weeks of data completes in under 2 minutes.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "data/processed")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "models")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train_prophet import train_prophet


def parse_hyperparameters() -> dict:
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hp_path.exists():
        with open(hp_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    log.info("=== SageMaker Prophet Training Job Starting ===")
    log.info(f"Input channel : {SM_CHANNEL_TRAIN}")
    log.info(f"Model dir     : {SM_MODEL_DIR}")

    data_path = str(Path(SM_CHANNEL_TRAIN) / "weekly_demand.csv")
    model_output_path = str(Path(SM_MODEL_DIR) / "prophet_model.pkl")
    metrics_output_path = str(Path(SM_OUTPUT_DATA_DIR) / "prophet_metrics.json")

    hps = parse_hyperparameters()
    test_size = int(hps.get("test_size", 12))

    metrics = train_prophet(
        data_path=data_path,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        test_size=test_size,
    )

    log.info("=== Training Complete ===")
    log.info(f"MAPE: {metrics['mape']:.2f}%  MAE: {metrics['mae']:.1f}  RMSE: {metrics['rmse']:.1f}")

    print(f"[Prophet] mape={metrics['mape']:.4f};mae={metrics['mae']:.4f};rmse={metrics['rmse']:.4f}")
