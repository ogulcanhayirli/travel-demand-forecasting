"""Launch SageMaker training jobs for LightGBM and/or Prophet.

This script:
  1. Uploads the processed weekly_demand.csv to S3
  2. Submits a SageMaker SKLearn/Script Mode training job
  3. Waits for completion and prints the final metrics

Usage
-----
    # Train LightGBM only (fast, runs locally too):
    python sagemaker/launch_training_job.py --model lgbm

    # Train Prophet only:
    python sagemaker/launch_training_job.py --model prophet

    # Train both, one after the other:
    python sagemaker/launch_training_job.py --model both

Prerequisites
-------------
    pip install -r requirements-aws.txt
    AWS credentials configured (aws configure or IAM role)
    .env file with AWS_S3_BUCKET and SAGEMAKER_ROLE_ARN

Why Script Mode?
----------------
SageMaker Script Mode lets you bring your own Python script and run it on a
managed instance without building a custom Docker image. SageMaker provides
pre-built containers with scikit-learn, pandas and numpy.
LightGBM and Prophet are not in that container, so sagemaker/requirements.txt
(installed automatically at job start) adds them.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config: read from environment (set in .env or export before running)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is optional; exported variables still work
    pass

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "travel-forecast-ogulcan")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
INSTANCE_TYPE = "ml.m5.large"  # 2 vCPU, 8 GB, sufficient for this dataset size
FRAMEWORK_VERSION = "1.4-2"
# The entry points import the shared training code from src/, which lives
# outside source_dir. SageMaker copies each dependency into the job's code
# directory next to the entry point.
SAGEMAKER_DIR = Path(__file__).resolve().parent
SRC_DIR = SAGEMAKER_DIR.parent / "src"


def upload_data_to_s3(local_path: str, s3_prefix: str = "data/processed") -> str:
    """Upload weekly_demand.csv to S3 and return the S3 URI.

    We upload every time so that retraining jobs always use the latest data.
    S3 versioning (enabled on the bucket) keeps the previous versions as a
    safety net in case a bad data file is uploaded.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    filename = Path(local_path).name
    s3_key = f"{s3_prefix}/{filename}"

    log.info(f"Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    s3_uri = f"s3://{S3_BUCKET}/{s3_prefix}"
    log.info(f"Upload complete. Input channel URI: {s3_uri}")
    return s3_uri


def launch_lgbm_job(s3_input_uri: str, session: sagemaker.Session) -> SKLearn:
    """Submit a SageMaker training job for LightGBM and wait for it."""
    job_name = f"lgbm-travel-demand-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log.info(f"Launching LightGBM job: {job_name}")

    estimator = SKLearn(
        entry_point="train_lgbm.py",
        source_dir=str(SAGEMAKER_DIR),
        dependencies=[str(SRC_DIR)],
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={"test_size": 12},
        output_path=f"s3://{S3_BUCKET}/models/lgbm",
        # Metric definitions let SageMaker push these to CloudWatch automatically
        metric_definitions=[
            {"Name": "lgbm:mape", "Regex": r"mape=([0-9\.]+)"},
            {"Name": "lgbm:mae",  "Regex": r"mae=([0-9\.]+)"},
            {"Name": "lgbm:rmse", "Regex": r"rmse=([0-9\.]+)"},
        ],
    )

    # job_name is a fit() argument; the estimator constructor ignores it
    estimator.fit({"train": s3_input_uri}, job_name=job_name, wait=True, logs="All")
    log.info(f"LightGBM job complete. Model at: {estimator.model_data}")
    return estimator


def launch_prophet_job(s3_input_uri: str, session: sagemaker.Session) -> SKLearn:
    """Submit a SageMaker training job for Prophet and wait for it."""
    job_name = f"prophet-travel-demand-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log.info(f"Launching Prophet job: {job_name}")

    estimator = SKLearn(
        entry_point="train_prophet.py",
        source_dir=str(SAGEMAKER_DIR),
        dependencies=[str(SRC_DIR)],
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={"test_size": 12},
        output_path=f"s3://{S3_BUCKET}/models/prophet",
        metric_definitions=[
            {"Name": "prophet:mape", "Regex": r"mape=([0-9\.]+)"},
            {"Name": "prophet:mae",  "Regex": r"mae=([0-9\.]+)"},
            {"Name": "prophet:rmse", "Regex": r"rmse=([0-9\.]+)"},
        ],
    )

    # job_name is a fit() argument; the estimator constructor ignores it
    estimator.fit({"train": s3_input_uri}, job_name=job_name, wait=True, logs="All")
    log.info(f"Prophet job complete. Model at: {estimator.model_data}")
    return estimator


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument(
        "--model",
        choices=["lgbm", "prophet", "both"],
        default="lgbm",
        help="Which model to train (default: lgbm)",
    )
    parser.add_argument(
        "--data",
        default="data/processed/weekly_demand.csv",
        help="Local path to weekly_demand.csv",
    )
    args = parser.parse_args()

    if not ROLE_ARN:
        raise ValueError(
            "SAGEMAKER_ROLE_ARN environment variable not set. "
            "Export it or add it to .env before running."
        )

    boto_session = boto3.Session(region_name=AWS_REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    # Upload data to S3 once; both jobs share the same input URI
    s3_input_uri = upload_data_to_s3(args.data)

    if args.model in ("lgbm", "both"):
        launch_lgbm_job(s3_input_uri, sm_session)

    if args.model in ("prophet", "both"):
        launch_prophet_job(s3_input_uri, sm_session)

    log.info("All requested training jobs complete.")


if __name__ == "__main__":
    main()
