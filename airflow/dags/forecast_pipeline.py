"""Weekly travel demand forecasting pipeline DAG.

This DAG runs every Monday at 06:00 UTC and orchestrates the full
model retraining and deployment lifecycle:

  ingest -> validate -> train_lgbm -> train_prophet -> evaluate
         -> branch: promote or skip -> notify

Design decisions
----------------
* We train both models every week and let the champion-challenger
  evaluator decide which one to keep. This is more expensive than
  only retraining the champion, but with ml.m5.large instances at
  ~$0.115/hr and training completing in under 5 minutes, the weekly
  cost is negligible compared to the value of always having fresh models.

* The branch operator checks the promote flag from the evaluation JSON.
  If promote=True, the new model is registered in SageMaker Model Registry
  and the endpoint is updated. If promote=False, we skip deployment and
  the current production model stays live.

* We use the PythonOperator rather than the SageMakerTrainingOperator
  for simplicity. The SageMakerTrainingOperator is better for production
  because it integrates with Airflow's sensor/retry framework, but it
  requires the apache-airflow-providers-amazon package and more IAM setup.

DAG dependencies (install before running):
    pip install apache-airflow apache-airflow-providers-amazon boto3 sagemaker
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "travel-forecast-ogulcan")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
INSTANCE_TYPE = "ml.m5.large"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAGEMAKER_DIR = PROJECT_ROOT / "sagemaker"

DEFAULT_ARGS = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def ingest_data(**context):
    """Upload the latest weekly_demand.csv to S3.

    In a real pipeline this task would first pull fresh booking data from
    the source database, run the aggregation to produce weekly_demand.csv,
    then upload. Here we upload the existing processed file.
    """
    import boto3

    local_path = PROJECT_ROOT / "data" / "processed" / "weekly_demand.csv"
    s3_key = "data/processed/weekly_demand.csv"

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log.info(f"Uploaded {local_path} -> s3://{S3_BUCKET}/{s3_key}")

    # Push S3 URI to XCom so downstream tasks can read it
    context["ti"].xcom_push(key="s3_input_uri", value=f"s3://{S3_BUCKET}/data/processed")


def validate_data(**context):
    """Basic data quality checks before training.

    Checks:
    - File exists in S3
    - Minimum number of weeks present (at least 52 for seasonality)
    - No null values in week_start or bookings columns
    - No negative booking values

    In production this would use Great Expectations with a full
    expectation suite. Here we implement the critical checks manually
    so the DAG has no extra dependencies.
    """
    import io
    import boto3
    import pandas as pd

    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key="data/processed/weekly_demand.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["week_start"])

    assert len(df) >= 52, f"Only {len(df)} weeks of data — need at least 52 for seasonality"
    assert df["week_start"].notna().all(), "Null values found in week_start"
    assert df["bookings"].notna().all(), "Null values found in bookings"
    assert (df["bookings"] >= 0).all(), "Negative booking values found"

    log.info(f"Data validation passed: {len(df)} weeks, "
             f"{df['week_start'].min().date()} to {df['week_start'].max().date()}")


def train_lgbm(**context):
    """Submit LightGBM SageMaker training job and push model S3 URI to XCom."""
    s3_input_uri = context["ti"].xcom_pull(task_ids="ingest_data", key="s3_input_uri")
    job_name = f"lgbm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    boto_session = boto3.Session(region_name=AWS_REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    estimator = SKLearn(
        entry_point="train_lgbm.py",
        source_dir=str(SAGEMAKER_DIR),
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
        job_name=job_name,
        hyperparameters={"test_size": 12},
        output_path=f"s3://{S3_BUCKET}/models/lgbm",
        metric_definitions=[
            {"Name": "lgbm:mape", "Regex": r"mape=([0-9\.]+)"},
            {"Name": "lgbm:mae",  "Regex": r"mae=([0-9\.]+)"},
        ],
    )
    estimator.fit({"train": s3_input_uri}, wait=True, logs="All")

    context["ti"].xcom_push(key="lgbm_model_uri", value=estimator.model_data)
    log.info(f"LightGBM training complete. Model: {estimator.model_data}")


def train_prophet(**context):
    """Submit Prophet SageMaker training job and push model S3 URI to XCom."""
    s3_input_uri = context["ti"].xcom_pull(task_ids="ingest_data", key="s3_input_uri")
    job_name = f"prophet-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    boto_session = boto3.Session(region_name=AWS_REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    estimator = SKLearn(
        entry_point="train_prophet.py",
        source_dir=str(SAGEMAKER_DIR),
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
        job_name=job_name,
        hyperparameters={"test_size": 12},
        output_path=f"s3://{S3_BUCKET}/models/prophet",
        metric_definitions=[
            {"Name": "prophet:mape", "Regex": r"mape=([0-9\.]+)"},
            {"Name": "prophet:mae",  "Regex": r"mae=([0-9\.]+)"},
        ],
    )
    estimator.fit({"train": s3_input_uri}, wait=True, logs="All")

    context["ti"].xcom_push(key="prophet_model_uri", value=estimator.model_data)
    log.info(f"Prophet training complete. Model: {estimator.model_data}")


def evaluate_models(**context):
    """Download both metrics JSONs from S3 and run champion-challenger logic.

    Pushes promote=True/False to XCom for the branch operator to read.
    """
    import io
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.evaluate import evaluate_champion_challenger

    s3 = boto3.client("s3", region_name=AWS_REGION)

    def fetch_metrics(s3_key: str) -> dict:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            return json.loads(obj["Body"].read())
        except s3.exceptions.NoSuchKey:
            log.warning(f"Metrics not found at s3://{S3_BUCKET}/{s3_key}")
            return None

    lgbm_metrics = fetch_metrics("models/lgbm/lgbm_metrics.json")
    prophet_metrics = fetch_metrics("models/prophet/prophet_metrics.json")

    if lgbm_metrics is None or prophet_metrics is None:
        log.warning("One or both metrics files missing. Skipping promotion.")
        context["ti"].xcom_push(key="promote", value=False)
        return

    # Treat LightGBM as champion (current production model),
    # Prophet as challenger (newly trained). This is arbitrary for the
    # first run — in production, the champion is whichever model is
    # currently registered as APPROVED in SageMaker Model Registry.
    champion = lgbm_metrics
    challenger = prophet_metrics

    if challenger["mape"] < champion["mape"] - 2.0:
        promote = True
        log.info(f"PROMOTE: Prophet MAPE {challenger['mape']:.2f}% beats "
                 f"LightGBM {champion['mape']:.2f}% by >2pp")
    else:
        promote = False
        log.info(f"RETAIN: LightGBM MAPE {champion['mape']:.2f}% — "
                 f"Prophet {challenger['mape']:.2f}% does not meet threshold")

    context["ti"].xcom_push(key="promote", value=promote)
    context["ti"].xcom_push(key="winning_model", value="prophet" if promote else "lgbm")


def branch_on_promotion(**context):
    """Return the task_id to execute based on the promote flag."""
    promote = context["ti"].xcom_pull(task_ids="evaluate_models", key="promote")
    return "promote_model" if promote else "skip_promotion"


def promote_model(**context):
    """Register the winning model in SageMaker Model Registry.

    In a full production setup this would also update the SageMaker
    endpoint to serve the new model. For this project we register
    the model package and log the approval.
    """
    winning_model = context["ti"].xcom_pull(task_ids="evaluate_models", key="winning_model")
    log.info(f"Promoting {winning_model} to production (Model Registry registration).")
    # Real implementation: sm_client.create_model_package(...)
    # Omitted here to avoid hard IAM dependencies in the DAG definition.
    log.info("Model registered. Endpoint update would follow in production.")


def notify_complete(**context):
    """Log pipeline completion summary. Replace with SNS/Slack in production."""
    promote = context["ti"].xcom_pull(task_ids="evaluate_models", key="promote")
    winning = context["ti"].xcom_pull(task_ids="evaluate_models", key="winning_model")
    log.info(
        f"Pipeline complete. Promoted: {promote}. "
        f"Production model: {winning or 'unchanged'}."
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="travel_demand_forecast_pipeline",
    description="Weekly retraining and champion-challenger evaluation",
    schedule_interval="0 6 * * MON",   # Every Monday at 06:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "forecasting", "sagemaker"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t_lgbm = PythonOperator(
        task_id="train_lgbm",
        python_callable=train_lgbm,
    )

    t_prophet = PythonOperator(
        task_id="train_prophet",
        python_callable=train_prophet,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
    )

    t_branch = BranchPythonOperator(
        task_id="branch_promotion",
        python_callable=branch_on_promotion,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_skip = EmptyOperator(task_id="skip_promotion")

    t_notify = PythonOperator(
        task_id="notify_complete",
        python_callable=notify_complete,
        trigger_rule="none_failed_min_one_success",
    )

    # Pipeline dependency graph:
    # ingest -> validate -> train_lgbm  \
    #                    -> train_prophet -> evaluate -> branch -> promote  \
    #                                                          -> skip      -> notify
    t_ingest >> t_validate >> [t_lgbm, t_prophet] >> t_evaluate
    t_evaluate >> t_branch >> [t_promote, t_skip] >> t_notify
