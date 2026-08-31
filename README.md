# Travel Demand Forecasting Platform

![CI](https://github.com/ogulcanhayirli/travel-demand-forecasting/actions/workflows/ci.yml/badge.svg)

Weekly demand forecasting for hotel bookings, built to practise the parts of ML
engineering that do not show up in a notebook: time-based validation that does not
leak, an automated promotion rule that refuses to ship models whose improvement is
within noise, and a retraining pipeline that runs without anyone remembering to run it.

The dataset is public and modest. What is being demonstrated is the surrounding
engineering, not the forecast accuracy.

**Live demo:** [travel-demand-forecasting.streamlit.app](https://travel-demand-forecasting.streamlit.app)

---

## Business Problem

Travel platforms need accurate, scenario-aware demand forecasts to drive financial planning, budgeting, and resource allocation. Finance teams want a single source of truth that produces pessimistic, baseline, and optimistic views of weekly booking volumes, with a transparent retraining policy they can trust.

This platform answers: *"How many bookings should we plan for over the next 6 months, and what are the upside and downside scenarios?"*

---

## What This System Does

1. Ingests hotel booking records and aggregates them into a weekly demand time series
2. Engineers lag, rolling statistics, and calendar features for gradient boosting
3. Trains two competing models: Prophet (additive seasonality) and LightGBM (lag features)
4. Runs a champion-challenger evaluation — the new model is only promoted if it beats the current one by at least 2 MAPE percentage points
5. Produces three 26-week scenario forecasts (pessimistic, baseline, optimistic) via recursive multi-step inference
6. Serves everything through an interactive Streamlit dashboard with a downloadable forecast table

---

## Architecture

```
Kaggle Hotel Booking Data
          |
          v
    S3 (raw + processed)
          |
          v
  Airflow DAG (runs every Monday 06:00 UTC)
          |
    ingest -> validate -> train_lgbm  \
                       -> train_prophet -> evaluate -> branch
                                                         |
                                              promote or retain champion
                                                         |
                                               SageMaker Model Registry
                                                         |
                                            Streamlit Dashboard (public)
```

---

## Models and Results

| Model      | MAPE   | MAE          | RMSE  | Test Period        |
|------------|--------|--------------|-------|--------------------|
| LightGBM   | 11.15% | 58 bookings  | 105   | Jun 2017 - Aug 2017|
| Prophet    | not yet evaluated | — | — | — |

LightGBM is the current champion. Prophet is implemented and ready to run as the
challenger but has not yet been evaluated, because CmdStan does not build on Apple
Silicon and the SageMaker run is pending. The promotion logic and its tests are
exercised against synthetic metric pairs in `tests/`, so the champion-challenger
rule is verified even though the second model's numbers are outstanding.

The LightGBM model is trained on 90 weeks of data with 17 features. Top predictors are `trend_signal` (short vs long-term momentum), `lag_4w` (monthly autocorrelation), and `week_cos` (cyclical seasonality encoding).

---

## Scenario Analysis

Three scenarios are generated from the trained LightGBM model using recursive one-step-ahead forecasting:

| Scenario    | Demand Multiplier | Use Case                          |
|-------------|-------------------|-----------------------------------|
| Pessimistic | 0.80x             | Downside planning, stress testing |
| Baseline    | 1.00x             | Central forecast, budget target   |
| Optimistic  | 1.20x             | Upside planning, capacity ceiling |

---

## Feature Engineering

All features are constructed to avoid target leakage — rolling statistics use `shift(1)` so the current week's value is never visible to the model at training time.

| Feature Group     | Features                                                        |
|-------------------|-----------------------------------------------------------------|
| Lag features      | lag_1w, lag_2w, lag_4w, lag_8w, lag_12w                        |
| Rolling stats     | rolling_mean_4w/8w/12w, rolling_std_4w, trend_signal           |
| Calendar          | week_of_year, month, quarter, year, is_peak_season             |
| Cyclical encoding | week_sin, week_cos (sine/cosine of week number)                |

---

## Repo Layout

```
travel-demand-forecasting/
  data/
    processed/          weekly_demand.csv, scenarios.csv (committed for dashboard)
  notebooks/
    01_eda.ipynb        Senior-level EDA: stationarity, decomposition, PSI, leakage audit
  src/
    features/
      build_features.py Lag, rolling, and calendar feature engineering
    models/
      metrics.py        Shared MAPE, MAE, RMSE (no heavy dependencies)
      train_lightgbm.py LightGBM trainer with early stopping
      train_prophet.py  Prophet trainer (runs on SageMaker)
      evaluate.py       Champion-challenger logic with 2pp MAPE threshold
    scenarios/
      scenario_generator.py  26-week recursive forecast, 3 scenario tracks
  sagemaker/
    train_lgbm.py       SageMaker entry point for LightGBM (/opt/ml contract)
    train_prophet.py    SageMaker entry point for Prophet
    launch_training_job.py  Uploads data to S3, submits SKLearn training jobs
  airflow/
    dags/
      forecast_pipeline.py  Weekly DAG: ingest, validate, train, evaluate, branch
  dashboard/
    app.py              Streamlit scenario explorer with Plotly charts
    requirements.txt    Dashboard-only dependencies for Streamlit Cloud
  tests/
    test_features.py    20 unit tests: features, leakage guard, metrics
  models/
    lgbm_metrics.json   Latest evaluation metrics (committed for dashboard)
  requirements.txt      Full project dependencies
  SETUP_CHECKLIST.md    Step-by-step environment and AWS setup guide
```

---

## How to Run Locally

```bash
# 1. Create and activate virtual environment
python -m venv forecast-env
source forecast-env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the dataset (requires Kaggle API key)
kaggle datasets download -d jessemostipak/hotel-booking-demand -p data/raw
unzip data/raw/hotel-booking-demand.zip -d data/raw/

# 4. Run EDA and generate processed data
jupyter lab notebooks/01_eda.ipynb

# 5. Train LightGBM
python src/models/train_lightgbm.py

# 6. Generate scenario forecasts
python src/scenarios/scenario_generator.py

# 7. Launch the dashboard
streamlit run dashboard/app.py
```

---

## How to Run on AWS SageMaker

```bash
# Configure AWS credentials and .env file first (see SETUP_CHECKLIST.md)

# Train LightGBM on a managed ml.m5.large instance
python sagemaker/launch_training_job.py --model lgbm

# Train Prophet (compiles cleanly on Amazon Linux 2)
python sagemaker/launch_training_job.py --model prophet

# Train both in parallel
python sagemaker/launch_training_job.py --model both
```

---

## Running Tests

```bash
pytest tests/ -v
# 20 tests covering feature engineering, leakage prevention, and metric functions
```

---

## Tech Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| Modelling          | Prophet, LightGBM                               |
| Feature engineering| pandas, numpy                                   |
| Cloud training     | AWS SageMaker (Script Mode, SKLearn container)  |
| Artifact storage   | AWS S3                                          |
| Orchestration      | Apache Airflow (weekly DAG, BranchPythonOperator)|
| Dashboard          | Streamlit, Plotly                               |
| Testing            | pytest (20 unit tests)                          |
| Infrastructure     | AWS IAM, SageMaker Execution Role               |

---

## Key Design Decisions

**Time-based train/test split** — never random for time series. The last 12 weeks are held out as the test set, simulating real forecast evaluation.

**Champion-challenger promotion threshold** — the challenger must beat the champion by at least 2 MAPE percentage points, not just marginally better. This prevents promoting models whose improvement is within noise.

**Lag selection** — the 52-week lag was excluded because it drops the first year of data as NaN warmup rows. With only ~2 years of data, this halved the training set and caused severe underfitting (best_iteration=2). Lags at 1, 2, 4, 8, 12 weeks provide short and medium-range autocorrelation without sacrificing data.

**Sine/cosine week encoding** — raw week number (1-52) would tell the model that week 52 and week 1 are 51 steps apart. Sine/cosine encoding makes them adjacent, which is correct.

**Shared metrics module** — MAPE, MAE, and RMSE live in `src/models/metrics.py` with no heavy dependencies so they can be imported and unit tested without installing lightgbm or prophet.

---

## Data

Kaggle Hotel Booking Demand dataset (Mostipak, 2020) — 119,390 bookings from a Portuguese city hotel and resort hotel, covering July 2015 to August 2017. Aggregated to 114 weeks of non-cancelled weekly arrivals.
