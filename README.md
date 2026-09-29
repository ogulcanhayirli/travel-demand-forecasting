# Travel Demand Forecasting Platform

[![CI](https://github.com/ogulcanhayirli/travel-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/ogulcanhayirli/travel-demand-forecasting/actions/workflows/ci.yml)

Weekly demand forecasting for hotel bookings, built to practise the parts of ML
engineering that do not show up in a notebook: time-based validation that does not
leak, honest comparison against naive baselines, and an automated promotion rule that
refuses to ship models whose improvement is within noise.

The dataset is public and modest. What is being demonstrated is the surrounding
engineering, not the forecast accuracy.

**Live demo:** [travel-demand-forecasting.streamlit.app](https://travel-demand-forecasting.streamlit.app)

---

## Business Problem

Travel platforms need scenario-aware demand forecasts to drive financial planning, budgeting, and resource allocation. Finance teams want a single source of truth that produces pessimistic, baseline, and optimistic views of weekly booking volumes, with a transparent retraining policy they can trust.

This project answers: *"How many bookings should we plan for over the next 6 months, and what are the upside and downside scenarios?"*

---

## What This System Does

1. Aggregates hotel booking records into a weekly series of non-cancelled arrivals, keeping only weeks fully covered by the data
2. Engineers lag, rolling statistics, and calendar features for gradient boosting
3. Trains two competing models, LightGBM (lag features) and Prophet (additive seasonality), and scores both against naive baselines on the same test weeks
4. Runs a champion-challenger rule: the challenger is only promoted if it beats the champion by at least 2 MAPE percentage points
5. Produces three 26-week scenario forecasts (pessimistic, baseline, optimistic) via recursive multi-step inference
6. Serves everything through a Streamlit dashboard with a downloadable forecast table

---

## Status

| Component | State |
|-----------|-------|
| Weekly aggregation, features, LightGBM, Prophet, baselines, champion-challenger rule | Implemented and run locally; results below |
| Scenario forecasts and Streamlit dashboard | Implemented and run locally |
| Unit tests | 36 tests, run by GitHub Actions on every push and pull request |
| SageMaker Script Mode entry points (`sagemaker/`) | Run on SageMaker once, launched manually (29 September 2026); both jobs succeeded (see below) |
| Airflow DAG (`airflow/dags/`) | Defined. **Not yet run.** The Model Registry step only logs, and the evaluate step reads metrics from S3 keys the jobs do not write yet (they end up inside `output.tar.gz`) |
| Drift monitoring (`src/monitoring/`) | Not implemented yet. PSI is computed in the EDA notebook only |

---

## Architecture

The intended production flow. Only the local path (aggregation, training, evaluation,
scenarios, dashboard) has been run so far.

```
Kaggle Hotel Booking Data
          |
          v
    S3 (raw + processed)
          |
          v
  Airflow DAG (scheduled for Mondays 06:00 UTC)
          |
    ingest -> validate -> train_lgbm  \
                       -> train_prophet -> evaluate -> branch
                                                         |
                                              promote or retain champion
                                                         |
                                  SageMaker Model Registry (stub, logs only)
                                                         |
                                            Streamlit Dashboard (public)
```

---

## Models and Results

All models are scored on the same 12 held-out weeks, 5 June 2017 to 21 August 2017.

| Model                          | MAPE  | MAE  | RMSE | How the test weeks are forecast               |
|--------------------------------|-------|------|------|-----------------------------------------------|
| Naive: last week (lag 1)       | 5.27% | 38.1 | 54.3 | One week ahead                                |
| LightGBM (champion)            | 5.75% | 42.4 | 53.8 | One week ahead, lag features read from actuals|
| Prophet (challenger)           | 7.66% | 57.6 | 82.8 | 1 to 12 weeks ahead, no test data seen        |
| Naive: same week last year     | 8.72% | 65.3 | 81.6 | Uses the actual from 52 weeks earlier         |

Numbers come from `models/lgbm_metrics.json`, `models/prophet_metrics.json` and
`models/promotion_decision.json`, all produced by the code in this repo.

**What the table says.** On this 12-week window, LightGBM does not beat the naive
"last week" forecast on MAPE or MAE; it is only marginally better on RMSE. Weekly
arrivals in summer 2017 were stable, so persistence is a strong baseline, and 12
weeks is a small sample for separating models that are this close. Prophet is not
directly comparable to the other rows: it forecasts all 12 weeks from the end of the
training period, whereas the others see the actual value of every previous week.
Prophet also warns that its yearly seasonality is fitted on under two years of
history (693 days).

**Promotion decision.** Prophet's MAPE is 1.90 points worse than LightGBM's, so the
rule retains LightGBM as champion. LightGBM therefore drives the scenario forecasts.

The LightGBM model uses 17 features and 88 training weeks. The tree count (39) is
chosen by early stopping on the last 12 of those weeks, and the model is then refit on
all 88. The most used features (by number of splits) are `rolling_std_4w`, `week_cos`
and `lag_1w`.

---

## Engineering Fixes Worth Knowing About

**Partial weeks.** The raw extract runs from Wednesday 1 July 2015 to Thursday
31 August 2017, so the first and last Monday-start weeks contained only a few days of
arrivals. The last one (379 arrivals against roughly 720 in the weeks before it) sat
in the test set and was the starting point for the recursive scenario forecast.
`src/data/build_weekly_demand.py` now keeps only weeks whose seven days all fall inside
the data range, by rule rather than by a hardcoded date, and a test covers it. The
series is 112 complete weeks.

**Early stopping on the test set.** An earlier version passed the test weeks to
LightGBM's early stopping, so the number of trees was chosen by looking at test data.
Early stopping now uses a validation window at the end of the training period. The
previously reported LightGBM figure (11.15% MAPE) came from that setup and the partial
week, and is superseded by the table above.

**Train/serve skew in scenarios.** The scenario generator built rolling features one
week staler than training did. Inference features are now built by one function that
a test checks against the training features for every week.

---

## Scenario Analysis

Three scenarios are generated from the champion LightGBM model using recursive
one-step-ahead forecasting, starting the week after the last complete week
(28 August 2017):

| Scenario    | Demand Multiplier | Use Case                          |
|-------------|-------------------|-----------------------------------|
| Pessimistic | 0.80x             | Downside planning, stress testing |
| Baseline    | 1.00x             | Central forecast, budget target   |
| Optimistic  | 1.20x             | Upside planning, capacity ceiling |

The multipliers are fixed planning assumptions, not statistically derived intervals.

---

## Feature Engineering

All features are constructed to avoid target leakage. Rolling statistics use `shift(1)`
so the current week's value is never visible to the model at training time.

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
  .github/workflows/
    ci.yml              Runs pytest on push and pull request
  data/
    processed/          weekly_demand.csv, scenarios.csv (committed for dashboard)
  notebooks/
    01_eda.ipynb        EDA: data checks, stationarity, decomposition, PSI, leakage audit
  src/
    data/
      build_weekly_demand.py  Raw bookings to weekly series, complete weeks only
    features/
      build_features.py Lag, rolling, and calendar feature engineering
    models/
      metrics.py        Shared MAPE, MAE, RMSE (no heavy dependencies)
      train_lightgbm.py LightGBM trainer with early stopping and naive baselines
      train_prophet.py  Prophet trainer
      evaluate.py       Champion-challenger logic with 2pp MAPE threshold
    scenarios/
      scenario_generator.py  26-week recursive forecast, 3 scenario tracks
    monitoring/
      drift_detector.py Placeholder, not implemented yet
  sagemaker/
    train_lgbm.py       SageMaker entry point for LightGBM (/opt/ml contract)
    train_prophet.py    SageMaker entry point for Prophet
    launch_training_job.py  Uploads data to S3, submits SKLearn training jobs
    requirements.txt    Installed inside the training container (lightgbm, prophet)
  airflow/
    dags/
      forecast_pipeline.py  Weekly DAG: ingest, validate, train, evaluate, branch
  dashboard/
    app.py              Streamlit scenario explorer with Plotly charts
    requirements.txt    Pinned dashboard dependencies for Streamlit Cloud
  tests/                36 unit tests: aggregation, features, leakage, baselines,
                        scenario features, metrics, promotion rule
  models/
    lgbm_metrics.json        LightGBM and naive baseline metrics
    prophet_metrics.json     Prophet metrics
    promotion_decision.json  Champion-challenger decision
  requirements-dev.txt  Minimal dependencies to run the tests
  requirements-aws.txt  Dependencies to launch the SageMaker jobs
  requirements.txt      Full project dependencies (includes Airflow, SageMaker SDK, MLflow)
```

---

## How to Run Locally

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv forecast-env
source forecast-env/bin/activate

# 2. Install what the local pipeline needs (requirements.txt also pulls in
#    Airflow, MLflow and the SageMaker SDK, which are not needed here)
pip install -r requirements-dev.txt -r dashboard/requirements.txt prophet

# 3. Pull the dataset (requires Kaggle API key)
kaggle datasets download -d jessemostipak/hotel-booking-demand -p data/raw
unzip data/raw/hotel-booking-demand.zip -d data/raw/

# 4. Build the weekly series (complete weeks only)
python src/data/build_weekly_demand.py

# 5. Train both models and run the promotion rule
python src/models/train_lightgbm.py
python src/models/train_prophet.py
python src/models/evaluate.py --champion models/lgbm_metrics.json \
    --challenger models/prophet_metrics.json --output models/promotion_decision.json

# 6. Generate scenario forecasts
python src/scenarios/scenario_generator.py

# 7. Launch the dashboard
streamlit run dashboard/app.py
```

---

## SageMaker

`sagemaker/train_lgbm.py` and `sagemaker/train_prophet.py` are Script Mode entry
points that follow the `/opt/ml` directory contract and call the same training
functions as the local scripts. Each job runs on the SKLearn `1.4-2` container, ships
`src/` alongside the entry point, and installs `sagemaker/requirements.txt`
(lightgbm, prophet) at start.

Both jobs were run once on SageMaker on 29 September 2026, launched manually with
`launch_training_job.py --model both` in `us-east-1` on `ml.m5.large` instances:

| Job      | Result    | Billable time | Metrics printed by the job                 |
|----------|-----------|---------------|--------------------------------------------|
| LightGBM | Completed | 94 s          | MAPE 5.7519%, MAE 42.3961, RMSE 53.8329    |
| Prophet  | Completed | 99 s          | MAPE 7.6853%, MAE 57.8214, RMSE 82.9636    |

LightGBM matches the local run exactly. Prophet differs slightly from the local 7.66%
because the container installs prophet 1.1.7 (Python 3.10, pandas 2.3.2) while the
local runs used prophet 1.4.0; Prophet warns that its fit on under two years of history
depends on the Prophet/Stan version. The promotion decision is the same either way.
The model artifacts were written to the S3 bucket as `model.tar.gz`. The jobs are not
scheduled; the Airflow DAG that would schedule them has not been run.

```bash
# Configure AWS credentials (aws configure) and copy .env.example to .env first
pip install -r requirements-aws.txt
python sagemaker/launch_training_job.py --model lgbm
python sagemaker/launch_training_job.py --model prophet
python sagemaker/launch_training_job.py --model both   # one after the other
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## Tech Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| Modelling          | LightGBM, Prophet                               |
| Feature engineering| pandas, numpy                                   |
| Dashboard          | Streamlit, Plotly                               |
| Testing and CI     | pytest, GitHub Actions                          |
| Cloud training     | AWS SageMaker Script Mode (run manually once, not scheduled) |
| Orchestration      | Apache Airflow (DAG defined, not yet run)       |

---

## Key Design Decisions

**Time-based train/test split.** Never random for time series. The last 12 weeks are
held out as the test set, and early stopping uses a separate window at the end of the
training period, so the test weeks play no part in model selection.

**Naive baselines on the same test weeks.** A model is only interesting if it beats
"same as last week" and "same as last year". Both are stored next to the model metrics
and shown on the dashboard.

**Champion-challenger promotion threshold.** The challenger must beat the champion by
at least 2 MAPE percentage points, not just marginally. This prevents promoting models
whose improvement is within noise. The Airflow DAG calls the same tested function.

**Complete weeks only.** Aggregating to weeks is only valid when every week has seven
days of data; the rule is derived from the data range, not a hardcoded date.

**Lag selection.** A 52-week lag feature was left out because it turns the first
52 weeks into NaN warmup rows. With 112 weeks of data, that would leave 48 training
weeks after the 12-week test split. The same-week-last-year signal is still evaluated,
as a naive baseline.

**Sine/cosine week encoding.** A raw week number (1 to 52) tells the model that week 52
and week 1 are 51 steps apart. Sine/cosine encoding makes them adjacent.

**Shared metrics module.** MAPE, MAE, and RMSE live in `src/models/metrics.py` with no
heavy dependencies so they can be imported and unit tested on their own.

---

## Data

Kaggle Hotel Booking Demand dataset (Mostipak, 2020): 119,390 bookings from a
Portuguese city hotel and resort hotel, with arrivals from 1 July 2015 to
31 August 2017. Aggregated to 112 complete weeks of non-cancelled arrivals
(6 July 2015 to 21 August 2017).
