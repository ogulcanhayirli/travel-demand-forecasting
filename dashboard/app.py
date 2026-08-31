"""Travel Demand Forecasting Dashboard.

Streamlit app that shows:
  1. Historical weekly demand with a 26-week scenario forecast
  2. Three scenario tracks: pessimistic, baseline, optimistic
  3. Model metrics card (LightGBM holdout evaluation)
  4. Feature importance bar chart
  5. Raw scenario table for export

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Travel Demand Forecaster",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths (relative to project root, where streamlit is launched from)
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/weekly_demand.csv")
SCENARIOS_PATH = Path("data/processed/scenarios.csv")
LGBM_METRICS_PATH = Path("models/lgbm_metrics.json")

# ---------------------------------------------------------------------------
# Data loaders (cached so re-renders don't re-read disk)
# ---------------------------------------------------------------------------
@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    return df.sort_values("week_start").reset_index(drop=True)


@st.cache_data
def load_scenarios() -> pd.DataFrame:
    df = pd.read_csv(SCENARIOS_PATH, parse_dates=["week_start"])
    return df.sort_values("week_start").reset_index(drop=True)


@st.cache_data
def load_metrics() -> dict:
    if LGBM_METRICS_PATH.exists():
        with open(LGBM_METRICS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

show_pessimistic = st.sidebar.checkbox("Show Pessimistic (×0.80)", value=True)
show_baseline = st.sidebar.checkbox("Show Baseline (×1.00)", value=True)
show_optimistic = st.sidebar.checkbox("Show Optimistic (×1.20)", value=True)

st.sidebar.markdown("---")
history_window = st.sidebar.slider(
    "Weeks of history to show",
    min_value=12,
    max_value=114,
    value=52,
    step=4,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
history = load_history()
scenarios = load_scenarios()
metrics = load_metrics()

history_trimmed = history.tail(history_window)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("✈️ Travel Demand Forecasting Platform")
st.markdown(
    "Weekly hotel booking demand — historical actuals and 26-week scenario forecasts "
    "produced by a LightGBM model with lag and calendar features."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", metrics.get("model", "LightGBM").upper())
with col2:
    mape = metrics.get("mape")
    st.metric("MAPE (test)", f"{mape:.2f}%" if mape else "N/A")
with col3:
    mae = metrics.get("mae")
    st.metric("MAE (test)", f"{mae:.0f} bookings" if mae else "N/A")
with col4:
    rmse = metrics.get("rmse")
    st.metric("RMSE (test)", f"{rmse:.0f}" if rmse else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Main forecast chart
# ---------------------------------------------------------------------------
st.subheader("📈 Demand Forecast — Scenario Explorer")

fig = go.Figure()

# Historical actuals
fig.add_trace(go.Scatter(
    x=history_trimmed["week_start"],
    y=history_trimmed["bookings"],
    mode="lines",
    name="Actuals",
    line=dict(color="#1f77b4", width=2),
))

# Shaded uncertainty band (pessimistic to optimistic)
if show_pessimistic and show_optimistic:
    fig.add_trace(go.Scatter(
        x=pd.concat([scenarios["week_start"], scenarios["week_start"][::-1]]),
        y=pd.concat([scenarios["optimistic"], scenarios["pessimistic"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Scenario band",
        showlegend=True,
    ))

scenario_styles = {
    "pessimistic": dict(color="#d62728", dash="dot", label="Pessimistic (×0.80)"),
    "baseline":    dict(color="#ff7f0e", dash="solid", label="Baseline (×1.00)"),
    "optimistic":  dict(color="#2ca02c", dash="dash", label="Optimistic (×1.20)"),
}
show_flags = {
    "pessimistic": show_pessimistic,
    "baseline": show_baseline,
    "optimistic": show_optimistic,
}

for scenario, style in scenario_styles.items():
    if show_flags[scenario]:
        fig.add_trace(go.Scatter(
            x=scenarios["week_start"],
            y=scenarios[scenario],
            mode="lines",
            name=style["label"],
            line=dict(color=style["color"], dash=style["dash"], width=2),
        ))

# Vertical divider at last actual date
last_actual = history["week_start"].max()
fig.add_shape(
    type="line",
    x0=str(last_actual),
    x1=str(last_actual),
    y0=0,
    y1=1,
    yref="paper",
    line=dict(dash="dash", color="grey", width=1),
)
fig.add_annotation(
    x=str(last_actual),
    y=1.02,
    yref="paper",
    text="Forecast start",
    showarrow=False,
    xanchor="left",
    font=dict(color="grey", size=11),
)

fig.update_layout(
    xaxis_title="Week",
    yaxis_title="Weekly Bookings",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480,
    margin=dict(t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance + metrics detail side by side
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🔍 Feature Importance")
    top_features = metrics.get("top_features", {})
    if top_features:
        feat_df = (
            pd.Series(top_features, name="importance")
            .sort_values(ascending=True)
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        fig_feat = go.Figure(go.Bar(
            x=feat_df["importance"],
            y=feat_df["feature"],
            orientation="h",
            marker_color="#1f77b4",
        ))
        fig_feat.update_layout(
            xaxis_title="Importance (split gain)",
            yaxis_title="",
            height=320,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Run `python src/models/train_lightgbm.py` to generate feature importance.")

with col_right:
    st.subheader("📋 Model Training Details")
    if metrics:
        detail_rows = {
            "Train period": f"{metrics.get('train_start')} → {metrics.get('train_end')}",
            "Test period": f"{metrics.get('test_start')} → {metrics.get('test_end')}",
            "Train weeks": metrics.get("train_weeks"),
            "Test weeks": metrics.get("test_weeks"),
            "Features": metrics.get("n_features"),
            "Best iteration": metrics.get("best_iteration"),
            "MAPE": f"{metrics.get('mape'):.2f}%",
            "MAE": f"{metrics.get('mae'):.1f} bookings",
            "RMSE": f"{metrics.get('rmse'):.1f}",
        }
        st.table(pd.DataFrame.from_dict(detail_rows, orient="index", columns=["Value"]))
    else:
        st.info("No metrics file found. Train the model first.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Scenario summary table
# ---------------------------------------------------------------------------
st.subheader("📊 26-Week Scenario Forecast Table")

display_df = scenarios.copy()
display_df["week_start"] = display_df["week_start"].dt.strftime("%Y-%m-%d")
display_df = display_df.rename(columns={
    "week_start": "Week",
    "pessimistic": "Pessimistic",
    "baseline": "Baseline",
    "optimistic": "Optimistic",
})

st.dataframe(
    display_df.style
        .format({"Pessimistic": "{:.1f}", "Baseline": "{:.1f}", "Optimistic": "{:.1f}"})
        .background_gradient(subset=["Baseline"], cmap="Blues"),
    use_container_width=True,
    height=400,
)

csv = display_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download scenarios CSV",
    data=csv,
    file_name="travel_demand_scenarios.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Model: LightGBM with lag (1,2,4,8,12w), rolling statistics, and calendar features. "
    "Scenarios apply demand multipliers of 0.80x / 1.00x / 1.20x to the baseline forecast. "
    "Data: Kaggle Hotel Booking Demand dataset (Mostipak, 2020)."
)
