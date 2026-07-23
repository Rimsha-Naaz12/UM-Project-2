"""
UAC Program - Predictive Forecasting Dashboard
Run locally:  streamlit run app.py
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="UAC Care Load & Placement Forecast", layout="wide")

DATA_DIR = "data"
OUT_DIR = "outputs"

TARGET_LABELS = {
    "hhs_care": "Children in HHS Care",
    "hhs_discharged": "Children Discharged from HHS Care (daily)",
}

# ---------- Data loading (cached) ----------
@st.cache_data
def load_clean():
    return pd.read_csv(os.path.join(DATA_DIR, "clean_uac_data.csv"), parse_dates=["date"], index_col="date")


@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(OUT_DIR, "model_metrics.csv"))


@st.cache_data
def load_backtest(target):
    return pd.read_csv(os.path.join(OUT_DIR, f"forecast_{target}.csv"), parse_dates=["date"])


@st.cache_data
def load_future(target):
    return pd.read_csv(os.path.join(OUT_DIR, f"future_forecast_{target}.csv"), parse_dates=["date"])


clean_df = load_clean()
metrics_df = load_metrics()

# ---------- Sidebar controls ----------
st.sidebar.title("Forecast Controls")
target = st.sidebar.selectbox(
    "Metric to forecast", options=list(TARGET_LABELS.keys()), format_func=lambda x: TARGET_LABELS[x]
)
horizon = st.sidebar.slider("Forecast horizon (days ahead)", min_value=1, max_value=21, value=14)
model_choice = st.sidebar.multiselect(
    "Models to display",
    options=["Random Forest", "Gradient Boosting", "ETS"],
    default=["Random Forest", "Gradient Boosting", "ETS"],
)
show_ci = st.sidebar.checkbox("Show 90% confidence interval (ETS)", value=True)
capacity_threshold = st.sidebar.number_input(
    "Capacity alert threshold", min_value=0, value=int(clean_df["hhs_care"].max() * 1.02), step=50
)

st.title("UAC Program — Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Unaccompanied Alien Children Program · forecasting dashboard")

# ---------- KPI row ----------
future_df = load_future(target)
future_view = future_df[future_df["date"] <= future_df["date"].min() + pd.Timedelta(days=horizon - 1)]

last_actual = clean_df[target].iloc[-1]
avg_forecast = future_view[[c for c in future_view.columns if c.startswith("forecast_")]].mean(axis=1).mean()
breach_prob = 0.0
if target == "hhs_care":
    ets_upper = future_view["ci_upper_90"]
    breach_prob = float((ets_upper > capacity_threshold).mean() * 100)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last reported value", f"{last_actual:,.0f}", help=f"As of {clean_df.index.max().date()}")
col2.metric(f"Avg forecast (next {horizon}d)", f"{avg_forecast:,.0f}",
            delta=f"{avg_forecast - last_actual:+,.0f}")
col3.metric("Capacity Breach Probability", f"{breach_prob:.0f}%" if target == "hhs_care" else "N/A")
available_horizons = sorted(metrics_df["horizon"].unique())
nearest_horizon = min(available_horizons, key=lambda h: abs(h - horizon))
best_model_row = metrics_df[(metrics_df.target == target) & (metrics_df.horizon == nearest_horizon)].sort_values("MAE").iloc[0]
col4.metric("Best model (by MAE)", best_model_row["model"],
            help=f"MAE={best_model_row['MAE']:.1f} (evaluated at {nearest_horizon}-day horizon, nearest to your {horizon}-day selection)")

# ---------- Forecast chart ----------
st.subheader(f"Future Care Load Forecast — {TARGET_LABELS[target]}")
fig = go.Figure()
hist = clean_df[target].iloc[-90:]
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical (last 90d)",
                          line=dict(color="#1f77b4")))

color_map = {"Random Forest": "#2ca02c", "Gradient Boosting": "#ff7f0e", "ETS": "#9467bd"}
for m in model_choice:
    col = f"forecast_{m.replace(' ', '_')}"
    if col in future_view.columns:
        fig.add_trace(go.Scatter(x=future_view["date"], y=future_view[col], name=f"{m} forecast",
                                  line=dict(color=color_map.get(m), dash="dash")))

if show_ci and "ci_lower_90" in future_view.columns:
    fig.add_trace(go.Scatter(x=future_view["date"], y=future_view["ci_upper_90"],
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_view["date"], y=future_view["ci_lower_90"],
                              fill="tonexty", fillcolor="rgba(148,103,189,0.15)",
                              line=dict(width=0), name="90% Confidence Interval"))

if target == "hhs_care":
    fig.add_hline(y=capacity_threshold, line_dash="dot", line_color="red",
                  annotation_text="Capacity threshold")

fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# ---------- Discharge demand panel ----------
st.subheader("Discharge Demand Forecast Panel")
disch_future = load_future("hhs_discharged")
disch_view = disch_future[disch_future["date"] <= disch_future["date"].min() + pd.Timedelta(days=horizon - 1)]
c1, c2 = st.columns([2, 1])
with c1:
    fig2 = go.Figure()
    hist_d = clean_df["hhs_discharged"].iloc[-60:]
    fig2.add_trace(go.Bar(x=hist_d.index, y=hist_d.values, name="Historical discharges", marker_color="#1f77b4"))
    fig2.add_trace(go.Bar(x=disch_view["date"], y=disch_view["forecast_Random_Forest"],
                           name="Forecast (Random Forest)", marker_color="#2ca02c"))
    fig2.update_layout(height=380, barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    net_pressure_fc = future_view["forecast_Random_Forest"].diff().mean() if target == "hhs_care" else None
    st.metric("Avg projected daily discharges", f"{disch_view['forecast_Random_Forest'].mean():.1f}")
    st.metric("Avg projected daily net pressure",
              f"{(clean_df['cbp_transferred_out'].iloc[-14:].mean() - disch_view['forecast_Random_Forest'].mean()):+.1f}",
              help="Transfers in minus projected discharges — positive means care load is expected to rise.")

# ---------- Model comparison ----------
st.subheader("Model Selection & Comparison")
mtab = metrics_df[metrics_df.target == target].pivot(index="model", columns="horizon", values="MAE")
mtab.columns = [f"{h}-day MAE" for h in mtab.columns]
st.dataframe(mtab.style.highlight_min(axis=0, color="#c6f6d5"), use_container_width=True)

with st.expander("Full evaluation metrics (MAE / RMSE / MAPE, all horizons)"):
    st.dataframe(metrics_df[metrics_df.target == target].sort_values(["horizon", "MAE"]), use_container_width=True)

st.caption(
    "Data source: HHS Unaccompanied Alien Children Program daily report. "
    "Forecasts are statistical estimates for planning support, not guarantees."
)
