"""Live forecast track record: realized returns of the saved baskets.

Reads `data/processed/forecast_track_record_{day}.parquet` produced by
`python -m src.backtest.forecast_track_record --signal-day {day}`. Each row in
that parquet represents one saved forecast; this page renders the per-forecast
table, the cumulative L/S curve, and the mean hit rate across resolved
forecasts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PROCESSED_DIR

SIGNAL_DAY_LABELS = {"WED": "Wednesday", "THU": "Thursday", "FRI": "Friday"}
REGION_ORDER = ["ALL", "US", "UK", "CA"]
REGION_COLOR = {
    "ALL": "#1f77b4",
    "US":  "#2ca02c",
    "UK":  "#d62728",
    "CA":  "#9467bd",
}

st.title("Live forecast — track record")
st.caption(
    "Realized per-region long/short basket returns for forecasts whose target "
    "Wednesday has already passed. To refresh after a new forecast resolves, "
    "run: `python -m src.backtest.forecast_track_record --signal-day {WED|THU|FRI}`"
)

with st.sidebar:
    st.header("Rebalance day")
    sd_label = st.selectbox(
        "Signal day",
        list(SIGNAL_DAY_LABELS.values()),
        index=0,
    )
    sd_code = {v: k for k, v in SIGNAL_DAY_LABELS.items()}[sd_label]

track_path = PROCESSED_DIR / f"forecast_track_record_{sd_code}.parquet"

if not track_path.exists():
    st.warning(
        f"No track record file for {sd_label}.\n\n"
        f"Generate one with:\n\n"
        f"```\npython -m src.backtest.forecast_track_record --signal-day {sd_code}\n```"
    )
    st.stop()

df = pd.read_parquet(track_path)
if df.empty:
    st.warning(f"Track record file is empty for {sd_label}.")
    st.stop()

eval_df = df[df.get("evaluable", False)].copy() if "evaluable" in df.columns else df.copy()
n_total = len(df)
n_eval = len(eval_df)

c1, c2, c3 = st.columns(3)
c1.metric("Forecasts saved", n_total)
c2.metric("Resolved", n_eval)
c3.metric("In-flight", n_total - n_eval)

if n_eval == 0:
    st.info(
        "No forecasts have resolved yet — either their target Wednesday is in "
        "the future, or the price cache doesn't extend that far. Re-run the "
        "track-record script after the target dates pass."
    )
    pending = df[["as_of_date", "target_date", "signal_day"]].copy()
    if "reason" in df.columns:
        pending["reason"] = df["reason"]
    pending = pending.sort_values("target_date")
    pending["as_of_date"] = pd.to_datetime(pending["as_of_date"]).dt.date
    pending["target_date"] = pd.to_datetime(pending["target_date"]).dt.date
    st.subheader("Pending forecasts")
    st.dataframe(pending, use_container_width=True)
    st.stop()

# Per-forecast realized returns
st.subheader("Per-forecast realized L/S returns")
display = eval_df[["as_of_date", "target_date"]].copy()
display["as_of_date"] = pd.to_datetime(display["as_of_date"]).dt.date
display["target_date"] = pd.to_datetime(display["target_date"]).dt.date
for region in REGION_ORDER:
    col = f"{region}_ls_ret"
    if col in eval_df.columns:
        display[region] = eval_df[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
st.dataframe(display, use_container_width=True)

# Cumulative L/S curves
st.subheader("Cumulative long-short return")
fig = go.Figure()
for region in REGION_ORDER:
    col = f"{region}_ls_ret"
    if col not in eval_df.columns:
        continue
    series = eval_df.set_index("target_date")[col].dropna()
    if len(series) == 0:
        continue
    cum = (1 + series).cumprod()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        mode="lines+markers", name=region,
        line=dict(color=REGION_COLOR.get(region)),
    ))
fig.update_layout(
    xaxis_title="Target Wednesday",
    yaxis_title="Cumulative growth (×)",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# Summary table: mean L/S, hit rate, cumulative
st.subheader("Summary across resolved forecasts")
summary_rows = []
for region in REGION_ORDER:
    ls_col = f"{region}_ls_ret"
    hr_col = f"{region}_hit_rate"
    if ls_col not in eval_df.columns:
        continue
    ls = eval_df[ls_col].dropna()
    if len(ls) == 0:
        continue
    summary_rows.append({
        "Region": region,
        "Forecasts": len(ls),
        "Mean weekly L/S": f"{ls.mean():+.2%}",
        "Cumulative L/S": f"{(1 + ls).prod() - 1:+.2%}",
        "Best week": f"{ls.max():+.2%}",
        "Worst week": f"{ls.min():+.2%}",
        "Mean hit rate": (
            f"{eval_df[hr_col].dropna().mean():.1%}"
            if hr_col in eval_df.columns and eval_df[hr_col].notna().any()
            else "—"
        ),
    })
if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.caption(
    f"Source: `{track_path.relative_to(PROCESSED_DIR.parent)}` — regenerate with "
    f"`python -m src.backtest.forecast_track_record --signal-day {sd_code}`."
)
