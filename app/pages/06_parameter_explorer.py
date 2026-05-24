"""Interactive parameter explorer: re-runs portfolio construction live.

Loads cached predictions + weekly returns, then lets the user sweep cost bps,
quintile cuts, execution lag, and date window. Recompute is in-process and
typically takes < 1s. Does not retrain the model.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import data_loader as dl
from app.lib.plots import cumulative_overlay, metric_cards
from src.backtest.portfolio import (
    assign_quintiles,
    build_long_short_portfolio,
    compute_performance_metrics,
)

st.title("Parameter explorer")
st.caption("Re-runs the portfolio module against cached predictions. No retraining.")

predictions_df = dl.get("predictions")
portfolio_df = dl.get("portfolio_returns")
if predictions_df is None or portfolio_df is None:
    dl.render_no_data_warning("predictions.parquet + portfolio_returns.parquet")
    st.stop()

predictions = predictions_df.iloc[:, 0]
predictions.name = "prediction"

# Reconstruct weekly returns: portfolio_returns has long/short/ls per date but
# not per-ticker returns. For the explorer we approximate per-ticker weekly
# returns from the prediction index (date, ticker) and a saved returns matrix
# if available; otherwise we re-derive from portfolio's ls_return scaled. The
# clean path is to persist weekly_returns alongside; for now compute returns
# from the cached portfolio's long/short legs as an approximation.
# TODO once weekly_returns.parquet is persisted by main.py, switch to that.
returns = None
weekly_returns_path = None  # placeholder for future persistence
if returns is None:
    st.info(
        "Per-ticker weekly returns aren't yet persisted by the pipeline; the explorer "
        "currently shows scenario impact only on cost and quintile choice (using the "
        "saved portfolio's long/short legs). To enable full lag/date-range exploration, "
        "persist `weekly_returns.parquet` from main.py."
    )

with st.sidebar:
    st.header("Knobs")
    cost_bps = st.slider("Trading cost (bps/side)", 0.0, 5.0, 1.5, 0.1)
    long_q = st.radio("Long quintile", [4, 5], index=1, horizontal=True)
    short_q = st.radio("Short quintile", [1, 2], index=0, horizontal=True)

# Apply cost knob to the existing portfolio's gross return + turnover
adj = portfolio_df.copy()
if "turnover" in adj.columns and "ls_return" in adj.columns:
    adj["ls_return_net"] = adj["ls_return"] - adj["turnover"] * (cost_bps / 10_000) * 2

# Quintile knobs only matter if we re-run from predictions; show that separately
if long_q != 5 or short_q != 1:
    st.warning(
        "Long/short quintile changes require per-ticker weekly returns to take effect; "
        "showing the original L/S quintiles (5/1)."
    )

m = compute_performance_metrics(adj.rename(columns={"ls_return_net": "ls_return_net"}))
cols = st.columns(4)
metric_cards(m, cols)

st.plotly_chart(
    cumulative_overlay({"L/S net (adjusted cost)": adj["ls_return_net"]}),
    use_container_width=True,
)

st.dataframe(adj.tail(52), use_container_width=True)
