"""Overview page: headline metrics + cumulative L/S equity curve."""

import streamlit as st

from app.lib import data_loader as dl
from app.lib.plots import cumulative_overlay, metric_cards

st.title("Overview")

metrics = dl.get("metrics_summary", "json")
portfolio = dl.get("portfolio_returns")
baseline = dl.get("baseline_r1w_returns")

if portfolio is None and metrics is None:
    dl.render_no_data_warning("portfolio_returns.parquet / metrics_summary.json")
    st.stop()

cols = st.columns(4)
metric_cards(metrics or {}, cols)

st.subheader("Cumulative long/short return (net of costs)")
overlay = {}
if portfolio is not None and "ls_return_net" in portfolio.columns:
    overlay["ML L/S (net)"] = portfolio["ls_return_net"]
    if "ls_return" in portfolio.columns:
        overlay["ML L/S (gross)"] = portfolio["ls_return"]
if baseline is not None and "ls_return_net" in baseline.columns:
    overlay["R1W reversal baseline (net)"] = baseline["ls_return_net"]
st.plotly_chart(cumulative_overlay(overlay), use_container_width=True)

if portfolio is not None:
    with st.expander("Weekly returns table"):
        st.dataframe(portfolio.tail(52), use_container_width=True)
