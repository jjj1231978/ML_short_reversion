"""Overview page: headline metrics + cumulative L/S equity curve."""

import streamlit as st

from app.lib import data_loader as dl
from app.lib.plots import (
    cumulative_overlay,
    matched_period_stats,
    metric_cards,
    pct_styler,
)

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

st.plotly_chart(
    cumulative_overlay(overlay, highlight={"ML L/S (net)"}),
    use_container_width=True,
)

# The model starts ~336 weeks after the baseline (its 260+52-week rolling warmup),
# so an unclipped chart hands the baseline a large head start over a window the
# model never traded. Say so on the page rather than leaving the reader to infer it.
if len(overlay) > 1:
    starts = {k: v.dropna().index.min() for k, v in overlay.items() if v is not None}
    if starts:
        common = max(starts.values())
        dropped = {k: (common - s).days // 7 for k, s in starts.items() if s < common}
        if dropped:
            st.caption(
                f"Curves are clipped to a common start of **{common:%Y-%m-%d}** and rebased "
                f"to 1.0. Series that begin earlier have their pre-{common:%Y-%m}, "
                f"un-comparable history dropped ("
                + ", ".join(f"{k}: −{w} weeks" for k, w in dropped.items())
                + "). Without this the earlier series compounds over a window the model "
                "never traded and looks far stronger than a like-for-like comparison shows."
            )

# Same mismatch in numeric form: metrics_summary.json scores the model over its
# 740 weeks while the baseline is scored over its full history, which flatters the
# baseline. Recompute both on shared weeks so table and chart agree.
matched = matched_period_stats(overlay)
if not matched.empty and len(matched) > 1:
    st.subheader("Like-for-like comparison (shared weeks only)")
    st.dataframe(pct_styler(matched.reset_index()), use_container_width=True, hide_index=True)
    st.caption(
        "Recomputed on the weeks every series has in common, so these can differ from "
        "the cards above — those come from metrics_summary.json, which scores each "
        "series over its own full history."
    )

if portfolio is not None:
    with st.expander("Weekly returns table"):
        st.dataframe(pct_styler(portfolio.tail(52).reset_index()), use_container_width=True)
