"""Weekday effect page: same model + features rebuilt against each signal day."""

import plotly.express as px
import streamlit as st

from app.lib import data_loader as dl

st.title("Weekday effect")
st.caption(
    "Per-weekday re-runs of the full pipeline. Paper p.8: Thursday best, "
    "Monday worst (correlates with macro/earnings release timing)."
)

df = dl.get("weekday_effect")
if df is None:
    dl.render_no_data_warning("weekday_effect.parquet")
    st.stop()

if "annualized_return" in df.columns:
    fig = px.bar(
        df["annualized_return"],
        title="Annualized return by signal day",
        labels={"value": "Ann. Return", "index": "Signal day"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-weekday metrics")
st.dataframe(df, use_container_width=True)
