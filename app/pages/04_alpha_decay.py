"""Alpha decay page: IR vs execution lag."""

import plotly.graph_objects as go
import streamlit as st

from app.lib import data_loader as dl

st.title("Alpha decay")
st.caption(
    "Information ratio as a function of execution lag (days). Paper p.7 reports "
    "monotone decline from 14.9% (lag 0) to 6.8% (lag 4) annualized."
)

df = dl.get("alpha_decay")
if df is None:
    dl.render_no_data_warning("alpha_decay.parquet")
    st.stop()

if "information_ratio" in df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["information_ratio"], mode="lines+markers", name="IR")
    )
    fig.update_layout(
        xaxis_title="Execution lag (days)",
        yaxis_title="Information ratio",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-lag metrics")
st.dataframe(df, use_container_width=True)
