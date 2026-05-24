"""Predictions page: per-week distribution + top/bottom names."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.lib import data_loader as dl

st.title("Predictions")

df = dl.get("predictions")
if df is None:
    dl.render_no_data_warning("predictions.parquet")
    st.stop()

s = df.iloc[:, 0]

dates = sorted(s.index.get_level_values("date").unique(), reverse=True)
sel = st.selectbox("Week", dates, format_func=lambda d: pd.Timestamp(d).date().isoformat())
week = s.xs(sel, level="date").sort_values(ascending=False)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribution")
    fig = px.histogram(week, nbins=40, title=f"Predicted score, {pd.Timestamp(sel).date()}")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Top 10 long picks (Q5)")
    st.dataframe(week.head(10).to_frame("score"), use_container_width=True)
    st.subheader("Bottom 10 short picks (Q1)")
    st.dataframe(week.tail(10).iloc[::-1].to_frame("score"), use_container_width=True)
