"""Shared Plotly chart helpers for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cumulative_returns(returns: pd.Series, name: str = "L/S net") -> go.Figure:
    """Cumulative product chart of a weekly returns series."""
    cum = (1 + returns.fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=name))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative growth (×)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def cumulative_overlay(series_dict: dict[str, pd.Series]) -> go.Figure:
    """Overlay several cumulative return curves."""
    fig = go.Figure()
    for name, ret in series_dict.items():
        cum = (1 + ret.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=name))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative growth (×)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def metric_cards(metrics: dict, cols) -> None:
    """Render top-line metrics as columns of st.metric cards."""
    pairs = [
        ("annualized_return", "Ann. Return", "{:.2%}"),
        ("information_ratio", "IR", "{:.2f}"),
        ("max_drawdown", "Max DD", "{:.2%}"),
        ("avg_weekly_turnover", "Avg Turnover", "{:.0%}"),
    ]
    for col, (key, label, fmt) in zip(cols, pairs):
        v = metrics.get(key)
        col.metric(label, fmt.format(v) if v is not None else "—")
