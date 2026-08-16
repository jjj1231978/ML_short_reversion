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


def cumulative_overlay(
    series_dict: dict[str, pd.Series], highlight: set[str] | None = None
) -> go.Figure:
    """Overlay several cumulative return curves from a COMMON start date.

    Series here begin at different dates: the ML model only produces predictions
    after its 260+52-week rolling warmup (first OOS week 2012-06-21), while the
    R1W baseline needs no training and runs from 2006-01-12. Plotting raw
    cumulative products gives the earlier-starting series a 336-week head start
    and makes the comparison meaningless — the baseline showed ~690% cumulative
    against the model's ~234% purely because it compounded over 20.6 years rather
    than 14.2, and its extra window covers the GFC, when reversal did unusually
    well. On matched weeks the baseline actually returns ~144%, i.e. it LOSES.

    So: clip every series to the latest common first-valid date and rebase each to
    1.0 there. Names in `highlight` are drawn thicker.
    """
    highlight = highlight or set()
    valid = {k: v.dropna() for k, v in series_dict.items() if v is not None and v.dropna().size}
    fig = go.Figure()
    if not valid:
        fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative growth (×)")
        return fig

    common_start = max(v.index.min() for v in valid.values())
    for name, ret in valid.items():
        r = ret[ret.index >= common_start]
        cum = (1 + r.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(width=4 if name in highlight else 1.6),
        ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative growth (×, rebased to 1.0 at common start)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def matched_period_stats(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-series IR / annualized return / cumulative return over the weeks ALL
    series share.

    The same period mismatch that distorts the equity chart also distorts the
    headline numbers: metrics_summary.json reports the model over 740 weeks while
    the R1W baseline is scored over its full 1076, which flattered the baseline's
    IR (0.896 vs 0.596 like-for-like). This recomputes every series on the shared
    index so the table cannot disagree with the chart above it.
    """
    valid = {k: v.dropna() for k, v in series_dict.items() if v is not None and v.dropna().size}
    if not valid:
        return pd.DataFrame()
    idx = None
    for v in valid.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    rows = []
    for name, s in valid.items():
        r = s.reindex(idx).dropna()
        if r.empty:
            continue
        ann = r.mean() * 52
        vol = r.std() * (52 ** 0.5)
        rows.append({
            "series": name,
            "weeks": len(r),
            "information_ratio": (ann / vol) if vol else float("nan"),
            "annualized_return": ann,
            "annualized_volatility": vol,
            "cumulative_return": (1 + r).prod() - 1,
            "max_drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
        })
    return pd.DataFrame(rows).set_index("series")


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


# Column-name fragments that denote a rate to render as a 2-decimal percentage
# (returns, volatility, drawdown, turnover). Information ratio stays a 2dp number.
_PCT_FRAGMENTS = ("return", "volatility", "drawdown", "turnover")


def pct_styler(df: pd.DataFrame):
    """Style a returns/metrics table: rate-like columns as 2-decimal percentages
    (e.g. 0.0481 → "4.81%"), information ratio as a 2dp number, counts as ints.

    Returns a pandas Styler; pass straight to st.dataframe(...)."""
    fmt: dict = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "num_weeks":
            fmt[c] = "{:.0f}"
        elif "information_ratio" in lc or lc.endswith("_ir") or lc in ("ir", "sharpe"):
            fmt[c] = "{:.2f}"
        elif any(frag in lc for frag in _PCT_FRAGMENTS):
            fmt[c] = "{:.2%}"
    return df.style.format(fmt)
