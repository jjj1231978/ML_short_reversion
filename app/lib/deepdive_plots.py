"""Charts for the Deep dive page.

Palette note. The region colours used elsewhere in this app
(`09_track_record.py`: US `#2ca02c` green, UK `#d62728` red) are only ΔE 3.9
apart under deuteranopia — the classic red/green pair, effectively one colour
for ~5% of readers. The set below was picked from ColorBrewer Dark2 and passes
the full validator sweep (lightness band, chroma floor, CVD separation ≥ 8,
normal-vision floor, 3:1 contrast against the surface):

    node scripts/validate_palette.js "#2166ac,#d95f02,#7570b3" --mode light

Region is never encoded by colour alone here — the legend, the marker symbol
(side) and the screening table all carry the same information.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical: identity. Fixed order, never cycled, never reassigned by rank.
REGION_COLOR = {"US": "#2166ac", "UK": "#d95f02", "CA": "#7570b3"}
REGION_ORDER = ["US", "UK", "CA"]

# Diverging: polarity. Warm/cool poles either side of a neutral grey midpoint.
POS_COLOR, NEG_COLOR, NEUTRAL = "#2166ac", "#d95f02", "#9e9e9e"

# Ordinal ramp for the moving averages — one hue, light to dark, because the
# periods are an ordered category (20 < 50 < 200).
SMA_COLORS = {"sma20": "#9ecae1", "sma50": "#4292c6", "sma200": "#08519c"}
PRICE_COLOR = "#111111"

GRID = "#e8e8e8"
BAND = "rgba(158,158,158,0.10)"
MARGIN = dict(l=40, r=20, t=30, b=40)

# Side is the secondary encoding, so the CVD floor is never the only separator.
SIDE_SYMBOL = {"LONG": "circle", "SHORT": "diamond"}

def _risk_text(alignment: str, side: str) -> str:
    """Tooltip wording, matching the page's "1-week risk" column.

    Side-aware: a stretched long faces a pullback, a stretched short a bounce.
    """
    if alignment == "conflict":
        return "pullback risk" if str(side).upper() == "LONG" else "bounce risk"
    return "reversion in our favour" if alignment == "aligned" else "neutral"


def _base(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(
        margin=MARGIN,
        plot_bgcolor="white",
        # title="" rather than leaving it unset: make_subplots leaves the title
        # object present-but-empty, which renders as a literal "undefined".
        title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(align="left"),
        **layout,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False)
    return fig


def model_vs_chart(df: pd.DataFrame) -> go.Figure:
    """Model conviction against where the price chart actually is.

    The whole point of the page in one chart. This is a reversion model, so a
    long (right-hand side) sitting in the overbought band, or a short (left) in
    the oversold band, is the model buying strength or selling weakness — the
    opposite of its design. Those are the names to look at first. The screening
    table below is this chart's table view.
    """
    fig = go.Figure()

    # Reference bands rather than dashed rules: dashes read as "projection" and
    # these are standing zones, not gridlines.
    for lo, hi in ((70, 100), (0, 30)):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND, line_width=0, layer="below")

    for region in REGION_ORDER:
        for side, symbol in SIDE_SYMBOL.items():
            grp = df[(df["region"] == region) & (df["side"] == side)]
            if grp.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=grp["prediction"],
                    y=grp["rsi14"],
                    mode="markers",
                    name=f"{region} {side.lower()}",
                    legendgroup=region,
                    marker=dict(
                        size=11,           # >= 8px, and the hit area clears 24px
                        symbol=symbol,
                        color=REGION_COLOR[region],
                        opacity=0.85,
                        line=dict(width=2, color="white"),   # surface ring, not a border
                    ),
                    customdata=grp.assign(
                        risk_text=[_risk_text(a, side) for a in grp["alignment"]]
                    )[["ticker", "verdict_label", "risk_text", "move_since_signal_pct"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{region} · {side.lower()}<br>"
                        "model score %{x:.3f}<br>"
                        "RSI(14) %{y:.1f}<br>"
                        "%{customdata[1]}<br>"
                        "1-week risk: %{customdata[2]}<br>"
                        "since signal %{customdata[3]:+.1f}%<extra></extra>"
                    ),
                )
            )

    # Label only the conflicts — the extremes that carry the story.
    conflicts = df[df["alignment"] == "conflict"]
    if not conflicts.empty:
        fig.add_trace(
            go.Scatter(
                x=conflicts["prediction"],
                # Lift the label clear of the 11px marker and its 2px ring;
                # ~3 RSI units is the gap at this plot height.
                y=conflicts["rsi14"] + 3.0,
                mode="text",
                text=conflicts["ticker"],
                textposition="top center",
                textfont=dict(size=10, color="#555555"),
                cliponaxis=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return _base(
        fig,
        xaxis_title="Model score (0 = strongest short, 1 = strongest long)",
        yaxis_title="RSI(14)",
        hovermode="closest",
        yaxis=dict(range=[0, 100], dtick=25),
        xaxis=dict(range=[-0.02, 1.02]),
        height=460,
    )


def moves_since_signal(df: pd.DataFrame, top: int = 15) -> tuple[go.Figure, int]:
    """Largest absolute moves between the signal close and the latest close.

    A pick that has already run 8% is not the pick the backtest measured, so
    these are the names whose entry price assumption is most stale. Returns the
    figure and the number of names not shown, so the caption can say what was
    cut rather than implying the chart is the whole book.
    """
    d = df.dropna(subset=["move_since_signal_pct"]).copy()
    hidden = max(0, len(d) - top)
    d = d.reindex(d["move_since_signal_pct"].abs().sort_values().index).tail(top)

    labels = [f"{t}  ({s.lower()})" for t, s in zip(d["ticker"], d["side"])]
    fig = go.Figure(
        go.Bar(
            x=d["move_since_signal_pct"],
            y=labels,
            orientation="h",
            marker=dict(
                color=[POS_COLOR if v >= 0 else NEG_COLOR for v in d["move_since_signal_pct"]],
                line=dict(width=0),
            ),
            customdata=d[["region", "verdict_label"]],
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br>"
                           "moved %{x:+.1f}% since the signal close<br>"
                           "%{customdata[1]}<extra></extra>"),
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=NEUTRAL)
    return _base(
        fig,
        xaxis_title="Move since the signal close (%)",
        yaxis_title=None,
        height=max(260, 26 * len(d) + 90),
        bargap=0.35,
    ), hidden


def price_and_rsi(bars: pd.DataFrame, tech: dict, as_of: str | None, ticker: str) -> go.Figure:
    """Twelve months of price with its moving averages, over an RSI panel.

    Two stacked panels sharing one x-axis, each with its own single y-scale —
    price and RSI are never plotted against two scales on one plot.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(x=bars.index, y=bars["close"], mode="lines", name="Close",
                   line=dict(color=PRICE_COLOR, width=2),
                   hovertemplate="%{x|%Y-%m-%d}<br>close %{y:,.2f}<extra></extra>"),
        row=1, col=1,
    )
    for col, label in (("sma20", "SMA 20"), ("sma50", "SMA 50"), ("sma200", "SMA 200")):
        window = int(col.removeprefix("sma"))
        if len(bars) < window:
            continue
        fig.add_trace(
            go.Scatter(x=bars.index, y=bars["close"].rolling(window).mean(), mode="lines",
                       name=label, line=dict(color=SMA_COLORS[col], width=2),
                       hovertemplate=f"{label} %{{y:,.2f}}<extra></extra>"),
            row=1, col=1,
        )

    # The entry point the model actually scored. Deliberately unannotated: the
    # signal date is always within days of the last bar, so any label sits on the
    # right edge and collides with the price line and the 52w rules. The caption
    # names it, and the 52w levels are in the metric table directly below.
    if as_of:
        fig.add_vline(x=pd.Timestamp(as_of), line_width=1, line_color=NEUTRAL,
                      row=1, col=1)

    for key in ("range_52w_high", "range_52w_low"):
        if tech.get(key) is not None:
            fig.add_hline(y=tech[key], line_width=1, line_color=GRID, row=1, col=1)

    if "rsi" in bars:
        for lo, hi in ((70, 100), (0, 30)):
            fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND, line_width=0, layer="below",
                          row=2, col=1)
        fig.add_trace(
            # One series in its own panel, named by the axis title — a legend
            # entry here would only invite confusion with the price traces.
            go.Scatter(x=bars.index, y=bars["rsi"], mode="lines", name="RSI(14)",
                       line=dict(color="#555f6d", width=2), showlegend=False,
                       hovertemplate="RSI %{y:.1f}<extra></extra>"),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI(14)", range=[0, 100], dtick=50, row=2, col=1)
    return _base(fig, hovermode="x unified", height=520)
