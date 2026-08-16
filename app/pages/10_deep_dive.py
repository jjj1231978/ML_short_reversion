"""Deep dive: what is actually happening at the names the model picked.

The Live forecast page says which names the model wants and which factors moved
the score. This page answers the next question, from three angles:

  fundamentals  last earnings vs estimate, next earnings, TTM valuation
  theme & news  what is driving the stock, and the headline risks this month
  technicals    RSI / moving averages / 52w range -> a trend-or-reversal verdict

Renders a cached artifact written by `python -m src.research deepdive`. Read-only
by default, like every other page; a per-ticker refresh appears only when
DEEPDIVE_ALLOW_REFRESH=1 and the API keys are present.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import app.lib.deepdive_plots as dp
from app.lib.forecasts import SIGNAL_DAY_LABELS, list_forecasts
from src.config import PROCESSED_DIR
from src.research.deepdive import artifact_path, load_artifact
from src.research.technicals import series_to_frame

DASH = "—"


def risk_badge(alignment: str, side: str) -> str:
    """What the technical setup implies for the coming week, given the side.

    This is forward-looking, not a statement about which way the stock has moved.
    A long on a name that has just run is exposed to a pullback; a short on a name
    that has just been sold off is exposed to a bounce — so the wording follows
    the side rather than saying "pullback" on both, which would be wrong half the
    time.
    """
    if alignment == "conflict":
        return "⚠️ pullback risk" if str(side).upper() == "LONG" else "⚠️ bounce risk"
    if alignment == "aligned":
        return "✅ reversion in our favour"
    return "· neutral"


st.title("Deep dive")
st.caption(
    "Company-level research on the model's live picks, from three angles: "
    "**fundamentals** (last reported earnings vs estimate, next earnings, TTM "
    "valuation), **theme** (what the market is reacting to, from news in the last "
    "month) and **technicals** (a deterministic trend-or-reversal verdict). "
    "The model scores factor data and cannot see any of this — the column worth "
    "reading first is **1-week risk**: this is a mean-reversion model, so a long "
    "on a name that has just run — or a short on one just sold off — is entering "
    "exactly where the strategy's own edge points against it over the next week. "
    "Covers the **Thursday** rebalance only, the day this book is traded."
)


@st.cache_data(show_spinner=False)
def _load(path_str: str, mtime: float) -> dict | None:
    """Mtime-keyed so a regenerated artifact invalidates the cache.

    `mtime` deliberately has NO leading underscore: st.cache_data skips hashing
    underscore-prefixed arguments entirely, so an `_mtime` parameter is inert and
    the page would keep serving the artifact it first read — silently showing
    pre-refresh numbers after a rerun of the batch job.
    """
    return load_artifact(path_str)


def _md(text) -> str:
    """Escape text that came from outside before handing it to markdown.

    Streamlit renders `$...$` as LaTeX, so a headline containing "C$230 million
    and C$250 million" silently turns into italic math and the currency vanishes.
    Every string on this page that came from a news article, a vendor field or a
    written brief goes through here.
    """
    return str(text).replace("$", r"\$") if text is not None else ""


def _get(rec: dict, *path, default=None):
    """Nested lookup that tolerates missing or null intermediate blocks."""
    cur = rec
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


# --- sidebar: one control surface scoping everything below --------------------

# Thursday only. Wed/Fri have their own trained bundles and forecast files, but
# Thursday is the day this book is actually traded on, and researching the other
# two would triple the API cost for names nobody takes a position in. The batch
# job still accepts --signal-day if that ever changes.
SIGNAL_DAY = "THU"

with st.sidebar:
    sd_code = SIGNAL_DAY
    sd_label = SIGNAL_DAY_LABELS[sd_code]
    forecasts = list_forecasts(sd_code)
    dd_dir = PROCESSED_DIR / "deepdive" / sd_code
    available = sorted((p.stem for p in dd_dir.glob("*.json")), reverse=True) if dd_dir.exists() else []

    st.header(f"{sd_label} rebalance")
    if not available:
        st.warning(f"No deep-dive artifacts for {sd_label}.")
        target = None
    else:
        target = st.selectbox("Target date", available, index=0)

if target is None:
    st.warning(
        f"No deep-dive research found for {sd_label}.\n\nGenerate it with:\n\n"
        f"```\npython -m src.research deepdive --signal-day {sd_code}\n```"
    )
    st.stop()

path = artifact_path(sd_code, target)
artifact = _load(str(path), path.stat().st_mtime)
if not artifact or not artifact.get("tickers"):
    st.error(f"Artifact is empty or unreadable: `{path}`")
    st.stop()

tickers = artifact["tickers"]

# --- flatten to one row per name ----------------------------------------------

rows = []
for ticker, rec in tickers.items():
    tech, fun, meta = rec.get("technicals") or {}, rec.get("fundamentals") or {}, rec.get("meta") or {}
    rows.append(
        {
            "ticker": ticker,
            "company": meta.get("company_name"),
            "region": meta.get("region"),
            "side": meta.get("side"),
            "rank": meta.get("rank"),
            "prediction": meta.get("prediction"),
            "industry": meta.get("industry"),
            "alignment": tech.get("alignment", "neutral"),
            "verdict_label": tech.get("verdict_label", DASH),
            "rsi14": tech.get("rsi14"),
            # stored 0-1; scaled to 0-100 here because ProgressColumn's `format`
            # is applied to the raw value, so a fraction would print as 0% / 1%.
            "pct_of_52w_range": (
                None if tech.get("pct_of_52w_range") is None
                else tech["pct_of_52w_range"] * 100.0
            ),
            "move_since_signal_pct": tech.get("move_since_signal_pct"),
            "ret_1w_pct": tech.get("ret_1w_pct"),
            "eps_surprise_pct": _get(fun, "last_earnings", "eps_surprise_pct"),
            "next_earnings": _get(fun, "next_earnings", "date"),
            "earnings_in_week": bool(_get(fun, "next_earnings", "in_holding_week", default=False)),
            "ev_ebitda": _get(fun, "valuation", "ev_ebitda_ttm"),
            "pe": _get(fun, "valuation", "pe_ttm"),
            "n_risks": len(_get(rec, "narrative", "headline_risks", default=[])),
            "n_articles": len(_get(rec, "news", "articles", default=[])),
            "has_brief": rec.get("narrative") is not None,
        }
    )
df = pd.DataFrame(rows)

with st.sidebar:
    st.header("Scope")
    # The slider can only narrow what the artifact already covers — the ceiling
    # is however many names the batch job researched, not a fixed 20. A scoped
    # run (--top-n 2, or a single --tickers refresh) must not break the page.
    covered_n = int(df["rank"].max()) if df["rank"].notna().any() else 1
    if covered_n > 1:
        n_top = st.slider(
            "Names per side per region", min_value=1, max_value=covered_n,
            value=covered_n, step=1,
            help="Narrows what is shown. To research more names, rerun the batch "
                 "job with a higher --top-n (capped at 20 by deepdive.max_top_n).",
        )
    else:
        n_top = covered_n
        st.caption(f"This artifact covers {covered_n} name per side per region.")
    side_choice = st.radio("Side", ["Both", "Long", "Short"], horizontal=True)
    regions = st.multiselect(
        "Regions", dp.REGION_ORDER,
        default=[r for r in dp.REGION_ORDER if r in set(df["region"].dropna())],
    )

view = df[df["rank"].fillna(9999) <= n_top]
if side_choice != "Both":
    view = view[view["side"] == side_choice.upper()]
if regions:
    view = view[view["region"].isin(regions)]

# --- header -------------------------------------------------------------------

generated = artifact.get("generated_at")
age_hours = None
if generated:
    try:
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(generated)).total_seconds() / 3600
    except ValueError:
        pass

m1, m2, m3, m4 = st.columns(4)
m1.metric("As-of close", artifact.get("as_of_date", DASH))
m2.metric("Target close", artifact.get("target_date", DASH))
m3.metric("Names shown", f"{len(view)} / {len(df)}")
m4.metric("Research age", f"{age_hours:.0f}h" if age_hours is not None else DASH)

# The artifact is pinned to one forecast; a newer forecast means it is stale.
newest = forecasts[0].stem if forecasts else None
if newest and newest > str(artifact.get("target_date", "")):
    st.warning(
        f"A newer forecast exists for {sd_label} (target **{newest}**), but this "
        f"research covers **{artifact.get('target_date')}**. Refresh with:\n\n"
        f"```\npython -m src.research deepdive --signal-day {sd_code}\n```"
    )

n_briefs = sum(df["has_brief"])
if n_briefs < len(df):
    st.info(
        f"{len(df) - n_briefs} of {len(df)} names have no written brief yet — their "
        "Theme tab falls back to the search-engine summary and the computed technical "
        "verdict. The numbers and articles below are unaffected."
    )

if view.empty:
    st.warning("No names match the current filters.")
    st.stop()

# --- the story in one chart ---------------------------------------------------

st.subheader("Where the entry carries reversion risk")
st.caption(
    "Each marker is one pick: model conviction across, RSI(14) up, colour by "
    "region, shape by side. Shaded bands are the conventional oversold (<30) and "
    "overbought (>70) zones. This is a *reversion* model, so a long sitting in the "
    "overbought band — or a short in the oversold band — is the model buying "
    "strength or selling weakness, the opposite of what it is designed to do. "
    "Those names are labelled, and flagged as a **pullback** or **bounce risk** in "
    "the table below, which is this chart's table view."
)
st.plotly_chart(dp.model_vs_chart(view), use_container_width=True)

conflicts = view[view["alignment"] == "conflict"]
if not conflicts.empty:
    st.caption(
        f"**{len(conflicts)} of {len(view)}** shown picks are entering against the "
        f"one-week reversion: "
        + ", ".join(f"`{t}`" for t in conflicts["ticker"].head(12))
        + (" …" if len(conflicts) > 12 else "")
    )

st.subheader("Picks that have already moved")
fig_moves, hidden = dp.moves_since_signal(view)
st.caption(
    "Move between the signal close the model scored and the most recent close. A "
    "name that has already run is no longer the entry the backtest measured."
    + (f" Showing the {len(view) - hidden} largest absolute moves; "
       f"{hidden} smaller movers are not plotted (all are in the table below)."
       if hidden else "")
)
st.plotly_chart(fig_moves, use_container_width=True)

# --- screening table ----------------------------------------------------------

st.subheader("All names")
table = view.assign(
    week_risk=[risk_badge(a, sd) for a, sd in zip(view["alignment"], view["side"])],
    next_earn=[
        f"⚠️ {d}" if w else (d or DASH)
        for d, w in zip(view["next_earnings"], view["earnings_in_week"])
    ],
)[
    ["ticker", "company", "region", "side", "rank", "prediction", "week_risk", "verdict_label",
     "rsi14", "pct_of_52w_range", "move_since_signal_pct", "ret_1w_pct",
     "eps_surprise_pct", "next_earn", "pe", "ev_ebitda", "n_risks", "n_articles"]
].sort_values(["region", "side", "rank"])

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", pinned=True),
        "company": st.column_config.TextColumn("Company", width="medium"),
        "region": "Region",
        "side": "Side",
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "prediction": st.column_config.NumberColumn("Score", format="%.4f"),
        "week_risk": st.column_config.TextColumn(
            "1-week risk",
            help="What the technical setup implies over the holding week. "
                 "'pullback risk' = a long on a name that has just run; "
                 "'bounce risk' = a short on a name just sold off. Both mean the "
                 "entry is stretched against the position. 'reversion in our favour' "
                 "means the stretch works for it.",
        ),
        "verdict_label": st.column_config.TextColumn("Technical verdict", width="medium"),
        "rsi14": st.column_config.NumberColumn("RSI", format="%.0f"),
        "pct_of_52w_range": st.column_config.ProgressColumn(
            "52w range", format="%.0f%%", min_value=0, max_value=100,
            help="Where the last close sits between the trailing 52-week low "
                 "and high. 0% = at the 52-week low, 100% = at the high.",
        ),
        "move_since_signal_pct": st.column_config.NumberColumn("Since signal", format="%+.1f%%"),
        "ret_1w_pct": st.column_config.NumberColumn("1w", format="%+.1f%%"),
        "eps_surprise_pct": st.column_config.NumberColumn("Last EPS surp.", format="%+.1f%%"),
        "next_earn": st.column_config.TextColumn("Next earnings", help="⚠️ = lands inside the holding week"),
        "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        "ev_ebitda": st.column_config.NumberColumn("EV/EBITDA", format="%.1f"),
        "n_risks": st.column_config.NumberColumn("Risks", format="%d"),
        "n_articles": st.column_config.NumberColumn("Articles", format="%d"),
    },
)
st.caption(
    "Valuation multiples are TTM and in the listing's own currency — UK prices are "
    "GBp pence, Canada CAD, US USD, so levels are not comparable across regions. "
    "A negative P/E means trailing losses, not cheapness."
)

# --- per-name detail ----------------------------------------------------------

st.divider()
st.subheader("Single name")

ordered = view.sort_values(
    ["alignment", "region", "side", "rank"],
    key=lambda s: s.map({"conflict": 0, "aligned": 1, "neutral": 2}) if s.name == "alignment" else s,
)
choice = st.selectbox(
    "Ticker",
    ordered["ticker"].tolist(),
    format_func=lambda t: (
        f"{t} · {view.loc[view.ticker == t, 'region'].iloc[0]} "
        f"{view.loc[view.ticker == t, 'side'].iloc[0].lower()} "
        f"· {risk_badge(view.loc[view.ticker == t, 'alignment'].iloc[0],
                        view.loc[view.ticker == t, 'side'].iloc[0])}"
    ),
    help="Names entering against the one-week reversion are listed first.",
)

rec = tickers[choice]
meta, fun, tech = rec.get("meta") or {}, rec.get("fundamentals") or {}, rec.get("technicals") or {}
news, brief = rec.get("news") or {}, rec.get("narrative")
cur = meta.get("currency") or ""

# The company name wraps happily; the numeric metrics do not — they clip to an
# ellipsis. So weight the width toward the metric columns.
h1, h2, h3, h4 = st.columns([2, 1.4, 1.1, 2])
h1.markdown(f"### {meta.get('company_name') or choice}")
# Currency belongs in the label, not the value: st.metric renders the value in a
# large face and truncates it to an ellipsis ("22.32 …") regardless of column
# width, whereas the label is small text that fits.
h2.metric(f"Last close ({cur})" if cur else "Last close",
          f"{tech['last_close']:,.2f}" if tech.get("last_close") else DASH)
h3.metric("Market cap", f"{meta['market_cap'] / 1e9:,.1f}B" if meta.get("market_cap") else DASH)
# Rendered as markdown, not st.metric: the metric value is a single clipped line,
# which truncated "conflict" to "co…" — the one word most worth reading here.
h4.markdown(
    "1-week risk  \n### "
    + risk_badge(tech.get("alignment", "neutral"), meta.get("side", ""))
)
st.caption(
    f"{meta.get('region')} · {meta.get('sector') or DASH} · {meta.get('industry') or DASH} "
    f"· model rank {meta.get('rank')} on the {str(meta.get('side', '')).lower()} side "
    f"(score {meta.get('prediction'):.4f})" if meta.get("prediction") is not None else ""
)
if brief:
    st.info(f"**{_md(brief['thesis'])}**")

if meta.get("model_commentary"):
    st.caption(f"Model's own reason for the pick: *{_md(meta['model_commentary'])}*")

tab_fund, tab_theme, tab_tech = st.tabs(["Fundamentals", "Theme & news", "Technicals"])

with tab_fund:
    le, ne = fun.get("last_earnings") or {}, fun.get("next_earnings") or {}
    st.markdown("**Most recent reported earnings**")
    if le.get("date"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Reported", le["date"])
        c2.metric(
            "EPS", f"{le['eps_actual']:,.2f}" if le.get("eps_actual") is not None else DASH,
            delta=f"{le['eps_surprise_pct']:+.1f}% vs est" if le.get("eps_surprise_pct") is not None else None,
        )
        c3.metric(
            "Revenue", f"{le['revenue_actual'] / 1e6:,.0f}M" if le.get("revenue_actual") else DASH,
            delta=f"{le['revenue_surprise_pct']:+.1f}% vs est" if le.get("revenue_surprise_pct") is not None else None,
        )
    else:
        st.caption("No reported earnings in the supplied data.")

    st.markdown("**Next earnings**")
    if ne.get("date"):
        if ne.get("in_holding_week"):
            st.warning(
                f"**{ne['date']} — inside the holding week.** An earnings print in the "
                "holding period is event risk the weekly model does not price, and can "
                "swamp a one-week reversion edge in either direction."
            )
        else:
            st.caption(f"{ne['date']} · {ne.get('days_until', DASH)} days away")
        c1, c2 = st.columns(2)
        c1.metric("Consensus EPS", f"{ne['eps_estimated']:,.2f}" if ne.get("eps_estimated") is not None else DASH)
        c2.metric("Consensus revenue",
                  f"{ne['revenue_estimated'] / 1e6:,.0f}M" if ne.get("revenue_estimated") else DASH)
    else:
        st.caption("No scheduled earnings date in the supplied data.")

    st.markdown("**Valuation (TTM)**")
    val = fun.get("valuation") or {}
    # Peer median across the covered names in the same industry — free context,
    # no extra API calls, and far more useful than an absolute multiple alone.
    peers = df[df["industry"] == meta.get("industry")]["ticker"].tolist()
    peer_recs = [tickers[p].get("fundamentals", {}).get("valuation", {}) for p in peers]
    labels = {
        "pe_ttm": "P/E", "pb_ttm": "P/B", "ps_ttm": "P/S", "ev_ebitda_ttm": "EV/EBITDA",
        "ev_sales_ttm": "EV/sales", "ev_fcf_ttm": "EV/FCF",
        "net_debt_ebitda_ttm": "Net debt/EBITDA", "gross_margin_ttm": "Gross margin %",
        "net_margin_ttm": "Net margin %", "roe_ttm": "ROE %", "roic_ttm": "ROIC %",
        "fcf_yield_ttm": "FCF yield %",
    }
    # A "peer median" over a cohort of one is just the name's own number in a
    # second column, which reads as corroboration it is not. Only show it once
    # there are enough covered names in the industry for the median to mean
    # something.
    MIN_PEERS = 3
    rows_v = [{"Metric": label, "Value": val.get(key)} for key, label in labels.items()]
    if len(peers) >= MIN_PEERS:
        for row, key in zip(rows_v, labels):
            row[f"Peer median ({len(peers)} names)"] = pd.Series(
                [p.get(key) for p in peer_recs], dtype="float64"
            ).median()
    vt = pd.DataFrame(rows_v)
    st.dataframe(vt, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in vt.columns if c != "Metric"})
    st.caption(
        (
            f"Peer median is taken across the {len(peers)} covered names in "
            f"**{meta.get('industry')}** on this page, not the full sector — a sanity "
            "check, not a valuation screen."
            if len(peers) >= MIN_PEERS
            else f"No peer column: only {len(peers)} covered name(s) in "
                 f"**{meta.get('industry') or 'this industry'}**, too few for a median. "
                 "Research more names (higher `--top-n`) to populate it."
        )
        + " Multiples are TTM in the listing's own currency."
    )

    cons = fun.get("consensus") or {}
    st.markdown("**Analyst consensus**")
    if cons.get("available"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rating", cons.get("label") or DASH)
        c2.metric("Avg price target",
                  f"{cons['avg_pt_last_month']:,.2f}" if cons.get("avg_pt_last_month") else DASH)
        c3.metric("vs last close",
                  f"{cons['pt_vs_price_pct']:+.1f}%" if cons.get("pt_vs_price_pct") is not None else DASH)
        st.caption(
            f"Strong buy {cons.get('strong_buy', 0)} · buy {cons.get('buy', 0)} · "
            f"hold {cons.get('hold', 0)} · sell {cons.get('sell', 0)} · "
            f"strong sell {cons.get('strong_sell', 0)}"
        )
    else:
        # The artifact's `reason` is already a finished sentence — do not append
        # to it, or the two run together into nonsense.
        st.info(_md(cons.get("reason") or "No analyst consensus available for this listing."))

    if brief and brief.get("fundamental_read"):
        st.markdown("**Read**")
        st.write(_md(brief["fundamental_read"]))

with tab_theme:
    if brief and brief.get("theme"):
        st.markdown("**Theme**")
        st.write(_md(brief["theme"]))
    elif news.get("answer"):
        st.markdown("**Search-engine summary** (no written brief for this name yet)")
        st.write(_md(news["answer"]))
    else:
        st.caption("No theme could be established — no articles were found for this name.")

    if brief:
        if brief.get("headline_risks"):
            st.markdown("**Headline risks**")
            for r in brief["headline_risks"]:
                st.markdown(f"- {_md(r)}")
        if brief.get("catalysts"):
            st.markdown("**Catalysts**")
            for c in brief["catalysts"]:
                st.markdown(f"- {_md(c)}")
        c1, c2 = st.columns(2)
        c1.metric("Evidence vs the model's side", brief.get("model_agreement", DASH))
        c2.metric("Brief confidence", brief.get("confidence", DASH))

    if news.get("answer") and brief and brief.get("theme"):
        with st.expander("Search-engine summary"):
            st.write(_md(news["answer"]))

    st.markdown(
        f"**Articles** · last {news.get('window_days', 30)} days · "
        f"source: `{news.get('source', 'none')}`"
    )
    if news.get("error"):
        st.caption(f"News lookup reported: {_md(news['error'])}")
    articles = news.get("articles") or []
    if not articles:
        st.info(
            "No articles found in the window. For UK and Canadian listings FMP "
            "publishes no news at all, so coverage there depends entirely on web "
            "search — an empty list means nothing was found, not that nothing happened."
        )
    for a in articles:
        st.markdown(
            f"**[{_md(a.get('title')) or 'untitled'}]({a.get('url')})**  \n"
            f"<span style='color:#666;font-size:0.85em'>{a.get('published') or 'undated'}"
            f" · {a.get('publisher') or '?'}</span>",
            unsafe_allow_html=True,
        )
        if a.get("snippet"):
            st.caption(_md(a["snippet"]))

with tab_tech:
    v1, v2, v3 = st.columns([2, 1, 1])
    v1.metric("Verdict", tech.get("verdict_label", DASH))
    v2.metric("RSI(14)", f"{tech['rsi14']:.1f}" if tech.get("rsi14") is not None else DASH)
    v3.metric("Since signal",
              f"{tech['move_since_signal_pct']:+.1f}%" if tech.get("move_since_signal_pct") is not None else DASH)

    if tech.get("verdict_reasons"):
        st.markdown("**Why** " + " · ".join(_md(r) for r in tech["verdict_reasons"]))
    if tech.get("flags"):
        st.markdown("**Flags** " + " · ".join(f"`{f}`" for f in tech["flags"]))

    bars = series_to_frame(tech.get("series"))
    if bars.empty:
        st.caption("No price history stored for this name.")
    else:
        st.plotly_chart(
            dp.price_and_rsi(bars, tech, artifact.get("as_of_date"), choice),
            use_container_width=True,
        )
        st.caption(
            "Twelve months of daily closes with the 20/50/200-day moving averages, "
            "over RSI(14). The vertical rule marks the signal close the model scored; "
            "the two faint horizontals are the 52-week high and low. "
            f"Prices are in {cur or 'the listing currency'}."
        )

    stats = {
        "Last close": tech.get("last_close"), "SMA20": tech.get("sma20"),
        "SMA50": tech.get("sma50"), "SMA200": tech.get("sma200"),
        "vs SMA20 %": tech.get("px_vs_sma20_pct"), "vs SMA50 %": tech.get("px_vs_sma50_pct"),
        "vs SMA200 %": tech.get("px_vs_sma200_pct"), "Return 1w %": tech.get("ret_1w_pct"),
        "Return 1m %": tech.get("ret_1m_pct"), "Return 3m %": tech.get("ret_3m_pct"),
        "52w low": tech.get("range_52w_low"), "52w high": tech.get("range_52w_high"),
        "ATR14 %": tech.get("atr14_pct"), "Ann. 20d vol %": tech.get("vol_20d_ann_pct"),
        "Volume vs 20d avg": tech.get("volume_vs_20d_avg"),
    }
    st.dataframe(
        pd.DataFrame({"Metric": list(stats), "Value": list(stats.values())}),
        use_container_width=True, hide_index=True,
        column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
    )

    if brief and brief.get("technical_read"):
        st.markdown("**Read**")
        st.write(_md(brief["technical_read"]))

# --- provenance ---------------------------------------------------------------

if rec.get("errors"):
    with st.expander(f"Data gaps for {choice} ({len(rec['errors'])})"):
        for e in rec["errors"]:
            st.markdown(f"- {_md(e)}")

if brief and brief.get("sources_used"):
    with st.expander("Sources the brief cites"):
        for s in brief["sources_used"]:
            st.markdown(f"- {s}" if not str(s).startswith("http") else f"- [{s}]({s})")

prov = artifact.get("provenance") or {}
st.divider()
st.caption(
    f"Source: `{path.relative_to(PROCESSED_DIR.parent)}` · generated {generated} · "
    f"{prov.get('n_fmp_calls', 0)} FMP calls, {prov.get('n_cache_hits', 0)} cache hits · "
    f"briefs {prov.get('narrative_coverage', DASH)} via "
    f"`{prov.get('llm_model', artifact.get('config', {}).get('llm_backend', DASH))}` · "
    f"regenerate with `python -m src.research deepdive --signal-day {sd_code}`"
)

# Refresh is opt-in. Every other page in this viewer is strictly read-only, and
# that stays true here unless the operator explicitly unlocks it.
if os.environ.get("DEEPDIVE_ALLOW_REFRESH") == "1" and os.environ.get("FMP_API_KEY"):
    with st.sidebar:
        st.header("Refresh")
        st.caption("Refetches this one name from FMP and the news search.")
        if st.button(f"Refresh {choice}", use_container_width=True):
            from src.research.deepdive import build_artifact, save_artifact

            with st.spinner(f"Refetching {choice}…"):
                fresh = build_artifact(sd_code, target, tickers=[choice], force=True)
                merged = load_artifact(path) or fresh
                merged["tickers"][choice] = fresh["tickers"][choice]
                save_artifact(merged, path)
            _load.clear()
            st.rerun()
