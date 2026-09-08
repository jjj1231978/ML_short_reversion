"""Live forecast track record: realized returns of the saved baskets.

Reads `data/processed/forecast_track_record_{day}.parquet` produced by
`python -m src.backtest.forecast_track_record --signal-day {day}`. Each row in
that parquet represents one saved forecast; this page renders the per-forecast
table, the cumulative L/S curve, and the mean hit rate across resolved
forecasts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest.live_trackrecord import load_commentary, load_performance
from src.config import PROCESSED_DIR

PUBLIC_BUILD_NOTE = (
    "The live signal book is not published with this repo — see "
    "*What is and isn't published* in the README. Run the pipeline "
    "locally to populate this page."
)

SIGNAL_DAY_LABELS = {"WED": "Wednesday", "THU": "Thursday", "FRI": "Friday"}
REGION_ORDER = ["ALL", "US", "UK", "CA"]
REGION_COLOR = {
    "ALL": "#1f77b4",
    "US":  "#2ca02c",
    "UK":  "#d62728",
    "CA":  "#9467bd",
}

st.title("Live forecast — track record")
st.caption(
    "Realized per-region long/short basket returns for forecasts whose target "
    "Wednesday has already passed. To refresh after a new forecast resolves, "
    "run: `python -m src.backtest.forecast_track_record --signal-day {WED|THU|FRI}`"
)


def _default_day_index() -> int:
    """Default the selector to the signal day with the most recent resolved
    week, so a page reload lands on the day that was most recently run instead
    of always snapping back to Wednesday."""
    codes = list(SIGNAL_DAY_LABELS.keys())
    latest: dict[str, pd.Timestamp] = {}
    for code in codes:
        p = PROCESSED_DIR / f"forecast_track_record_{code}.parquet"
        if not p.exists():
            continue
        try:
            tdf = pd.read_parquet(p, columns=["target_date"])
        except Exception:
            continue
        if not tdf.empty:
            latest[code] = pd.to_datetime(tdf["target_date"]).max()
    if not latest:
        return 0
    return codes.index(max(latest, key=latest.get))


with st.sidebar:
    st.header("Rebalance day")
    sd_label = st.selectbox(
        "Signal day",
        list(SIGNAL_DAY_LABELS.values()),
        index=_default_day_index(),
    )
    sd_code = {v: k for k, v in SIGNAL_DAY_LABELS.items()}[sd_label]

track_path = PROCESSED_DIR / f"forecast_track_record_{sd_code}.parquet"

if not track_path.exists():
    st.warning(
        f"No track record file for {sd_label}.\n\n"
        f"{PUBLIC_BUILD_NOTE}\n\n"
        f"Generate one with:\n\n"
        f"```\npython -m src.backtest.forecast_track_record --signal-day {sd_code}\n```"
    )
    st.stop()

df = pd.read_parquet(track_path)
if df.empty:
    st.warning(f"Track record file is empty for {sd_label}.")
    st.stop()

eval_df = df[df.get("evaluable", False)].copy() if "evaluable" in df.columns else df.copy()
n_total = len(df)
n_eval = len(eval_df)

c1, c2, c3 = st.columns(3)
c1.metric("Forecasts saved", n_total)
c2.metric("Resolved", n_eval)
c3.metric("In-flight", n_total - n_eval)

# --- Weekly market commentary (from live_trackrecord.db) --------------------
st.subheader("Weekly market commentary")
st.caption(
    "Market-development notes + the strategy's realized hit rates per resolved "
    "week. Record a week with `python -m src.backtest.live_trackrecord "
    f"--signal-day {sd_code} --target-date <wed> --commentary-file ...`."
)
comm = load_commentary(sd_code)
perf = load_performance(sd_code)
if comm.empty:
    st.info(
        "No commentary recorded yet for "
        f"{sd_label}. After a week resolves, run "
        f"`python -m src.backtest.live_trackrecord --signal-day {sd_code} "
        "--target-date <target-wed> --commentary-file <notes.txt>`."
    )
else:
    for _, row in comm.sort_values("target_date", ascending=False).iterrows():
        tgt = row["target_date"]
        with st.expander(f"Week ending {tgt}", expanded=False):
            m1, m2, m3 = st.columns(3)
            spx = row.get("spx_ret")
            ndx = row.get("ndx_ret")
            vix = row.get("vix_chg")
            m1.metric("S&P 500", f"{spx:+.2%}" if pd.notna(spx) else "—")
            m2.metric("Nasdaq", f"{ndx:+.2%}" if pd.notna(ndx) else "—")
            m3.metric("VIX change", f"{vix:+.1f} pts" if pd.notna(vix) else "—")
            if row.get("market_commentary"):
                st.markdown(row["market_commentary"])
            # Per-scope realized hit rates / EW L/S for this week
            wk = perf[perf["target_date"] == tgt] if not perf.empty else perf
            if not wk.empty:
                tbl = []
                for _, r in wk.sort_values("scope").iterrows():
                    ss = r["single_stock_hit_rate"]
                    tbl.append({
                        "Scope": r["scope"],
                        "EW L/S": f"{r['ls_ret']:+.2%}" if pd.notna(r["ls_ret"]) else "—",
                        "Long": f"{r['long_ret']:+.2%}" if pd.notna(r["long_ret"]) else "—",
                        "Short": f"{r['short_ret']:+.2%}" if pd.notna(r["short_ret"]) else "—",
                        "Single-stock hit": f"{ss:.0%}" if pd.notna(ss) else "—",
                        "Names": f"{int(r['n_correct'])}/{int(r['n_picks'])}",
                    })
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

if n_eval == 0:
    st.info(
        "No forecasts have resolved yet — either their target Wednesday is in "
        "the future, or the price cache doesn't extend that far. Re-run the "
        "track-record script after the target dates pass."
    )
    pending = df[["as_of_date", "target_date", "signal_day"]].copy()
    if "reason" in df.columns:
        pending["reason"] = df["reason"]
    pending = pending.sort_values("target_date")
    pending["as_of_date"] = pd.to_datetime(pending["as_of_date"]).dt.date
    pending["target_date"] = pd.to_datetime(pending["target_date"]).dt.date
    st.subheader("Pending forecasts")
    st.dataframe(pending, use_container_width=True)
    st.stop()

_fmt = lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
_HAS_LAG = any(c.endswith("_ls_ret_lag") for c in eval_df.columns)

# Per-forecast realized returns
st.subheader("Per-forecast realized L/S returns")
st.caption(
    "Two return bases per week: **no-lag** = signal close → target close; "
    "**lag** = next session (entry) → target, matching the backtest's enforced "
    "1-day execution lag. The lagged number is the *implementable* one — it drops "
    "the first-day move after the signal that you can't actually trade."
)
display = eval_df[["as_of_date"]].copy()
display["as_of_date"] = pd.to_datetime(display["as_of_date"]).dt.date
if "entry_lag_used" in eval_df.columns:
    display["entry (lag)"] = pd.to_datetime(eval_df["entry_lag_used"]).dt.date
display["target_date"] = pd.to_datetime(eval_df["target_date"]).dt.date
for region in REGION_ORDER:
    if f"{region}_ls_ret" in eval_df.columns:
        display[region] = eval_df[f"{region}_ls_ret"].map(_fmt)
        if f"{region}_ls_ret_lag" in eval_df.columns:
            display[f"{region} (lag)"] = eval_df[f"{region}_ls_ret_lag"].map(_fmt)
st.dataframe(display, use_container_width=True)

# Cumulative L/S curves — both bases (lag dashed)
st.subheader("Cumulative long-short return")
fig = go.Figure()
for region in REGION_ORDER:
    col = f"{region}_ls_ret"
    if col not in eval_df.columns:
        continue
    series = eval_df.set_index("target_date")[col].dropna()
    if len(series) > 0:
        cum = (1 + series).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines+markers",
            name=f"{region} (no lag)", line=dict(color=REGION_COLOR.get(region)),
        ))
    lagcol = f"{region}_ls_ret_lag"
    if lagcol in eval_df.columns:
        series_l = eval_df.set_index("target_date")[lagcol].dropna()
        if len(series_l) > 0:
            cum_l = (1 + series_l).cumprod()
            fig.add_trace(go.Scatter(
                x=cum_l.index, y=cum_l.values, mode="lines+markers",
                name=f"{region} (lag)",
                line=dict(color=REGION_COLOR.get(region), dash="dash"),
            ))
fig.update_layout(
    xaxis_title="Target date",
    yaxis_title="Cumulative growth (×)",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Solid = no-lag (signal close → target). Dashed = 1-day-lag (implementable).")

# Summary table: mean L/S, hit rate, cumulative — both bases side by side
st.subheader("Summary across resolved forecasts")
summary_rows = []
for region in REGION_ORDER:
    ls_col = f"{region}_ls_ret"
    if ls_col not in eval_df.columns:
        continue
    ls = eval_df[ls_col].dropna()
    if len(ls) == 0:
        continue
    row = {
        "Region": region,
        "Forecasts": len(ls),
        "Mean L/S (no lag)": f"{ls.mean():+.2%}",
        "Cum L/S (no lag)": f"{(1 + ls).prod() - 1:+.2%}",
    }
    if _HAS_LAG:
        lsl = eval_df[f"{region}_ls_ret_lag"].dropna()
        row["Mean L/S (lag)"] = f"{lsl.mean():+.2%}" if len(lsl) else "—"
        row["Cum L/S (lag)"] = f"{(1 + lsl).prod() - 1:+.2%}" if len(lsl) else "—"
    hr_col = f"{region}_hit_rate"
    row["Mean hit rate"] = (
        f"{eval_df[hr_col].dropna().mean():.1%}"
        if hr_col in eval_df.columns and eval_df[hr_col].notna().any()
        else "—"
    )
    summary_rows.append(row)
if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.caption(
    f"Source: `{track_path.relative_to(PROCESSED_DIR.parent)}` — regenerate with "
    f"`python -m src.backtest.forecast_track_record --signal-day {sd_code}`."
)
