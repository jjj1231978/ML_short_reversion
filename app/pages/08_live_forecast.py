"""Live forecast page: Wed / Thu / Fri rebalance choice.

Reads cached forecasts under `data/processed/forecasts/{signal_day}/` produced
by `python -m src.predict --signal-day {WED|THU|FRI}`. This page is read-only —
it does NOT trigger a new prediction run. To regenerate, run predict.py from
the CLI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.forecasts import (
    SIGNAL_DAY_LABELS,
    code_for_label,
    default_day_index,
    list_forecasts,
)
from src.backtest.return_calibration import (
    annualize_weekly,
    expected_excess_return,
    load_calibration,
)
from src.config import PROCESSED_DIR

PUBLIC_BUILD_NOTE = (
    "The live signal book is not published with this repo — see "
    "*What is and isn't published* in the README. Run the pipeline "
    "locally to populate this page."
)

st.title("Live forecast")
st.caption(
    "Top-10 long and top-10 short picks for the next rebalance close. "
    "Switch the rebalance weekday in the sidebar — Wed/Thu/Fri each have "
    "their own trained model and forecast file. The **commentary** column "
    "names the 1-2 factors (via SHAP over the tree members) that moved each "
    "pick's score the most — e.g. *\"1-week reversal (R1W) high (pushes score "
    "down)\"* reads as: the stock's recent 1-week return is high and the model "
    "expects mean reversion to drag the forward return down."
)


with st.sidebar:
    st.header("Rebalance day")
    sd_label = st.selectbox(
        "Signal day",
        list(SIGNAL_DAY_LABELS.values()),
        index=default_day_index(),
        help=(
            "The weekday whose close is used as the as-of date for the "
            "forecast and the executed rebalance. Per the weekday-effect "
            "diagnostic, Thursday and Friday historically outperform Wednesday."
        ),
    )
    sd_code = code_for_label(sd_label)

    files = list_forecasts(sd_code)
    if not files:
        st.warning(
            f"No forecasts found for {sd_label}.\n\n"
            f"{PUBLIC_BUILD_NOTE}\n\n"
            f"Generate one with:\n\n"
            f"```\npython -m src.predict --signal-day {sd_code}\n```"
        )
    selected_file = None
    if files:
        labels = [f.stem for f in files]
        chosen_label = st.selectbox(
            "Target date",
            labels,
            index=0,
            help="Date format: target weekday whose close ends the holding week.",
        )
        selected_file = files[labels.index(chosen_label)]

if selected_file is None:
    st.info(
        f"Run `python -m src.predict --signal-day {sd_code}` to produce the "
        f"first {sd_label} forecast."
    )
    st.stop()

df = pd.read_parquet(selected_file)
if df.empty:
    st.error(f"Forecast file is empty: {selected_file}")
    st.stop()

# Header metadata
target_date = pd.Timestamp(df["target_date"].iloc[0]).date()
as_of_date = pd.Timestamp(df["as_of_date"].iloc[0]).date()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rebalance day", sd_label)
m2.metric("As-of close", as_of_date.isoformat())
m3.metric("Target close", target_date.isoformat())
m4.metric("Eligible universe", f"{int(df['eligible'].sum()):,} / {len(df):,}")

# Flag any region whose exchange is shut on the target, so the displayed target
# close is not silently read as the date that region actually trades. Absent on
# forecasts written before per-region holiday resolution existed.
if "target_trade_date" in df.columns and "region" in df.columns:
    shifted = (
        df.loc[pd.to_datetime(df["target_trade_date"]).dt.date != target_date,
               ["region", "target_trade_date"]]
        .drop_duplicates("region")
        .sort_values("region")
    )
    if not shifted.empty:
        moves = ", ".join(
            f"{r.region} → {pd.Timestamp(r.target_trade_date).date().isoformat()}"
            for r in shifted.itertuples()
        )
        st.caption(
            f"⚠️ {target_date.isoformat()} is a market holiday for: {moves}. "
            "Those legs close on the date shown."
        )

st.divider()

# Translate the rank-mean prediction into an expected 1-week excess return
# (the alpha the model predicts) via the backtest calibration curve. Optional —
# the page degrades gracefully if the calibration artifact hasn't been built.
_calib = load_calibration()
has_exp_ret = _calib is not None
if has_exp_ret:
    df["exp_1w_excess"] = expected_excess_return(df["prediction"].values, _calib)

eligible_only = df[df["eligible"]].copy()

# Older forecast files predate the per-pick commentary column.
has_commentary = "commentary" in df.columns
_display_cols = (
    ["prediction"]
    + (["exp_1w_excess"] if has_exp_ret else [])
    + ["industry"]
    + (["commentary"] if has_commentary else [])
)


def _fmt_picks(frame: pd.DataFrame) -> pd.DataFrame:
    """Stringify prediction (4dp) and expected excess return (signed %)."""
    out = frame.copy()
    if "prediction" in out:
        out["prediction"] = out["prediction"].map(lambda x: f"{x:.4f}")
    if "exp_1w_excess" in out:
        out = out.rename(columns={"exp_1w_excess": "exp 1w excess"})
        out["exp 1w excess"] = out["exp 1w excess"].map(lambda x: f"{x * 100:+.2f}%")
    return out

with st.sidebar:
    n_top = st.slider("Names per side per region", min_value=5, max_value=25, value=10, step=1)
    st.session_state["top_n"] = n_top

# Per-region tables — each region ranked independently. Pooled rankings carry
# strong region bias because the cross-section is industry-neutralized but not
# region-neutralized; per-region splits give an actionable basket per market.
regions_present = sorted(eligible_only["region"].unique().tolist())
counts = eligible_only["region"].value_counts().to_dict()
st.caption(
    "Per-region top/bottom. Counts of eligible names: "
    + ", ".join(f"**{r}** {counts.get(r, 0)}" for r in regions_present)
)
if has_exp_ret:
    lo = expected_excess_return([0.02], _calib)[0]
    hi = expected_excess_return([0.98], _calib)[0]
    st.caption(
        "**exp 1w excess** = expected 1-week return *relative to the universe* "
        "(the alpha the rank score implies), from the backtest calibration of "
        "prediction rank → realized cross-sectional excess return. Longs are "
        "positive, shorts negative. Strong picks span roughly "
        f"{hi * 100:+.2f}% (top) to {lo * 100:+.2f}% (bottom) per week "
        f"(≈ {annualize_weekly(hi) * 100:+.0f}% to {annualize_weekly(lo) * 100:+.0f}% "
        "annualized). It is a historical average, not a per-name guarantee."
    )

# Long and short tables are stacked full-width (not side-by-side) so the
# commentary column stays visible without horizontal scrolling.
for region in regions_present:
    region_df = (
        eligible_only[eligible_only["region"] == region]
        .sort_values("prediction", ascending=False)
    )
    st.subheader(f"{region}")
    longs = region_df.head(n_top)[_display_cols]
    shorts = region_df.tail(n_top)[_display_cols].iloc[::-1]
    st.markdown(f"**Top {n_top} long ({region})**")
    st.dataframe(_fmt_picks(longs), use_container_width=True)
    st.markdown(f"**Top {n_top} short ({region})**")
    st.dataframe(_fmt_picks(shorts), use_container_width=True)

st.divider()
with st.expander("Combined global ranking (all regions pooled)"):
    pooled = eligible_only.sort_values("prediction", ascending=False)
    st.caption(
        "For reference only — heavy region tilt is expected because the "
        "cross-section is industry-neutralized but not region-neutralized."
    )
    pooled_cols = (
        ["prediction"]
        + (["exp_1w_excess"] if has_exp_ret else [])
        + ["region", "industry"]
        + (["commentary"] if has_commentary else [])
    )
    st.dataframe(_fmt_picks(pooled[pooled_cols]), use_container_width=True)

with st.expander("Filtered-out names (not eligible — failed ADV/price/PIT)"):
    ineligible = df[~df["eligible"]]
    if len(ineligible) == 0:
        st.write("None.")
    else:
        st.dataframe(
            ineligible[["prediction", "region", "industry"]],
            use_container_width=True,
        )

st.caption(f"Source: `{selected_file.relative_to(PROCESSED_DIR.parent)}`")
