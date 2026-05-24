"""Live forecast page: Wed / Thu / Fri rebalance choice.

Reads cached forecasts under `data/processed/forecasts/{signal_day}/` produced
by `python -m src.predict --signal-day {WED|THU|FRI}`. This page is read-only —
it does NOT trigger a new prediction run. To regenerate, run predict.py from
the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR

FORECASTS_ROOT = PROCESSED_DIR / "forecasts"

SIGNAL_DAY_LABELS = {
    "WED": "Wednesday",
    "THU": "Thursday",
    "FRI": "Friday",
}

st.title("Live forecast")
st.caption(
    "Top-10 long and top-10 short picks for the next rebalance close. "
    "Switch the rebalance weekday in the sidebar — Wed/Thu/Fri each have "
    "their own trained model and forecast file."
)


def _list_forecasts(signal_day: str) -> list[Path]:
    """Return parquet files for a given signal day, sorted newest first."""
    d = FORECASTS_ROOT / signal_day
    if not d.exists():
        return []
    return sorted(d.glob("*.parquet"), reverse=True)


with st.sidebar:
    st.header("Rebalance day")
    sd_label = st.selectbox(
        "Signal day",
        list(SIGNAL_DAY_LABELS.values()),
        index=0,
        help=(
            "The weekday whose close is used as the as-of date for the "
            "forecast and the executed rebalance. Per the weekday-effect "
            "diagnostic, Thursday and Friday historically outperform Wednesday."
        ),
    )
    sd_code = {v: k for k, v in SIGNAL_DAY_LABELS.items()}[sd_label]

    files = _list_forecasts(sd_code)
    if not files:
        st.warning(
            f"No forecasts found for {sd_label}.\n\n"
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

st.divider()

eligible_only = df[df["eligible"]].copy()

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

for region in regions_present:
    region_df = (
        eligible_only[eligible_only["region"] == region]
        .sort_values("prediction", ascending=False)
    )
    st.subheader(f"{region}")
    c1, c2 = st.columns(2)
    longs = region_df.head(n_top)[["prediction", "industry"]]
    shorts = region_df.tail(n_top)[["prediction", "industry"]].iloc[::-1]
    with c1:
        st.markdown(f"**Top {n_top} long ({region})**")
        ld = longs.copy()
        ld["prediction"] = ld["prediction"].map(lambda x: f"{x:.4f}")
        st.dataframe(ld, use_container_width=True)
    with c2:
        st.markdown(f"**Top {n_top} short ({region})**")
        sd = shorts.copy()
        sd["prediction"] = sd["prediction"].map(lambda x: f"{x:.4f}")
        st.dataframe(sd, use_container_width=True)

st.divider()
with st.expander("Combined global ranking (all regions pooled)"):
    pooled = eligible_only.sort_values("prediction", ascending=False)
    st.caption(
        "For reference only — heavy region tilt is expected because the "
        "cross-section is industry-neutralized but not region-neutralized."
    )
    st.dataframe(pooled[["prediction", "region", "industry"]], use_container_width=True)

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
