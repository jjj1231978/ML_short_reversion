"""Naive baseline strategies for paper-comparison.

Per industry research p.5, two baselines:
  (a) Basic Price Reversals: sector-relative 1-week total return.
      Long the worst performers (predicted reversion winners), short the best.
  (b) Earnings-Filtered Price Reversals: same as (a) but exclude losers with
      negative EPS revisions and winners with positive revisions. Requires
      UPDOWN1W which is a Phase 2 factor.

The ML strategy should out-perform both baselines net of costs (paper p.6).
"""

from __future__ import annotations

import pandas as pd

from src.features.build import resample_to_weekday
from src.features.factors import r1w


def r1w_reversal_predictions(
    close: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    signal_day: str = "WED",
) -> pd.Series:
    """Sector-relative 1-week reversal baseline.

    Predicted score = -(R1W - sector_mean(R1W)). Negation maps "past loser →
    predicted winner". Forms a (date, ticker) Series the portfolio module can
    consume directly.
    """
    r_daily = r1w(close)
    r_weekly = resample_to_weekday(r_daily, signal_day)
    stacked = r_weekly.stack(future_stack=True)
    stacked.index.names = ["date", "ticker"]
    stacked.name = "prediction"
    stacked = stacked.dropna()

    if sector_map is not None:
        sectors = (
            pd.Series(stacked.index.get_level_values("ticker"), index=stacked.index)
            .map(sector_map)
            .fillna("Unknown")
        )
        grouper = [stacked.index.get_level_values("date"), sectors]
        sector_means = stacked.groupby(grouper).transform("mean")
        stacked = stacked - sector_means

    return -stacked


def earnings_filtered_reversal_predictions(
    close: pd.DataFrame,
    eps_revisions: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    signal_day: str = "WED",
) -> pd.Series:
    """Earnings-filtered reversal baseline (paper p.5).

    Drop losers (R1W in bottom decile) with negative EPS revisions and
    winners (R1W top decile) with positive EPS revisions, then run the
    sector-relative reversal score on what remains.

    Args:
        eps_revisions: (date × ticker) of UPDOWN1W (or similar). Phase 2 only.
    """
    r_daily = r1w(close)
    r_weekly = resample_to_weekday(r_daily, signal_day)
    rev_weekly = resample_to_weekday(eps_revisions, signal_day)

    pred = r_weekly.stack(future_stack=True)
    pred.index.names = ["date", "ticker"]
    pred.name = "raw_r1w"
    pred = pred.dropna()

    rev = rev_weekly.stack(future_stack=True)
    rev.index.names = ["date", "ticker"]
    rev = rev.reindex(pred.index)

    # Per-week deciles of R1W
    deciles = pred.groupby(level="date").transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
    )

    # Filter rule: drop bottom-decile losers with negative revisions, drop
    # top-decile winners with positive revisions.
    drop_loser_with_neg_rev = (deciles == 0) & (rev < 0)
    drop_winner_with_pos_rev = (deciles == 9) & (rev > 0)
    keep = ~(drop_loser_with_neg_rev | drop_winner_with_pos_rev)
    pred = pred[keep]

    if sector_map is not None:
        sectors = (
            pd.Series(pred.index.get_level_values("ticker"), index=pred.index)
            .map(sector_map)
            .fillna("Unknown")
        )
        grouper = [pred.index.get_level_values("date"), sectors]
        pred = pred - pred.groupby(grouper).transform("mean")

    return -pred
