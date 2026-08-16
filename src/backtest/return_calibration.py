"""Translate the ensemble's rank-mean prediction into an expected return.

The live forecaster's `prediction` column is a cross-sectional **percentile
rank** in [0, 1] (rank_mean ensemble — see `src/model/ensemble.py`), not a
return. This module fits a monotone calibration curve from the backtest that
maps that rank to an **expected 1-week cross-sectional excess return** (the
alpha the model actually predicts, since the training target is the
cross-sectionally demeaned forward return).

Excess (not total) return is the right target here: the strategy is long/short
and market-neutral, so a short pick should carry a *negative* expected return.
Total weekly returns are dominated by market drift and would show every name as
mildly positive, which is misleading for the short book.

Build once after a backtest:

    python -m src.backtest.return_calibration

then `load_calibration()` + `expected_excess_return()` convert any forecast's
prediction column to an expected weekly excess return. The Streamlit live
forecast page does exactly this.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, load_config

log = logging.getLogger(__name__)

CALIB_PATH = PROCESSED_DIR / "prediction_return_calibration.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

# Weekly returns past this magnitude are clipped before averaging so a handful
# of split/merger artifacts or biotech blow-ups don't dominate a bin's mean.
_RET_CLIP = 0.5
# Number of trading weeks per year, for annualizing the weekly figure.
_WEEKS_PER_YEAR = 52


def build_calibration(
    cfg: dict | None = None,
    n_bins: int = 20,
    grid_points: int = 101,
    save: bool = True,
) -> pd.DataFrame:
    """Fit rank -> expected weekly excess return from the backtest predictions.

    Joins `predictions.parquet` (WED ensemble ranks) to cross-sectionally
    demeaned forward 1-week returns from the cached close panel, bins by
    prediction quantile, then isotonic-fits (monotone increasing) the binned
    means onto a dense [0, 1] grid.

    Returns a frame with columns ``prediction`` and ``exp_excess_ret`` (the
    lookup table); also written to `CALIB_PATH` when ``save``.
    """
    from sklearn.isotonic import IsotonicRegression

    from src.data.fetch import fetch_price_data
    from src.features.build import resample_to_weekday

    if cfg is None:
        cfg = load_config()
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} missing — run `python -m src.main` first."
        )

    preds = pd.read_parquet(PREDICTIONS_PATH)["prediction"].dropna()
    tickers = preds.index.get_level_values("ticker").unique().tolist()

    px = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        source=cfg["data"].get("price_source", "fmp"),
        skip_fetch=True,  # calibration is offline — never hit the API
    )
    weekly = resample_to_weekday(px["Close"], "WED")
    fwd = weekly.pct_change(fill_method=None).shift(-1).clip(-_RET_CLIP, _RET_CLIP)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)  # demean per WED cross-section
    excess = excess.stack()
    excess.index.names = ["date", "ticker"]
    excess.name = "ex_ret"

    joined = pd.concat([preds, excess], axis=1).dropna()
    if joined.empty:
        raise RuntimeError("No overlap between predictions and realized returns.")

    bins = pd.qcut(joined["prediction"], n_bins, labels=False, duplicates="drop")
    binned = joined.groupby(bins).agg(
        pred=("prediction", "mean"), ex_ret=("ex_ret", "mean")
    )

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(binned["pred"].values, binned["ex_ret"].values)
    grid = np.linspace(0.0, 1.0, grid_points)
    calib = pd.DataFrame({"prediction": grid, "exp_excess_ret": iso.predict(grid)})

    log.info(
        "Calibration: %d obs, %d bins. rank 0.05 -> %.3f%%/wk, "
        "rank 0.95 -> %.3f%%/wk",
        len(joined), binned.shape[0],
        np.interp(0.05, calib["prediction"], calib["exp_excess_ret"]) * 100,
        np.interp(0.95, calib["prediction"], calib["exp_excess_ret"]) * 100,
    )
    if save:
        calib.to_parquet(CALIB_PATH)
        log.info("Wrote calibration: %s", CALIB_PATH)
    return calib


def load_calibration(path=None) -> pd.DataFrame | None:
    """Load the lookup table, or None if it hasn't been built yet."""
    path = path or CALIB_PATH
    if not path.exists():
        return None
    return pd.read_parquet(path)


def expected_excess_return(
    predictions, calib: pd.DataFrame | None = None
) -> np.ndarray | None:
    """Map prediction ranks to expected weekly excess returns via interpolation.

    `predictions` is array-like of rank scores in [0, 1]. Returns an array of
    expected 1-week excess returns (decimal), or None if no calibration exists.
    """
    if calib is None:
        calib = load_calibration()
    if calib is None:
        return None
    return np.interp(
        np.asarray(predictions, dtype=float),
        calib["prediction"].values,
        calib["exp_excess_ret"].values,
    )


def annualize_weekly(weekly_excess) -> np.ndarray:
    """Compound a weekly excess return to an annual figure."""
    w = np.asarray(weekly_excess, dtype=float)
    return (1.0 + w) ** _WEEKS_PER_YEAR - 1.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_calibration()
