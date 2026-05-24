"""Evaluate the track record of saved live forecasts.

For each parquet under data/processed/forecasts/{signal_day}/, joins the
as_of→target close-to-close return from the cached daily FMP price panel and
emits per-region long/short basket realized metrics. Forecasts whose target
Wednesday has not yet been reached (or whose target close isn't available in
the cache) are flagged `evaluable=False`.

Usage:
    python -m src.backtest.forecast_track_record --signal-day WED
    python -m src.backtest.forecast_track_record --signal-day WED --top-n 10

Output:
    data/processed/forecast_track_record_{signal_day}.parquet

The Streamlit "Track record" page reads this parquet directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.fmp import _load_prices_wide, _resolve_cache_dir

log = logging.getLogger(__name__)

FORECASTS_ROOT = PROCESSED_DIR / "forecasts"


def _safe_ret(close: pd.DataFrame, t_from: pd.Timestamp, t_to: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """Per-ticker total return close-to-close. NaN if either endpoint missing."""
    if t_from not in close.index or t_to not in close.index:
        return pd.Series(np.nan, index=tickers)
    start = close.loc[t_from].reindex(tickers)
    end = close.loc[t_to].reindex(tickers)
    return (end / start) - 1.0


def evaluate_forecast(
    forecast: pd.DataFrame,
    close: pd.DataFrame,
    n_per_side: int = 10,
) -> dict[str, Any]:
    """Compute realized basket metrics for one saved forecast.

    Returns one record (dict) with as_of/target dates, per-region long/short/
    L/S returns, hit rates, plus a pooled "ALL" row. If the target close isn't
    yet available in the cache, returns `evaluable=False` with no metrics.
    """
    as_of = pd.Timestamp(forecast["as_of_date"].iloc[0]).normalize()
    target = pd.Timestamp(forecast["target_date"].iloc[0]).normalize()
    signal_day = forecast["signal_day"].iloc[0]

    # Snap as_of/target to the latest available trading day <= the label.
    # For as_of: forecast was made as-of that Wednesday's close, so the
    # corresponding trading day should be present in the cache.
    avail = close.index
    as_of_actual = avail[avail <= as_of].max() if (avail <= as_of).any() else None
    target_actual = avail[avail <= target].max() if (avail <= target).any() else None

    base = {
        "as_of_date": as_of,
        "target_date": target,
        "signal_day": signal_day,
    }

    # Strict evaluability: the cache must contain a close >= target. Otherwise
    # we'd be measuring a partial holding period.
    if target_actual is None or target_actual < target:
        return {**base, "evaluable": False, "reason": "target close not yet in cache"}
    if as_of_actual is None or as_of_actual < as_of:
        return {**base, "evaluable": False, "reason": "as-of close missing from cache"}

    rets = _safe_ret(close, as_of_actual, target_actual, list(forecast.index))

    eligible = forecast[forecast["eligible"]].copy()

    out: dict[str, Any] = {
        **base,
        "evaluable": True,
        "as_of_used": as_of_actual,
        "target_used": target_actual,
        "n_eligible_total": int(len(eligible)),
    }

    def _basket_metrics(grp: pd.DataFrame, prefix: str) -> None:
        if len(grp) < 2 * n_per_side:
            return
        ranked = grp.sort_values("prediction", ascending=False)
        longs = ranked.head(n_per_side)
        shorts = ranked.tail(n_per_side).iloc[::-1]
        long_r = rets.reindex(longs.index)
        short_r = rets.reindex(shorts.index)
        out[f"{prefix}_long_ret"] = float(long_r.mean())
        out[f"{prefix}_short_ret"] = float(short_r.mean())
        out[f"{prefix}_ls_ret"] = float(long_r.mean() - short_r.mean())
        long_hits = (long_r > 0).sum()
        short_hits = (short_r < 0).sum()
        valid = (~long_r.isna()).sum() + (~short_r.isna()).sum()
        out[f"{prefix}_hit_rate"] = float((long_hits + short_hits) / valid) if valid else np.nan
        out[f"{prefix}_n_long"] = int(len(longs))
        out[f"{prefix}_n_short"] = int(len(shorts))

    for region in ("US", "UK", "CA"):
        region_grp = eligible[eligible["region"] == region]
        _basket_metrics(region_grp, region)

    # Pooled (ALL): rank everything eligible, take top-N/bottom-N. Useful as
    # a reference but bear in mind the cross-section isn't region-neutralized.
    _basket_metrics(eligible, "ALL")
    return out


def evaluate_all_forecasts(
    signal_day: str, close: pd.DataFrame, n_per_side: int = 10
) -> pd.DataFrame:
    """Walk every forecast for `signal_day`, return one summary row per file."""
    d = FORECASTS_ROOT / signal_day.upper()
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.glob("*.parquet"))
    rows: list[dict] = []
    for path in files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        rows.append(evaluate_forecast(df, close, n_per_side=n_per_side))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)


def _load_close_panel(cache_dir: Path | None, tickers: list[str]) -> pd.DataFrame:
    base = _resolve_cache_dir(cache_dir) / "prices" / "by_symbol"
    price_data = _load_prices_wide(tickers, base)
    if "Close" not in price_data:
        raise RuntimeError(f"No Close prices in cache at {base}")
    return price_data["Close"]


def _collect_forecast_tickers(signal_day: str) -> list[str]:
    d = FORECASTS_ROOT / signal_day.upper()
    if not d.exists():
        return []
    tickers: set[str] = set()
    for path in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(path)
        tickers.update(df.index.tolist())
    return sorted(tickers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate realized returns of saved live forecasts."
    )
    parser.add_argument("--signal-day", default="WED",
                        choices=["MON", "TUE", "WED", "THU", "FRI"])
    parser.add_argument("--top-n", type=int, default=10,
                        help="Basket size per region (and pooled). Default 10.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="FMP cache root; defaults to data.fmp_cache_dir.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sd = args.signal_day.upper()

    tickers = _collect_forecast_tickers(sd)
    if not tickers:
        log.error(f"No forecasts found under {FORECASTS_ROOT / sd}/")
        return 2
    log.info(f"Loading close prices for {len(tickers)} tickers from FMP cache...")
    close = _load_close_panel(args.cache_dir, tickers)
    log.info(f"Close panel: {close.shape[0]} days × {close.shape[1]} tickers "
             f"(latest = {close.index.max().date()})")

    log.info(f"Evaluating {sd} forecasts (top_n={args.top_n})...")
    track = evaluate_all_forecasts(sd, close, n_per_side=args.top_n)
    if track.empty:
        log.warning("No forecasts to evaluate.")
        return 0

    n_eval = int(track["evaluable"].sum())
    log.info(f"Evaluated {len(track)} forecast(s); {n_eval} resolved, "
             f"{len(track) - n_eval} still in-flight")

    out_path = PROCESSED_DIR / f"forecast_track_record_{sd}.parquet"
    track.to_parquet(out_path)
    log.info(f"Wrote: {out_path}")

    # Console summary
    print(f"\n=== Forecast track record — {sd} ===")
    if n_eval == 0:
        in_flight = track.copy()
        print(f"All {len(in_flight)} saved forecasts still in-flight "
              f"(target dates in the future or not yet in price cache).")
        print(in_flight[["as_of_date", "target_date", "reason"]].to_string(index=False))
        return 0

    resolved = track[track["evaluable"]].copy()
    cols = ["target_date"]
    for prefix in ("ALL", "US", "UK", "CA"):
        c = f"{prefix}_ls_ret"
        if c in resolved.columns:
            cols.append(c)
    pretty = resolved[cols].copy()
    for c in cols:
        if c.endswith("_ls_ret"):
            pretty[c] = pretty[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
        elif c == "target_date":
            pretty[c] = pretty[c].dt.date
    print(pretty.to_string(index=False))

    if len(resolved) >= 2:
        for prefix in ("ALL", "US", "UK", "CA"):
            c = f"{prefix}_ls_ret"
            if c in resolved.columns:
                ls = resolved[c].dropna()
                if len(ls) >= 2:
                    cum = float((1 + ls).prod() - 1)
                    mean = float(ls.mean())
                    print(f"  {prefix}: cumulative L/S = {cum:+.2%}, "
                          f"mean weekly = {mean:+.2%} ({len(ls)} weeks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
