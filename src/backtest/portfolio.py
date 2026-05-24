"""Portfolio construction and backtesting from model predictions."""

import numpy as np
import pandas as pd

from src.config import load_config


def assign_quintiles(predictions: pd.Series) -> pd.Series:
    """Assign quintile ranks (1-5) cross-sectionally each week.

    Rank-based to handle weeks with heavily-tied predictions (which broke
    `pd.qcut(labels=[1..5], duplicates="drop")` when ties caused < 5 unique
    bin edges). Tickers with NaN predictions remain NaN.
    """
    def _bucket(x: pd.Series) -> pd.Series:
        valid = x.dropna()
        if len(valid) == 0:
            return pd.Series(np.nan, index=x.index)
        # rank(pct=True) → (0, 1]; np.ceil(rank * 5) → ints in {1..5}
        pct = valid.rank(method="first", pct=True)
        bucket = np.ceil(pct * 5).clip(1, 5).astype(int)
        return bucket.reindex(x.index)

    return predictions.groupby(level="date").transform(_bucket)


def _basket_weights(
    tickers: set,
    adv_today: pd.Series | None,
    adv_threshold: "float | pd.Series",
) -> pd.Series:
    """Equal weight if `adv_today` is None, else ADV-scaled weights summing to 1.

    Per paper p.10: stocks with ADV >= threshold get full equal weight; stocks
    below threshold are scaled by ADV/threshold. The basket then re-normalizes.

    `adv_threshold` may be a scalar (single region / paper-faithful) OR a
    Series indexed by ticker (per-region thresholds for multi-region runs).
    """
    if not tickers:
        return pd.Series(dtype=float)
    tickers = list(tickers)
    if adv_today is None:
        return pd.Series(1.0 / len(tickers), index=tickers)

    raw = adv_today.reindex(tickers).fillna(0.0)
    if isinstance(adv_threshold, pd.Series):
        thr = adv_threshold.reindex(tickers).fillna(float("inf"))
        scale = (raw / thr).clip(upper=1.0)
    else:
        scale = (raw / adv_threshold).clip(upper=1.0)
    total = scale.sum()
    if total <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers)
    return scale / total


def build_long_short_portfolio(
    predictions: pd.Series,
    returns: pd.DataFrame,
    adv: pd.DataFrame | None = None,
    cfg: dict | None = None,
    region_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Construct long/short portfolio from quintile assignments.

    Args:
        predictions: Series indexed by (date, ticker) with predicted scores.
        returns: DataFrame with columns=tickers, index=weekly dates, values=returns.
        adv: Optional 6m-rolling average dollar volume (date × ticker). If
            provided, positions are scaled by min(1, ADV/threshold) per
            paper p.10. If None, equal-weighted within basket.
        cfg: Config dict with backtest settings.
        region_map: Optional {ticker: region_code}. When provided alongside
            cfg.universe.adv_scaling_threshold_by_region, applies per-region
            thresholds (paper-faithful $20M USD for US, native-currency
            calibrated values for UK GBp pence and CA CAD).

    Returns:
        DataFrame with columns: long_return, short_return, ls_return,
        ls_return_net, turnover.
    """
    if cfg is None:
        cfg = load_config()

    bt_cfg = cfg["backtest"]
    cost_bps = bt_cfg["trading_cost_bps"] / 10_000

    by_region = cfg["universe"].get("adv_scaling_threshold_by_region")
    if by_region and region_map:
        # Per-ticker threshold series (paper-faithful per region)
        adv_threshold = pd.Series({
            t: by_region.get(region_map.get(t), float("inf"))
            for t in returns.columns
        }, dtype=float)
    else:
        adv_threshold = cfg["universe"].get("adv_scaling_threshold", 20_000_000)

    quintiles = assign_quintiles(predictions)

    dates = predictions.index.get_level_values("date").unique().sort_values()
    results = []

    prev_long_tickers = set()
    prev_short_tickers = set()

    for date in dates:
        week_q = quintiles.loc[date]
        week_ret = returns.loc[date] if date in returns.index else pd.Series(dtype=float)
        adv_today = adv.loc[date] if (adv is not None and date in adv.index) else None

        long_tickers = set(week_q[week_q == 5].index)
        short_tickers = set(week_q[week_q == 1].index)

        long_w = _basket_weights(long_tickers, adv_today, adv_threshold)
        short_w = _basket_weights(short_tickers, adv_today, adv_threshold)

        long_ret = (week_ret.reindex(long_w.index).fillna(0.0) * long_w).sum() if len(long_w) else 0.0
        short_ret = (week_ret.reindex(short_w.index).fillna(0.0) * short_w).sum() if len(short_w) else 0.0
        ls_ret = long_ret - short_ret

        # Turnover (2-way set-difference)
        if prev_long_tickers or prev_short_tickers:
            long_turnover = 1 - len(long_tickers & prev_long_tickers) / max(len(long_tickers), 1)
            short_turnover = 1 - len(short_tickers & prev_short_tickers) / max(len(short_tickers), 1)
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0

        # Net of costs (both sides)
        ls_ret_net = ls_ret - turnover * cost_bps * 2

        results.append({
            "date": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "ls_return": ls_ret,
            "ls_return_net": ls_ret_net,
            "turnover": turnover,
        })

        prev_long_tickers = long_tickers
        prev_short_tickers = short_tickers

    return pd.DataFrame(results).set_index("date")


def compute_performance_metrics(portfolio: pd.DataFrame) -> dict:
    """Calculate key backtest performance metrics."""
    ls = portfolio["ls_return_net"]
    ann_factor = 52  # weekly data

    ann_return = ls.mean() * ann_factor
    ann_vol = ls.std() * np.sqrt(ann_factor)
    ir = ann_return / ann_vol if ann_vol > 0 else 0.0

    cumulative = (1 + ls).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()

    avg_turnover = portfolio["turnover"].mean()

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "information_ratio": ir,
        "max_drawdown": max_dd,
        "avg_weekly_turnover": avg_turnover,
        "num_weeks": len(ls),
    }


if __name__ == "__main__":
    print("Backtest module — run via src.main or import directly.")
