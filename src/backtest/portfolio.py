"""Portfolio construction and backtesting from model predictions."""

import numpy as np
import pandas as pd

from src.config import load_config


def assign_quintiles(predictions: pd.Series) -> pd.Series:
    """Assign quintile ranks (1-5) cross-sectionally each week."""
    return predictions.groupby(level="date").transform(
        lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    )


def build_long_short_portfolio(
    predictions: pd.Series,
    returns: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Construct equal-weight long/short portfolio from quintile assignments.

    Args:
        predictions: Series indexed by (date, ticker) with predicted scores.
        returns: DataFrame with columns=tickers, index=weekly dates, values=returns.
        cfg: Config dict with backtest settings.

    Returns:
        DataFrame with columns: long_return, short_return, ls_return, turnover.
    """
    if cfg is None:
        cfg = load_config()

    bt_cfg = cfg["backtest"]
    cost_bps = bt_cfg["trading_cost_bps"] / 10_000

    quintiles = assign_quintiles(predictions)

    dates = predictions.index.get_level_values("date").unique().sort_values()
    results = []

    prev_long_tickers = set()
    prev_short_tickers = set()

    for date in dates:
        week_q = quintiles.loc[date]
        week_ret = returns.loc[date] if date in returns.index else pd.Series(dtype=float)

        long_tickers = set(week_q[week_q == 5].index)
        short_tickers = set(week_q[week_q == 1].index)

        # Returns (equal-weighted)
        long_ret = week_ret.reindex(long_tickers).mean() if long_tickers else 0.0
        short_ret = week_ret.reindex(short_tickers).mean() if short_tickers else 0.0
        ls_ret = long_ret - short_ret

        # Turnover (2-way)
        if prev_long_tickers or prev_short_tickers:
            long_turnover = 1 - len(long_tickers & prev_long_tickers) / max(len(long_tickers), 1)
            short_turnover = 1 - len(short_tickers & prev_short_tickers) / max(len(short_tickers), 1)
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0

        # Net of costs
        ls_ret_net = ls_ret - turnover * cost_bps * 2  # both sides

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
