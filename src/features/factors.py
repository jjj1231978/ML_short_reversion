"""Factor computation from price and fundamental data."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Price reversal factors
# ---------------------------------------------------------------------------

def r1w(close: pd.DataFrame) -> pd.DataFrame:
    """1-week return (5 trading days)."""
    return close.pct_change(5)


def irev1w(close: pd.DataFrame) -> pd.DataFrame:
    """Intraday reversal — 1-week intraday return residual.

    Approximated as R1W minus the overnight component.
    For a first pass, this is identical to R1W; refine with open prices later.
    """
    return r1w(close)


def rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Relative Strength Index over `window` days."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Price momentum factors
# ---------------------------------------------------------------------------

def momentum(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Return from `lookback` days ago to `skip` days ago.

    E.g. R3M1M: lookback=63, skip=21.
    """
    return close.shift(skip) / close.shift(lookback) - 1


# ---------------------------------------------------------------------------
# Low risk factors
# ---------------------------------------------------------------------------

def rolling_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling volatility of daily returns."""
    daily_ret = close.pct_change()
    return daily_ret.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)


def rolling_beta(
    close: pd.DataFrame, market_close: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling beta vs market index over `window` days."""
    stock_ret = close.pct_change()
    market_ret = market_close.pct_change()

    cov = stock_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    return cov / var.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Fundamental factors (trailing)
# ---------------------------------------------------------------------------

def price_to_earnings(close: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """Trailing P/E ratio. `eps` should be TTM EPS aligned to daily dates."""
    return close / eps.replace(0, np.nan)


def price_to_book(close: pd.DataFrame, bvps: pd.DataFrame) -> pd.DataFrame:
    """Trailing P/B ratio. `bvps` = book value per share, daily-aligned."""
    return close / bvps.replace(0, np.nan)


def roe(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Return on equity (TTM net income / avg equity)."""
    return net_income / equity.replace(0, np.nan)


def gross_profit_over_assets(
    gross_profit: pd.DataFrame, total_assets: pd.DataFrame
) -> pd.DataFrame:
    """Gross profitability = gross profit / total assets."""
    return gross_profit / total_assets.replace(0, np.nan)
