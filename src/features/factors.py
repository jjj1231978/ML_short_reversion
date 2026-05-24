"""Factor computation from price, fundamental, and macro data."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broadcast_to_tickers(series: pd.Series, tickers: list[str]) -> pd.DataFrame:
    """Replicate a single time series across all tickers as a (date × ticker)
    DataFrame. Used for macro features that take the same value for every
    stock on a given date (regime flags, raw macro returns/changes).
    """
    return pd.DataFrame(
        np.tile(series.values.reshape(-1, 1), (1, len(tickers))),
        index=series.index,
        columns=tickers,
    )


# ---------------------------------------------------------------------------
# Price reversal factors
# ---------------------------------------------------------------------------

def r1w(close: pd.DataFrame) -> pd.DataFrame:
    """1-week return (5 trading days)."""
    return close.pct_change(5)


def irev1w(
    close: pd.DataFrame,
    market_close: pd.Series,
    beta: pd.DataFrame,
) -> pd.DataFrame:
    """1-week beta-adjusted residual return (paper p.13).

    residual = stock_1w_return - beta * market_1w_return

    Args:
        close: daily stock closes (date × ticker).
        market_close: daily market index close (date Series).
        beta: rolling beta (date × ticker), same shape as `close`.
    """
    stock_ret = close.pct_change(5)
    market_ret = market_close.pct_change(5)
    expected = beta.mul(market_ret, axis=0)
    return stock_ret - expected


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


def _align_market_to_stocks(close: pd.DataFrame, market_close: pd.Series) -> pd.Series:
    """Reindex market_close to the stocks' (potentially wider) daily calendar
    with forward-fill, so multi-region runs (US ∪ UK ∪ CA dates) don't blow
    holes into the rolling window. On a UK/CA-only date the US market price
    carries over unchanged → pct_change() = 0 on that day, which is the right
    null hypothesis (US market literally didn't move on a US holiday).

    Without this alignment, EVERY rolling-window factor that touches the market
    (BETA, IVOL, IREV, IMOM, …) returns NaN for nearly all rows in a multi-
    region universe, because the default `rolling(window)` requires the full
    window non-NaN and even one NaN-day in 126 trips it.
    """
    return market_close.reindex(close.index).ffill()


def rolling_beta(
    close: pd.DataFrame, market_close: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling beta vs market index over `window` days.

    cov(X, Y) = E[XY] - E[X]E[Y], computed per-ticker against the market.
    """
    stock_ret = close.pct_change()
    market_ret = _align_market_to_stocks(close, market_close).pct_change()

    xy_mean = stock_ret.mul(market_ret, axis=0).rolling(window).mean()
    x_mean = stock_ret.rolling(window).mean()
    y_mean = market_ret.rolling(window).mean()
    cov = xy_mean.sub(x_mean.mul(y_mean, axis=0), axis=0)
    var = market_ret.rolling(window).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def downside_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling semi-deviation — std of negative daily returns only.

    Captures left-tail risk specifically. The paper's low-risk factor group
    treats it as a separate factor from total vol because it tends to be
    more predictive in stress periods.
    """
    daily_ret = close.pct_change()
    neg = daily_ret.where(daily_ret < 0)  # NaN above zero, keep negatives
    return neg.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)


def idiosyncratic_volatility(
    close: pd.DataFrame, market_close: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling stdev of residual = stock_ret - beta_rolling * market_ret.

    `beta_rolling` is the rolling beta over the same window. Annualized.
    This is the IVOL factor in the industry research's low-risk group — captures
    name-specific risk not explained by market exposure.
    """
    stock_ret = close.pct_change()
    market_ret = _align_market_to_stocks(close, market_close).pct_change()
    beta = rolling_beta(close, market_close, window)
    residual = stock_ret.sub(beta.mul(market_ret, axis=0), fill_value=0)
    return residual.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)


def residual_return(
    close: pd.DataFrame,
    market_close: pd.Series,
    beta: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """`period`-day beta-adjusted residual return — generalized IREV.

    residual = stock_period_return - beta * market_period_return

    `period=5` reproduces the paper's IREV1W; `period=21` is IREV1M.
    """
    stock_ret = close.pct_change(period)
    market_ret = _align_market_to_stocks(close, market_close).pct_change(period)
    expected = beta.mul(market_ret, axis=0)
    return stock_ret - expected


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


# ---------------------------------------------------------------------------
# Earnings / analyst-action revisions
# ---------------------------------------------------------------------------

def updown_ratings_1w(
    grade_events: pd.DataFrame,
    daily_dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """7-day rolling (upgrade − downgrade) / total rating actions per ticker.

    Honest naming: this uses analyst RATING revisions (from FMP `/stable/grades`)
    as a proxy for the industry research's UPDOWN1W (which uses EPS estimate revisions
    from IBES). Both share the same formula structure — rolling 7-day count of
    upward vs downward analyst actions normalized by total — but rating-change
    events ≠ EPS-estimate revisions. Keeping the factor name distinct
    (`UPDOWN1W_RATINGS`) prevents confusion in SHAP comparisons against the paper.

    Args:
        grade_events: long-format DataFrame from `fetch_analyst_grades`,
            columns: symbol, date, action ∈ {'upgrade','downgrade','maintain'}.
        daily_dates: Daily date grid the output should be aligned to.
        tickers: Universe to include as columns.

    Returns:
        Wide DataFrame (date × ticker) with values in [-1, 1] (or NaN where
        no events occurred in the trailing 7 days).
    """
    if grade_events is None or grade_events.empty:
        return pd.DataFrame(index=daily_dates, columns=tickers, dtype=float)

    df = grade_events[grade_events["symbol"].isin(tickers)].copy()
    if df.empty:
        return pd.DataFrame(index=daily_dates, columns=tickers, dtype=float)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    action = df["action"].astype(str).str.lower()
    df["dir"] = action.map({"upgrade": 1, "downgrade": -1}).fillna(0).astype(int)
    df["any"] = 1  # all events count toward the denominator

    # Daily counts per (date, ticker)
    by_dt = df.groupby(["date", "symbol"], as_index=False).agg(
        up_dn=("dir", "sum"), total=("any", "sum")
    )

    full_idx = pd.MultiIndex.from_product(
        [daily_dates, tickers], names=["date", "symbol"]
    )
    by_dt = (
        by_dt.set_index(["date", "symbol"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    up_dn = by_dt.pivot(index="date", columns="symbol", values="up_dn").sort_index()
    total = by_dt.pivot(index="date", columns="symbol", values="total").sort_index()

    # 7-calendar-day rolling sum
    up_dn_7d = up_dn.rolling("7D").sum()
    total_7d = total.rolling("7D").sum()

    # "No events in the last 7 days" → factor = 0 (neutral analyst signal),
    # not NaN. Returning NaN for empty windows yielded 70% NaN on US large-caps
    # in 2018+ (analysts don't change ratings every week), which crossed the
    # neutralize_stacked max_missing_factors=10 threshold for ~80% of tickers
    # weekly and collapsed the per-week universe from ~127 → ~25.
    out = up_dn_7d / total_7d.replace(0, np.nan)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Macro factors — three families
#
# Macro time series (oil, DXY, 10Y yield, VIX, …) are scalar series per date.
# To enter a *cross-sectional* feature matrix usefully they must take one of:
#   (a) per-stock sensitivities — rolling betas / correlations of stock
#       returns vs macro returns. Varies across the cross-section → goes
#       through full neutralization like any other factor.
#   (b) regime flags / HMM posteriors — broadcast to every ticker on each
#       date. Constant across the cross-section per week → must be marked
#       as passthrough in `neutralize_stacked` or the z-score collapses
#       them to zero.
#   (c) raw broadcast series — macro returns/changes/levels, replicated
#       across tickers. Same passthrough requirement as (b).
#
# Trees see one weekly cross-section at a time, so (b) and (c) only help via
# tree interactions across many splits. MLP sees them directly as global
# state. SHAP across model classes will show which family carries the load.
# ---------------------------------------------------------------------------

def macro_changes(macro: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Absolute differences. Natural for rates (Δbps) and vol (ΔVIX), where
    pct-change of a level near zero is unstable."""
    return macro.diff(periods)


def macro_returns(macro: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Percentage changes. Natural for prices (oil, gold, FX, indexes)."""
    return macro.pct_change(periods)


def macro_rolling_zscore(macro: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Rolling z-score of each macro level. Captures regime intensity:
    `VIX_z > 1` means elevated vs trailing 6 months. Window in business days.
    Causal — no look-ahead.
    """
    rolling_mean = macro.rolling(window, min_periods=window // 2).mean()
    rolling_std = macro.rolling(window, min_periods=window // 2).std()
    return (macro - rolling_mean) / rolling_std.replace(0, float("nan"))


def macro_regime_flags(
    macro: pd.DataFrame, window: int = 126, hi_threshold: float = 1.0
) -> pd.DataFrame:
    """Threshold-based regime flags — the cheap baseline for the HMM comparison.

    Each input macro column produces two flag columns:
        `{NAME}_HI`        = 1 if rolling z-score > `hi_threshold` else 0
        `{NAME}_TREND_UP`  = 1 if current level > rolling median else 0

    Causal (uses only past data within each rolling window). These exist
    primarily as a control for the HMM-based regime features: if HMM
    posteriors don't out-rank these in SHAP, the HMM isn't carrying its weight.
    """
    z = macro_rolling_zscore(macro, window=window)
    med = macro.rolling(window, min_periods=window // 2).median()
    flags = {}
    for col in macro.columns:
        flags[f"{col}_HI"] = (z[col] > hi_threshold).astype(float)
        flags[f"{col}_TREND_UP"] = (macro[col] > med[col]).astype(float)
    return pd.DataFrame(flags, index=macro.index)


def stock_macro_beta(
    stock_close: pd.DataFrame,
    macro_series: pd.Series,
    window: int = 126,
) -> pd.DataFrame:
    """Rolling beta of each stock's daily return vs one macro's daily return.

    cov(stock_ret, macro_ret) / var(macro_ret), both over `window` days.
    Macro aligned to the stock-price calendar and forward-filled across the
    upstream gaps already filled in the fetcher.

    Output shape matches `stock_close` (date × ticker). Varies across stocks
    on each date, so it's a normal cross-sectional factor (NOT passthrough).
    """
    stock_ret = stock_close.pct_change()
    macro_ret = macro_series.reindex(stock_close.index).ffill().pct_change()

    xy_mean = stock_ret.mul(macro_ret, axis=0).rolling(window).mean()
    x_mean = stock_ret.rolling(window).mean()
    y_mean = macro_ret.rolling(window).mean()
    cov = xy_mean.sub(x_mean.mul(y_mean, axis=0), axis=0)
    var = macro_ret.rolling(window).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def stock_macro_corr(
    stock_close: pd.DataFrame,
    macro_series: pd.Series,
    window: int = 126,
) -> pd.DataFrame:
    """Rolling Pearson correlation of stock returns vs a macro's returns.

    Bounded in [-1, 1] — more stable than beta because it's not sensitive
    to macro-vol regime shifts. Useful when the macro is itself heteroskedastic
    (oil, VIX) and a beta would oscillate just because the denominator moves.
    """
    stock_ret = stock_close.pct_change()
    macro_ret = macro_series.reindex(stock_close.index).ffill().pct_change()

    xy_mean = stock_ret.mul(macro_ret, axis=0).rolling(window).mean()
    x_mean = stock_ret.rolling(window).mean()
    y_mean = macro_ret.rolling(window).mean()
    cov = xy_mean.sub(x_mean.mul(y_mean, axis=0), axis=0)
    x_var = stock_ret.rolling(window).var()
    y_var = macro_ret.rolling(window).var()
    denom = (x_var.mul(y_var, axis=0)).pow(0.5)
    return cov.div(denom.replace(0, np.nan), axis=0)
