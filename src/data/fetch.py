"""Data fetching from Databento / Massive (prices) and simfin (fundamentals)."""

import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, load_config

log = logging.getLogger(__name__)


def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
    source: str | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data from configured price source.

    Dispatches to Databento or Massive based on config or the `source` argument.

    Args:
        tickers: List of ticker symbols (e.g. ["AAPL", "MSFT"]).
        start: Start date inclusive (ISO 8601, e.g. "2023-04-01").
        end: End date exclusive (ISO 8601, e.g. "2026-04-12").
        source: "databento" or "massive". If None, reads from config.
        cache_dir: Directory for parquet caches. Skips API call if cached.
        api_key: API key for the selected provider. Falls back to env var.
        **kwargs: Passed to the provider-specific function.

    Returns:
        Dict with keys 'Open', 'High', 'Low', 'Close', 'Volume'.
        Each value is a DataFrame with DatetimeIndex rows and ticker columns.
    """
    if source is None:
        cfg = load_config()
        source = cfg["data"].get("price_source", "databento")

    if source == "massive":
        return _fetch_price_massive(tickers, start, end, cache_dir, api_key, **kwargs)
    elif source == "databento":
        return _fetch_price_databento(tickers, start, end, cache_dir, api_key, **kwargs)
    else:
        raise ValueError(f"Unknown price_source: {source!r}. Use 'databento' or 'massive'.")


def _fetch_price_databento(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    dataset: str = "EQUS.MINI",
    preview_cost: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data via Databento."""
    import databento as db

    if cache_dir is None:
        cache_dir = RAW_DIR / "databento"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ohlcv_{dataset}_{start}_{end}.parquet"
    if cache_path.exists():
        log.info(f"Loading cached price data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return _long_to_wide(df)

    client = db.Historical(key=api_key) if api_key else db.Historical()

    if preview_cost:
        cost = client.metadata.get_cost(
            dataset=dataset,
            symbols=tickers,
            schema="ohlcv-1d",
            start=start,
            end=end,
        )
        log.info(f"Databento estimated cost: ${cost:.4f}")

    data = client.timeseries.get_range(
        dataset=dataset,
        schema="ohlcv-1d",
        symbols=tickers,
        start=start,
        end=end,
    )

    df = data.to_df()
    df.to_parquet(cache_path)
    log.info(f"Cached {len(df)} rows to {cache_path}")

    return _long_to_wide(df)


def _fetch_price_massive(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data via Massive (formerly Polygon.io).

    Uses the polygon-api-client (pip install polygon-api-client).
    Set MASSIVE_API_KEY (or POLYGON_API_KEY) environment variable.
    Free tier: 5 calls/min. Paid tiers have higher limits.
    """
    from polygon import RESTClient

    if cache_dir is None:
        cache_dir = RAW_DIR / "massive"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ohlcv_{start}_{end}.parquet"
    if cache_path.exists():
        log.info(f"Loading cached Massive price data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return _long_to_wide(df)

    if api_key is None:
        api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set MASSIVE_API_KEY (or POLYGON_API_KEY) environment variable."
        )

    client = RESTClient(api_key=api_key)

    all_rows = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 5 == 0:
            # Respect free-tier rate limit (5 calls/min)
            time.sleep(12)
        log.info(f"Fetching {ticker} ({i+1}/{len(tickers)}) from Massive...")
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start,
                to=end,
                limit=50000,
            )
            for bar in aggs:
                all_rows.append(
                    {
                        "ts_event": pd.Timestamp(bar.timestamp, unit="ms"),
                        "symbol": ticker,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )
        except Exception:
            log.warning(f"Failed to fetch {ticker}, skipping", exc_info=True)

    if not all_rows:
        raise RuntimeError("No price data retrieved from Massive.")

    df = pd.DataFrame(all_rows)
    df.to_parquet(cache_path)
    log.info(f"Cached {len(df)} rows to {cache_path}")

    return _long_to_wide(df)


def _long_to_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert Databento long-format DataFrame to wide-format dict.

    Input: rows per (date, symbol), columns: open, high, low, close, volume.
    Output: dict of DataFrames, each with DatetimeIndex rows x ticker columns.
    """
    df = df.reset_index()

    # Normalize the timestamp column name (ts_event is the index name from .to_df())
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col]).dt.normalize()  # daily granularity

    result = {}
    for field, col in [
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume"),
    ]:
        if col in df.columns:
            wide = df.pivot_table(index=ts_col, columns="symbol", values=col)
            wide.index.name = "Date"
            result[field] = wide

    return result


def fetch_fundamentals_simfin(
    tickers: list[str] | None = None,
    cache_dir: Path | None = None,
    pub_lag_days: int = 60,
) -> dict[str, pd.DataFrame]:
    """Fetch fundamental factors (PE, PB, ROE, GPOA) via simfin.

    Computes ratios manually from raw income, balance sheet, and price data
    to avoid StockHub reindexing issues. Applies a publication lag to prevent
    look-ahead bias: each report's data only becomes available `pub_lag_days`
    after the Report Date.

    Returns daily-aligned DataFrames (DatetimeIndex rows x ticker columns)
    matching the format of price data, ready for Wednesday resampling.
    """
    import simfin as sf

    sf.set_api_key(os.environ.get("SIMFIN_API_KEY", "free"))
    sf.set_data_dir(str(cache_dir or RAW_DIR / "simfin"))

    log.info("Loading SimFin raw data (income, balance, prices)...")
    income = sf.load_income(variant="ttm", market="us")
    balance = sf.load_balance(variant="ttm", market="us")
    prices = sf.load_shareprices(variant="daily", market="us")

    # Filter to requested tickers if specified
    if tickers:
        available = set(income.index.get_level_values("Ticker"))
        valid = [t for t in tickers if t in available]
        income = income.loc[income.index.get_level_values("Ticker").isin(valid)]
        balance = balance.loc[balance.index.get_level_values("Ticker").isin(valid)]
        prices = prices.loc[prices.index.get_level_values("Ticker").isin(valid)]

    # Build per-ticker fundamental series, lagged by pub_lag_days,
    # then forward-filled onto a daily date index to align with prices.
    price_wide = prices["Close"].unstack("Ticker")
    daily_dates = price_wide.index  # the daily date grid

    # Prepare per-report fundamental values: (Ticker, Report Date) -> metrics
    # Apply publication lag: shift report date forward
    lag = pd.DateOffset(days=pub_lag_days)

    def _to_daily(report_series: pd.Series) -> pd.DataFrame:
        """Convert (Ticker, Report Date) series to daily (Date x Ticker) DataFrame.

        Shifts report dates by pub_lag_days and forward-fills to daily frequency.
        """
        frames = {}
        for ticker in report_series.index.get_level_values("Ticker").unique():
            ts = report_series.loc[ticker].copy()
            ts.index = ts.index + lag  # apply publication lag
            ts = ts[~ts.index.duplicated(keep="last")]  # drop any duplicate dates
            ts = ts.reindex(daily_dates, method="ffill")
            frames[ticker] = ts
        return pd.DataFrame(frames)

    # --- Compute ratios at report level, then spread to daily ---

    # EPS (TTM) for PE
    eps = income["Net Income (Common)"] / income["Shares (Diluted)"].replace(0, float("nan"))
    eps_daily = _to_daily(eps)
    pe = price_wide / eps_daily.replace(0, float("nan"))

    # Book Value Per Share for PB
    bvps = balance["Total Equity"] / income["Shares (Diluted)"].replace(0, float("nan"))
    bvps_daily = _to_daily(bvps)
    pb = price_wide / bvps_daily.replace(0, float("nan"))

    # ROE = Net Income / Total Equity (at report level, no price needed)
    roe_raw = income["Net Income"] / balance["Total Equity"].replace(0, float("nan"))
    roe = _to_daily(roe_raw)

    # GPOA = Gross Profit / Total Assets
    gpoa_raw = income["Gross Profit"] / balance["Total Assets"].replace(0, float("nan"))
    gpoa = _to_daily(gpoa_raw)

    result = {"PE": pe, "PB": pb, "ROE": roe, "GPOA": gpoa}

    for name, df in result.items():
        n_tickers = df.dropna(how="all", axis=1).shape[1]
        log.info(f"  {name}: {n_tickers} tickers, {df.shape[0]} days")

    return result


if __name__ == "__main__":
    cfg = load_config()
    from src.data.universe import get_sp500_tickers

    tickers = get_sp500_tickers()
    source = cfg["data"].get("price_source", "databento")
    print(f"Fetching data for {len(tickers)} tickers via {source}...")

    prices = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        source=source,
    )
    print(f"Price data shape: {prices['Close'].shape}")
