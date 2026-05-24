"""Data fetching from Databento / Massive (prices), simfin (fundamentals), FMP (analyst grades)."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config

# Load .env at import so API keys (FMP_API_KEY, DATABENTO_API_KEY, etc.) are
# available to module-level fetchers regardless of entry point.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

log = logging.getLogger(__name__)

DEFAULT_DATABENTO_CACHE = Path("~/data_lake/databento/equities").expanduser()
DEFAULT_MASSIVE_CACHE = Path("~/data_lake/massive").expanduser()
DEFAULT_SIMFIN_CACHE = Path("~/data_lake/simfin").expanduser()
DEFAULT_FMP_CACHE = Path("~/data_lake/fmp").expanduser()


def _resolve_databento_cache_dir(schema: str, override: Path | None = None) -> Path:
    """Resolve the cache directory for a given Databento schema.

    Reads `data.databento_cache_dir` from config, expands `~`, and appends
    `L0/{schema}`.
    """
    if override is not None:
        return override
    base = load_config()["data"].get("databento_cache_dir")
    root = Path(base).expanduser() if base else DEFAULT_DATABENTO_CACHE
    return root / "L0" / schema


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
    elif source == "fmp":
        from src.data.fmp import fetch_price_data_fmp

        rate_limit = (
            kwargs.pop("rate_limit_per_min", None)
            or load_config()["data"].get("fmp_rate_limit_per_min", 240)
        )
        return fetch_price_data_fmp(
            tickers, start, end,
            cache_dir=cache_dir, api_key=api_key, rate_limit_per_min=rate_limit,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown price_source: {source!r}. Use 'databento', 'massive', or 'fmp'.")


def fetch_market_close(
    symbol: str,
    start: str,
    end: str,
    source: str | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    dataset: str = "EQUS.MINI",
) -> pd.Series:
    """Fetch a single market-index close series.

    Dispatches on `source` (or `data.price_source` from config). Each backend
    uses its own cache layout.
    """
    if source is None:
        source = load_config()["data"].get("price_source", "databento")

    if source == "fmp":
        from src.data.fmp import fetch_market_close_fmp

        rate_limit = load_config()["data"].get("fmp_rate_limit_per_min", 240)
        return fetch_market_close_fmp(
            symbol, start, end,
            cache_dir=cache_dir, api_key=api_key, rate_limit_per_min=rate_limit,
        )

    import databento as db

    cache_dir = _resolve_databento_cache_dir("ohlcv-1d", override=cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ohlcv-1d_{dataset}_{symbol}_{start}_{end}.parquet"

    if cache_path.exists():
        log.info(f"Loading cached market close ({symbol}) from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        log.info(f"Fetching market close ({symbol}) from Databento...")
        client = db.Historical(key=api_key) if api_key else db.Historical()
        data = client.timeseries.get_range(
            dataset=dataset, schema="ohlcv-1d", symbols=[symbol],
            start=start, end=end,
        )
        df = data.to_df()
        df.to_parquet(cache_path)
        log.info(f"Cached {len(df)} rows to {cache_path}")

    df = df.reset_index()
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df[ts_col] = _normalize_ts(df[ts_col])
    s = df.set_index(ts_col)["close"]
    s.name = symbol
    s.index.name = "Date"
    return s


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

    cache_dir = _resolve_databento_cache_dir("ohlcv-1d", override=cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"ohlcv-1d_{dataset}_sp500_{start}_{end}.parquet"
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
        base = load_config()["data"].get("massive_cache_dir")
        cache_dir = Path(base).expanduser() if base else DEFAULT_MASSIVE_CACHE
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


def _normalize_ts(s: pd.Series) -> pd.Series:
    """Parse, strip timezone, and normalize to midnight (daily granularity)."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _long_to_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert Databento long-format DataFrame to wide-format dict.

    Input: rows per (date, symbol), columns: open, high, low, close, volume.
    Output: dict of DataFrames, each with DatetimeIndex rows x ticker columns.
    """
    df = df.reset_index()

    # Normalize the timestamp column name (ts_event is the index name from .to_df())
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df[ts_col] = _normalize_ts(df[ts_col])

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

    if cache_dir is None:
        base = load_config()["data"].get("simfin_cache_dir")
        cache_dir = Path(base).expanduser() if base else DEFAULT_SIMFIN_CACHE

    sf.set_api_key(os.environ.get("SIMFIN_API_KEY", "free"))
    sf.set_data_dir(str(cache_dir))

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


def fetch_fundamentals(
    tickers: list[str],
    source: str | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    pub_lag_days: int | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Dispatch fundamentals fetch to simfin or fmp based on config.

    Returns the same dict shape regardless of source: {"PE","PB","ROE","GPOA"}
    of (date × ticker) DataFrames.
    """
    if source is None:
        source = load_config()["data"].get("fundamental_source", "simfin")

    cfg = load_config()["data"]
    if pub_lag_days is None:
        pub_lag_days = cfg.get("fundamental_pub_lag_days", 90)

    if source == "fmp":
        from src.data.fmp import fetch_fundamentals_fmp

        return fetch_fundamentals_fmp(
            tickers,
            cache_dir=cache_dir, api_key=api_key,
            pub_lag_days=pub_lag_days,
            rate_limit_per_min=cfg.get("fmp_rate_limit_per_min", 240),
            **kwargs,
        )
    elif source == "simfin":
        return fetch_fundamentals_simfin(
            tickers, cache_dir=cache_dir, pub_lag_days=pub_lag_days,
        )
    else:
        raise ValueError(f"Unknown fundamental_source: {source!r}. Use 'simfin' or 'fmp'.")


def fetch_analyst_grades(
    tickers: list[str],
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh: bool = False,
    topup_existing: bool = False,
) -> pd.DataFrame:
    """Fetch per-analyst rating-change events from FMP `/stable/grades`.

    Each row is one rating action by one analyst on one ticker on one date,
    with columns: symbol, date, gradingCompany, previousGrade, newGrade, action.
    Action ∈ {'upgrade', 'downgrade', 'maintain'}. This is the timestamped
    event series from which UPDOWN1W can be reconstructed per the industry research
    formula (rolling 7-day up minus down divided by total).

    Cached to a single combined parquet so first-run ~3-4 min for ~425 SP500
    tickers; subsequent runs are instant.

    Args:
        tickers: list of symbols to fetch.
        cache_dir: defaults to ~/data_lake/fmp/.
        api_key: defaults to FMP_API_KEY env var.
        rate_limit_per_min: conservative throttle for Starter tier (250/min).
        refresh: if True, ignore cache and re-fetch.
        topup_existing: if True (and cache exists, `refresh` is False), also
            re-pulls grades for tickers already in cache and merge-dedupes
            against the cache key (symbol, date, gradingCompany, action,
            previousGrade, newGrade). Needed by live-inference paths that
            must catch new rating events since the last full fetch. FMP's
            `/grades` endpoint has no "from" cutoff so this still costs one
            call per ticker — but each call is small.
    """
    import json
    import time
    import urllib.request

    if api_key is None:
        api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FMP_API_KEY missing. Add it to .env at the repo root."
        )

    if cache_dir is None:
        base = load_config()["data"].get("fmp_cache_dir")
        cache_dir = Path(base).expanduser() if base else DEFAULT_FMP_CACHE
    cache_dir = cache_dir / "grades"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "grades_combined.parquet"
    if cache_path.exists() and not refresh:
        log.info(f"Loading cached analyst grades from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Fetch missing tickers (never fetched before)
        missing = sorted(set(tickers) - set(df["symbol"].unique()))
        # Optionally re-fetch all already-cached tickers to catch new events
        # since the last fetch. Dedup happens on the combined dataframe.
        existing_to_topup: list[str] = []
        if topup_existing:
            existing_to_topup = sorted(set(tickers) & set(df["symbol"].unique()))
            log.info(
                f"Top-up enabled — re-pulling {len(existing_to_topup)} cached tickers "
                f"to catch new grade events"
            )
        to_fetch = sorted(set(missing) | set(existing_to_topup))
        if not to_fetch:
            return df
        if missing:
            log.info(f"Cache missing {len(missing)} tickers — fetching incrementally")
        new_rows = _fetch_grades_batch(to_fetch, api_key, rate_limit_per_min)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df["date"] = pd.to_datetime(new_df["date"])
            df = pd.concat([df, new_df], ignore_index=True)
            # Dedupe by the natural event key — same analyst, same day, same
            # rating transition for a ticker is the same event.
            dedupe_cols = [c for c in
                ["symbol", "date", "gradingCompany", "action", "previousGrade", "newGrade"]
                if c in df.columns
            ]
            before = len(df)
            df = df.drop_duplicates(subset=dedupe_cols, keep="first")
            log.info(f"Grades cache: {before:,} → {len(df):,} rows after dedupe")
            df.to_parquet(cache_path)
        return df

    log.info(f"Fetching analyst grades from FMP for {len(tickers)} tickers...")
    rows = _fetch_grades_batch(tickers, api_key, rate_limit_per_min)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(cache_path)
    log.info(f"Cached {len(df):,} grade events to {cache_path}")
    return df


def _fetch_grades_batch(
    tickers: list[str], api_key: str, rate_limit_per_min: int
) -> list[dict]:
    """Sequential per-ticker fetch with rate-limit sleep."""
    import json
    import time
    import urllib.request
    import urllib.error

    sleep_per_call = 60.0 / max(rate_limit_per_min, 1)
    rows: list[dict] = []
    failed = 0
    for i, t in enumerate(tickers):
        url = f"https://financialmodelingprep.com/stable/grades?symbol={t}&apikey={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            for d in data:
                d["symbol"] = t
                rows.append(d)
        except urllib.error.HTTPError as e:
            log.warning(f"Failed to fetch {t} (HTTP {e.code}): skipping")
            failed += 1
        except Exception as e:
            log.warning(f"Failed to fetch {t}: {type(e).__name__}; skipping")
            failed += 1
        if (i + 1) % 50 == 0:
            log.info(f"  fetched {i+1}/{len(tickers)} ({failed} failed so far)")
        time.sleep(sleep_per_call)
    log.info(f"Done: {len(rows):,} events from {len(tickers)-failed}/{len(tickers)} tickers")
    return rows


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
