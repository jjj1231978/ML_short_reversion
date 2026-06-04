"""FMP (Financial Modeling Prep) data fetchers.

Single-vendor backfill for prices, fundamentals, indexes, company profiles, and
universe membership across US (SP500), UK (LSE), and Canada (TSX). Designed to
be re-run safely: per-ticker parquet caches with non-destructive merge updates.

All endpoints use the `/stable/` namespace; the legacy `/api/v3/*` endpoints
return 403 on Premium plans. Stable single-symbol-only — no comma-separated
batching is supported (verified empirically against the live API).

Public entry points:
    fetch_index_membership_fmp(index, ...)        # SP500/FTSE100/TSX60 membership
    fetch_symbol_changes_fmp(...)                  # ticker rename history
    fetch_company_profiles_fmp(tickers, ...)       # sector/industry/exchange map
    fetch_price_data_fmp(tickers, start, end, ...) # daily OHLCV (equities and indexes)
    fetch_market_close_fmp(symbol, start, end, ...) # single-series convenience
    fetch_fundamentals_fmp(tickers, ...)           # PE/PB/ROE/GPOA dict
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env at import so FMP_API_KEY is available regardless of entry point.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

log = logging.getLogger(__name__)

DEFAULT_FMP_CACHE = Path("~/data_lake/fmp").expanduser()
FMP_BASE = "https://financialmodelingprep.com/stable"

# Trailing window re-fetched on each incremental update. Captures FMP back-revisions
# (split adjustments, late corrections) without rewriting historical rows.
DEFAULT_REFRESH_DAYS = 7


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Resolve the FMP cache root, honoring config override if available."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    try:
        from src.config import load_config

        base = load_config()["data"].get("fmp_cache_dir")
        if base:
            return Path(base).expanduser()
    except Exception:
        pass
    return DEFAULT_FMP_CACHE


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise EnvironmentError("FMP_API_KEY missing. Add it to .env at the repo root.")
    return key


def _fmp_url(path: str, api_key: str, **params) -> str:
    """Build an FMP stable-API URL with apikey appended."""
    qs = urllib.parse.urlencode({**{k: v for k, v in params.items() if v is not None}, "apikey": api_key})
    return f"{FMP_BASE}/{path.lstrip('/')}?{qs}"


def _fetch_with_429_retry(url: str, timeout: int = 30, max_retries: int = 1) -> list | dict:
    """HTTP GET with one retry on 429. On second 429, raises.

    FMP returns no `Retry-After` header; sleep 60s by default.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                log.warning(f"HTTP 429 from FMP — sleeping 60s, then retrying once")
                time.sleep(60)
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            raise
    if last_error:
        raise last_error
    raise RuntimeError("unreachable")


def _merge_and_write(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> int:
    """Append-only merge: existing rows win on key collision. Atomic write.

    Reads `path` if it exists, identifies rows in `new_df` whose key tuple is
    NOT already present, concatenates, then atomically replaces `path`.

    Returns the number of NEW rows added (0 if nothing new).

    Atomic rename guarantees a SIGINT mid-write doesn't corrupt the cache.
    """
    if not new_df.empty:
        # De-dupe within new_df itself first (FMP can return duplicates)
        new_df = new_df.drop_duplicates(subset=key_cols, keep="first")

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        if new_df.empty:
            return 0
        # Build a set of existing key tuples for membership test
        existing_keys = set(map(tuple, existing[key_cols].itertuples(index=False, name=None)))
        new_keys = list(map(tuple, new_df[key_cols].itertuples(index=False, name=None)))
        mask = [k not in existing_keys for k in new_keys]
        new_only = new_df.iloc[mask]
        if new_only.empty:
            return 0
        out = pd.concat([existing, new_only], ignore_index=True)
        added = len(new_only)
    else:
        if new_df.empty:
            return 0
        out = new_df
        added = len(out)

    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp)
    tmp.replace(path)
    return added


def _normalize_ts(s: pd.Series) -> pd.Series:
    """Parse to datetime, strip tz, normalize to midnight."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _throttle_sleep(rate_limit_per_min: int) -> float:
    """Seconds to sleep per call to respect a per-minute rate limit."""
    return 60.0 / max(rate_limit_per_min, 1)


# -----------------------------------------------------------------------------
# Membership: SP500 (point-in-time), FTSE 100 / TSX 60 (current snapshot only)
# -----------------------------------------------------------------------------


def fetch_index_membership_fmp(
    index: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh: bool = False,
    min_market_cap: float = 5_000_000_000,
    country_filter: str | None = None,
) -> pd.DataFrame:
    """Fetch and cache index membership for SP500 / FTSE100 / TSX60.

    SP500: pulls current snapshot + full historical event stream and derives a
    point-in-time `(symbol, date_added, date_removed, ...)` interval table.

    FTSE100 / TSX60: FMP exposes no historical constituent stream for these
    indexes. We approximate the universe via `/stable/company-screener` filtered
    by exchange, market cap, and (optionally) country. This is current-snapshot
    only — survivorship-biased, accepted as a documented limitation of the data
    source (see `.specify/001-overview.md`).

    Args:
        index: "sp500" | "ftse100" | "tsx60".
        min_market_cap: cap floor for the screener-based UK/CA universes.
        country_filter: e.g. "GB" to restrict LSE to UK-domiciled, "CA" for TSX.
            Set None to keep all listings on the exchange.

    Returns:
        DataFrame with columns:
            [symbol, date_added, date_removed, sector, sub_industry, region, exchange]
        For UK/CA, `date_added` is the (current) snapshot fetch date, `date_removed`
        is NaT. For SP500, both reflect actual historical events.
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = _resolve_cache_dir(cache_dir)
    sleep = _throttle_sleep(rate_limit_per_min)

    index = index.lower()
    if index == "sp500":
        return _fetch_sp500_membership(cache_dir, api_key, sleep, refresh)
    elif index == "ftse100":
        return _fetch_screener_membership(
            cache_dir=cache_dir,
            api_key=api_key,
            exchange="LSE",
            region="UK",
            cache_name="ftse100",
            min_market_cap=min_market_cap,
            country_filter=country_filter or "GB",
            refresh=refresh,
        )
    elif index == "tsx60":
        return _fetch_screener_membership(
            cache_dir=cache_dir,
            api_key=api_key,
            exchange="TSX",
            region="CA",
            cache_name="tsx60",
            min_market_cap=min_market_cap,
            country_filter=country_filter or "CA",
            refresh=refresh,
        )
    else:
        raise ValueError(f"Unknown index: {index!r}. Use 'sp500', 'ftse100', or 'tsx60'.")


def _fetch_sp500_membership(
    cache_dir: Path, api_key: str, sleep: float, refresh: bool
) -> pd.DataFrame:
    """SP500 point-in-time membership intervals derived from FMP event stream.

    Caches:
        constituents/sp500_current.parquet      — current snapshot, append-only
        constituents/sp500_events_raw.parquet   — full event log, append-only
        constituents/sp500_membership.parquet   — derived intervals (rebuilt each run)
    """
    base = cache_dir / "constituents"
    base.mkdir(parents=True, exist_ok=True)
    current_path = base / "sp500_current.parquet"
    events_path = base / "sp500_events_raw.parquet"
    membership_path = base / "sp500_membership.parquet"

    if membership_path.exists() and not refresh:
        log.info(f"Loading cached SP500 membership from {membership_path}")
        return pd.read_parquet(membership_path)

    # 1) Current snapshot — defines who is a member as of today
    log.info("Fetching SP500 current snapshot from FMP...")
    current_raw = _fetch_with_429_retry(_fmp_url("sp500-constituent", api_key))
    time.sleep(sleep)
    current = pd.DataFrame(current_raw)
    if current.empty:
        raise RuntimeError("FMP returned empty SP500 current snapshot")
    _merge_and_write(current_path, current, key_cols=["symbol"])

    # 2) Historical events — full log of adds/removes since 1957
    log.info("Fetching SP500 historical event stream from FMP...")
    events_raw = _fetch_with_429_retry(_fmp_url("historical-sp500-constituent", api_key))
    time.sleep(sleep)
    events = pd.DataFrame(events_raw)
    if events.empty:
        raise RuntimeError("FMP returned empty SP500 historical events")
    _merge_and_write(events_path, events, key_cols=["date", "symbol"])

    # 3) Derive point-in-time intervals
    membership = _derive_sp500_intervals(current, events)
    membership.to_parquet(membership_path)
    log.info(f"Derived {len(membership)} SP500 membership intervals → {membership_path}")
    return membership


def _derive_sp500_intervals(current: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build (symbol, date_added, date_removed) intervals from the event stream.

    Each event row contains BOTH the added ticker (`symbol` / `addedSecurity`)
    AND the corresponding removed ticker (`removedTicker` / `removedSecurity`)
    on the same `date`. We process events chronologically:
        - For each event date d:
              - the `removedTicker` (if non-empty): close any OPEN interval for it at d
              - the added `symbol`: open a new interval starting at d
        - Tickers in the current snapshot that have no open interval get one
          from their `dateFirstAdded` (or NaT if missing).
        - Tickers in the current snapshot with an open interval keep date_removed=NaT.

    Sector info comes from the current snapshot when available; for delisted
    tickers it is None (would require a per-ticker profile fetch — too expensive).
    """
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    events = events.sort_values("date", ascending=True).reset_index(drop=True)

    open_intervals: dict[str, pd.Timestamp] = {}
    closed_intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    for _, row in events.iterrows():
        d = row["date"]
        removed = (row.get("removedTicker") or "").strip()
        added_security = (row.get("addedSecurity") or "").strip()
        added = (row.get("symbol") or "").strip() if added_security else ""

        if removed and removed in open_intervals:
            start = open_intervals.pop(removed)
            closed_intervals.append((removed, start, d))
        elif removed:
            # Removed without a prior add we tracked — still record so we
            # know when it left, with NaT start (pre-event-stream membership)
            closed_intervals.append((removed, pd.NaT, d))

        if added:
            # Reopen if it had been closed historically
            open_intervals[added] = d

    # Tickers currently in the SP500 snapshot keep open intervals (date_removed=NaT)
    current_symbols = set(current["symbol"].astype(str))
    rows = []

    # Closed intervals first
    for sym, start, end in closed_intervals:
        rows.append({"symbol": sym, "date_added": start, "date_removed": end})

    # Open intervals from the event stream → must be in current snapshot
    for sym, start in open_intervals.items():
        if sym in current_symbols:
            rows.append({"symbol": sym, "date_added": start, "date_removed": pd.NaT})
        else:
            # Open in events but not in current snapshot — treat as removed
            # at the latest event date we saw for safety
            rows.append({"symbol": sym, "date_added": start, "date_removed": pd.NaT})

    # Tickers in current snapshot that we never saw added in events
    # (members from before 1957 or ones the event stream missed)
    seen_in_events = {r["symbol"] for r in rows}
    for _, c in current.iterrows():
        sym = str(c["symbol"])
        if sym not in seen_in_events:
            first_added = c.get("dateFirstAdded")
            try:
                first_added = pd.to_datetime(first_added) if first_added else pd.NaT
            except Exception:
                first_added = pd.NaT
            rows.append({"symbol": sym, "date_added": first_added, "date_removed": pd.NaT})

    membership = pd.DataFrame(rows)

    # Attach sector / sub_industry from current snapshot where possible
    sector_map = current.set_index("symbol")[["sector", "subSector"]].to_dict(orient="index")
    membership["sector"] = membership["symbol"].map(lambda s: sector_map.get(s, {}).get("sector"))
    membership["sub_industry"] = membership["symbol"].map(
        lambda s: sector_map.get(s, {}).get("subSector")
    )
    membership["region"] = "US"
    membership["exchange"] = "NYSE/NASDAQ"

    return membership.sort_values(["symbol", "date_added"]).reset_index(drop=True)


def _fetch_screener_membership(
    cache_dir: Path,
    api_key: str,
    exchange: str,
    region: str,
    cache_name: str,
    min_market_cap: float,
    country_filter: str | None,
    refresh: bool,
) -> pd.DataFrame:
    """Approximate UK/CA universe via /stable/company-screener.

    No historical event stream available — current snapshot only.
    Survivorship-biased; documented limitation.
    """
    base = cache_dir / "constituents"
    base.mkdir(parents=True, exist_ok=True)
    snapshot_path = base / f"{cache_name}_current.parquet"

    if snapshot_path.exists() and not refresh:
        log.info(f"Loading cached {cache_name} snapshot from {snapshot_path}")
        return pd.read_parquet(snapshot_path)

    log.info(f"Fetching {cache_name} via screener (exchange={exchange}, country={country_filter})")
    raw = _fetch_with_429_retry(
        _fmp_url(
            "company-screener",
            api_key,
            exchange=exchange,
            marketCapMoreThan=int(min_market_cap),
            isEtf="false",
            isFund="false",
            limit=500,
        )
    )
    df = pd.DataFrame(raw)
    if df.empty:
        raise RuntimeError(f"FMP screener returned empty {cache_name} universe")

    if country_filter:
        df = df[df["country"] == country_filter].reset_index(drop=True)

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    membership = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date_added": today,
            "date_removed": pd.NaT,
            "sector": df["sector"],
            "sub_industry": df["industry"],
            "region": region,
            "exchange": df["exchangeShortName"],
        }
    )
    membership.to_parquet(snapshot_path)
    log.info(f"Cached {len(membership)} {cache_name} members → {snapshot_path}")
    return membership


# -----------------------------------------------------------------------------
# Symbol changes
# -----------------------------------------------------------------------------


def fetch_symbol_changes_fmp(
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh: bool = False,
    limit: int = 5000,
) -> pd.DataFrame:
    """Fetch FMP's symbol-change log. Append-only cache.

    Used to map old tickers (e.g. FB) to current ones (META) before issuing
    price requests, since FMP's price endpoint serves history under the
    CURRENT symbol only.
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = _resolve_cache_dir(cache_dir)
    base = cache_dir / "symbol_changes"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "symbol_changes_raw.parquet"

    if path.exists() and not refresh:
        log.info(f"Loading cached symbol changes from {path}")
        return pd.read_parquet(path)

    log.info("Fetching symbol-change log from FMP...")
    raw = _fetch_with_429_retry(_fmp_url("symbol-change", api_key, limit=limit))
    time.sleep(_throttle_sleep(rate_limit_per_min))

    df = pd.DataFrame(raw)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    added = _merge_and_write(path, df, key_cols=["date", "oldSymbol", "newSymbol"])
    log.info(f"Symbol changes cache: {added} new rows; total now in {path}")
    return pd.read_parquet(path) if path.exists() else df


# -----------------------------------------------------------------------------
# Company profiles
# -----------------------------------------------------------------------------


def fetch_company_profiles_fmp(
    tickers: list[str],
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    stale_after_days: int = 90,
) -> pd.DataFrame:
    """Fetch profile (sector/industry/exchange/currency/country) per ticker.

    Combined parquet at profiles/profiles_combined.parquet.
    Skips tickers already cached unless the file is older than `stale_after_days`.
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = _resolve_cache_dir(cache_dir)
    base = cache_dir / "profiles"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "profiles_combined.parquet"
    sleep = _throttle_sleep(rate_limit_per_min)

    cached_symbols: set[str] = set()
    if path.exists():
        # Only re-fetch if the file is older than stale_after_days
        mtime_age = (time.time() - path.stat().st_mtime) / 86400
        existing = pd.read_parquet(path)
        if mtime_age < stale_after_days:
            cached_symbols = set(existing["symbol"].astype(str))

    missing = [t for t in tickers if t not in cached_symbols]
    if not missing:
        log.info(f"All {len(tickers)} profile entries already cached")
        return pd.read_parquet(path)

    log.info(f"Fetching {len(missing)} profile entries from FMP (have {len(cached_symbols)} cached)")
    rows: list[dict] = []
    failed = 0
    for i, t in enumerate(missing):
        try:
            data = _fetch_with_429_retry(_fmp_url("profile", api_key, symbol=t))
            if isinstance(data, list) and data:
                rows.append(data[0])
        except urllib.error.HTTPError as e:
            log.warning(f"Profile fetch failed for {t} (HTTP {e.code}); skipping")
            failed += 1
        except Exception as e:
            log.warning(f"Profile fetch failed for {t} ({type(e).__name__}); skipping")
            failed += 1
        if (i + 1) % 50 == 0:
            log.info(f"  profiles {i+1}/{len(missing)} ({failed} failed)")
        time.sleep(sleep)

    if rows:
        df_new = pd.DataFrame(rows)
        added = _merge_and_write(path, df_new, key_cols=["symbol"])
        log.info(f"Profiles: added {added} new rows ({failed} fetch failures)")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# Prices (equities AND indexes — same endpoint)
# -----------------------------------------------------------------------------


def fetch_price_data_fmp(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    skip_fetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for many tickers. Returns wide dict (Open/High/.../Volume).

    Per-ticker parquet at prices/by_symbol/{TICKER}.parquet — non-destructive on
    re-run: trailing `refresh_days` re-pulled to absorb FMP back-revisions, but
    older rows are NEVER overwritten.

    Note: same endpoint serves indexes; just pass `^GSPC` etc. as a ticker.

    Args:
        skip_fetch: if True, bypass all HTTP calls and load whatever is in the
            cache. Tickers without a cache file are silently dropped from the
            returned wide matrix. Used by `predict.py --no-refresh-prices` for
            fast iteration once the cache is warm.
    """
    api_key = _resolve_api_key(api_key) if not skip_fetch else None
    cache_dir = _resolve_cache_dir(cache_dir)
    base = cache_dir / "prices" / "by_symbol"
    if skip_fetch:
        log.info(f"skip_fetch=True — loading prices from cache only ({base})")
        return _load_prices_wide(tickers, base)
    base.mkdir(parents=True, exist_ok=True)
    sleep = _throttle_sleep(rate_limit_per_min)

    failed = 0
    fetched = 0
    skipped_uptodate = 0

    for i, t in enumerate(tickers):
        path = base / f"{_safe_filename(t)}.parquet"
        # Build the list of (from, to) ranges we still need to fetch.
        # - If no cache: fetch the full requested range.
        # - If cache covers requested range: fetch only the trailing refresh window.
        # - If requested start is BEFORE cached start: also fetch the missing
        #   leading range (allows backward extension of an existing cache).
        ranges_to_fetch: list[tuple[str, str]] = []
        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty:
                cached_min = pd.Timestamp(existing["date"].min())
                cached_max = pd.Timestamp(existing["date"].max())
                req_start = pd.Timestamp(start)
                req_end = pd.Timestamp(end)

                # Backward gap: requested start before cached start
                if req_start < cached_min:
                    leading_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    ranges_to_fetch.append((start, leading_end))

                # Forward refresh: trailing window
                refresh_start = max(req_start, cached_max - pd.Timedelta(days=refresh_days))
                if req_end >= refresh_start:
                    ranges_to_fetch.append((refresh_start.strftime("%Y-%m-%d"), end))
                elif not ranges_to_fetch:
                    skipped_uptodate += 1
                    continue
            else:
                ranges_to_fetch.append((start, end))
        else:
            ranges_to_fetch.append((start, end))

        try:
            any_new = False
            for r_start, r_end in ranges_to_fetch:
                new_df = _fetch_one_price(t, r_start, r_end, api_key)
                if new_df is not None and not new_df.empty:
                    _merge_and_write(path, new_df, key_cols=["symbol", "date"])
                    any_new = True
            if any_new:
                fetched += 1
        except urllib.error.HTTPError as e:
            log.warning(f"Price fetch failed for {t} (HTTP {e.code}); skipping")
            failed += 1
        except Exception as e:
            log.warning(f"Price fetch failed for {t} ({type(e).__name__}: {e}); skipping")
            failed += 1
        if (i + 1) % 100 == 0:
            log.info(f"  prices {i+1}/{len(tickers)} ({fetched} updated, {skipped_uptodate} cached, {failed} failed)")
        time.sleep(sleep)

    log.info(f"Prices done: {fetched} updated, {skipped_uptodate} cached, {failed} failed")

    # Refuse to load a wide matrix if too many tickers failed (likely throttled).
    # Only enforce on batches large enough for the rate to be statistically meaningful;
    # a single missing ticker in a 3-element benchmark list shouldn't abort the run.
    if len(tickers) >= 20 and failed / len(tickers) > 0.05:
        raise RuntimeError(
            f"Price fetch failure rate {failed}/{len(tickers)} > 5%. "
            "Refusing to build wide matrix from a partial cache. Investigate before retrying."
        )

    return _load_prices_wide(tickers, base)


def _fetch_one_price(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame | None:
    """Single-ticker price fetch, return as long-format DataFrame."""
    url = _fmp_url("historical-price-eod/full", api_key, symbol=symbol, **{"from": start, "to": end})
    data = _fetch_with_429_retry(url)
    if not isinstance(data, list) or not data:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def _load_prices_wide(tickers: list[str], base: Path) -> dict[str, pd.DataFrame]:
    """Concat all per-symbol parquets and return wide dict."""
    frames = []
    for t in tickers:
        path = base / f"{_safe_filename(t)}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return {}
    long = pd.concat(frames, ignore_index=True)
    long["ts_event"] = _normalize_ts(long["date"])

    result: dict[str, pd.DataFrame] = {}
    for field, col in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
        if col in long.columns:
            wide = long.pivot_table(index="ts_event", columns="symbol", values=col)
            wide.index.name = "Date"
            result[field] = wide
    return result


def fetch_market_close_fmp(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
) -> pd.Series:
    """Fetch a single index/symbol close series. Mirrors fetch_market_close signature.

    Uses the same `/stable/historical-price-eod/full` endpoint as equities —
    works fine for index symbols like ^GSPC, ^FTSE, ^GSPTSE.
    """
    result = fetch_price_data_fmp(
        [symbol], start, end, cache_dir=cache_dir, api_key=api_key,
        rate_limit_per_min=rate_limit_per_min, refresh_days=refresh_days,
    )
    if "Close" not in result or symbol not in result["Close"].columns:
        raise RuntimeError(f"FMP returned no close data for {symbol}")
    s = result["Close"][symbol].dropna().copy()
    s.name = symbol
    s.index.name = "Date"
    return s


def fetch_treasury_rates_fmp(
    start: str,
    end: str | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
) -> pd.DataFrame:
    """Fetch the US Treasury par-yield curve from FMP `/stable/treasury-rates`.

    The CBOE yield *indexes* (^IRX/^TNX/^TYX) live in FMP's index namespace,
    which is gated above Premium and 402s on the equity EOD endpoint. The
    `treasury-rates` endpoint IS on Premium and is strictly better for our use:
    it returns the full par-yield curve (month1..year30) as real percentages
    rather than the ×10 CBOE convention, and downstream change/z-score/beta
    factors are scale-invariant either way.

    Cached to `<root>/macro/treasury_rates.parquet`, keyed on date, with the same
    trailing-`refresh_days` incremental re-pull semantics as the price cache:
    older rows are immutable, only the recent window is re-fetched.

    Returns a DataFrame indexed by tz-naive midnight-normalized date with one
    column per tenor (e.g. "month3", "year10", "year30").
    """
    api_key = _resolve_api_key(api_key)
    cache_dir = _resolve_cache_dir(cache_dir)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = cache_dir / "macro" / "treasury_rates.parquet"

    # Incremental: re-pull only the trailing window when a cache already exists.
    fetch_start = start
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.empty:
            last = pd.to_datetime(existing["date"]).max()
            fetch_start = (last - pd.Timedelta(days=refresh_days)).strftime("%Y-%m-%d")

    data = _fetch_with_429_retry(
        _fmp_url("treasury-rates", api_key, **{"from": fetch_start, "to": end})
    )
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)
        df["date"] = _normalize_ts(df["date"])
        _merge_and_write(path, df, key_cols=["date"])

    if not path.exists():
        raise RuntimeError("FMP returned no treasury-rates data")

    out = pd.read_parquet(path)
    out["date"] = _normalize_ts(out["date"])
    return out.set_index("date").sort_index()


def _safe_filename(symbol: str) -> str:
    """Make a symbol safe for use as a filename (^GSPC, BRK-B, etc.)."""
    return symbol.replace("/", "_").replace("\\", "_")


# -----------------------------------------------------------------------------
# Fundamentals: PE / PB / ROE / GPOA
# -----------------------------------------------------------------------------


def fetch_fundamentals_fmp(
    tickers: list[str],
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
    pub_lag_days: int = 90,
    quarters_on_update: int = 8,
    initial_quarters: int = 120,
    price_data: dict[str, pd.DataFrame] | None = None,
    skip_fetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch quarterly statements and compute PE, PB, ROE, GPOA.

    Per-ticker parquets at fundamentals/{income,balance}_by_symbol/{TICKER}.parquet.
    On update, only the last `quarters_on_update` quarters are re-pulled; existing
    rows win on key collision (point-in-time integrity — restated quarters do NOT
    overwrite the originally-cached version).

    Returns the same dict shape as `fetch_fundamentals_simfin` so downstream
    feature code is unchanged: {"PE", "PB", "ROE", "GPOA"} → DataFrames of
    daily-aligned values (DatetimeIndex × ticker columns).

    Args:
        price_data: pre-fetched wide dict from fetch_price_data_fmp. If None,
            we fetch internally for the daily-alignment step (1 extra call/ticker).
        skip_fetch: if True, bypass HTTP and compute ratios from whatever
            quarterly parquets are already cached. Used by
            `predict.py --no-refresh-fundamentals` for fast iteration.
    """
    api_key = _resolve_api_key(api_key) if not skip_fetch else None
    cache_dir = _resolve_cache_dir(cache_dir)

    income_dir = cache_dir / "fundamentals" / "income_by_symbol"
    balance_dir = cache_dir / "fundamentals" / "balance_by_symbol"
    cashflow_dir = cache_dir / "fundamentals" / "cashflow_by_symbol"

    if skip_fetch:
        log.info(f"skip_fetch=True — computing ratios from cached fundamentals only")
        return _compute_fundamental_ratios(
            tickers, income_dir, balance_dir, cashflow_dir, price_data, pub_lag_days
        )

    sleep = _throttle_sleep(rate_limit_per_min)
    income_dir.mkdir(parents=True, exist_ok=True)
    balance_dir.mkdir(parents=True, exist_ok=True)
    cashflow_dir.mkdir(parents=True, exist_ok=True)

    failed_income = 0
    failed_balance = 0
    failed_cashflow = 0
    for i, t in enumerate(tickers):
        income_path = income_dir / f"{_safe_filename(t)}.parquet"
        balance_path = balance_dir / f"{_safe_filename(t)}.parquet"
        cashflow_path = cashflow_dir / f"{_safe_filename(t)}.parquet"
        income_limit = quarters_on_update if income_path.exists() else initial_quarters
        balance_limit = quarters_on_update if balance_path.exists() else initial_quarters
        cashflow_limit = quarters_on_update if cashflow_path.exists() else initial_quarters

        try:
            data = _fetch_with_429_retry(
                _fmp_url("income-statement", api_key, symbol=t, period="quarter", limit=income_limit)
            )
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                _merge_and_write(income_path, df, key_cols=["symbol", "date"])
        except Exception as e:
            log.warning(f"Income fetch failed for {t} ({type(e).__name__}); skipping")
            failed_income += 1
        time.sleep(sleep)

        try:
            data = _fetch_with_429_retry(
                _fmp_url("balance-sheet-statement", api_key, symbol=t, period="quarter", limit=balance_limit)
            )
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                _merge_and_write(balance_path, df, key_cols=["symbol", "date"])
        except Exception as e:
            log.warning(f"Balance fetch failed for {t} ({type(e).__name__}); skipping")
            failed_balance += 1
        time.sleep(sleep)

        # Cash-flow statement — required for OCF/FCF/CAPEX-based Tier-2 factors
        # (PSALES, POCF, PFCF, EVOCF, EVFCF, FCFROIC, CAPEXPPE, STKP, EXTFA).
        try:
            data = _fetch_with_429_retry(
                _fmp_url("cash-flow-statement", api_key, symbol=t, period="quarter", limit=cashflow_limit)
            )
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                _merge_and_write(cashflow_path, df, key_cols=["symbol", "date"])
        except Exception as e:
            log.warning(f"Cashflow fetch failed for {t} ({type(e).__name__}); skipping")
            failed_cashflow += 1
        time.sleep(sleep)

        if (i + 1) % 50 == 0:
            log.info(
                f"  fundamentals {i+1}/{len(tickers)} "
                f"(income fail: {failed_income}, bal fail: {failed_balance}, cf fail: {failed_cashflow})"
            )

    total_failures = failed_income + failed_balance + failed_cashflow
    if tickers and total_failures / (3 * len(tickers)) > 0.05:
        log.warning(
            f"Fundamentals failure rate >5% "
            f"({failed_income}+{failed_balance}+{failed_cashflow}/{3*len(tickers)}). "
            "Computed ratios may be sparse for affected tickers."
        )

    # Aggregate per-symbol parquets and compute ratios
    return _compute_fundamental_ratios(
        tickers, income_dir, balance_dir, cashflow_dir, price_data, pub_lag_days
    )


def _compute_fundamental_ratios(
    tickers: list[str],
    income_dir: Path,
    balance_dir: Path,
    cashflow_dir: Path | None,
    price_data: dict[str, pd.DataFrame] | None,
    pub_lag_days: int,
) -> dict[str, pd.DataFrame]:
    """Compute PE, PB, ROE, GPOA, plus the Tier-2 paper factor set from cached
    statement parquets.

    Mirrors the SimFin path's logic at fetch.py:_compute (income/balance ratios
    forward-filled to daily after pub_lag_days). PE/PB need price data — if not
    provided, we pull `Close` series from cached price parquets.

    `cashflow_dir` may be None for back-compat (legacy callers); OCF/FCF/CAPEX-
    based factors are silently skipped in that case.
    """
    income_frames = []
    balance_frames = []
    cashflow_frames = []
    for t in tickers:
        ip = income_dir / f"{_safe_filename(t)}.parquet"
        bp = balance_dir / f"{_safe_filename(t)}.parquet"
        if ip.exists():
            income_frames.append(pd.read_parquet(ip))
        if bp.exists():
            balance_frames.append(pd.read_parquet(bp))
        if cashflow_dir is not None:
            cp = cashflow_dir / f"{_safe_filename(t)}.parquet"
            if cp.exists():
                cashflow_frames.append(pd.read_parquet(cp))

    if not income_frames or not balance_frames:
        log.warning("No fundamental data cached; returning empty ratio dict")
        return {"PE": pd.DataFrame(), "PB": pd.DataFrame(), "ROE": pd.DataFrame(), "GPOA": pd.DataFrame()}

    income = pd.concat(income_frames, ignore_index=True)
    balance = pd.concat(balance_frames, ignore_index=True)
    cashflow = pd.concat(cashflow_frames, ignore_index=True) if cashflow_frames else pd.DataFrame()

    # Need Close prices for PE/PB
    if price_data is None or "Close" not in price_data:
        from src.data.fetch import DEFAULT_FMP_CACHE  # local import to avoid cycle
        cache_dir = DEFAULT_FMP_CACHE
        price_path_dir = cache_dir / "prices" / "by_symbol"
        frames = []
        for t in tickers:
            p = price_path_dir / f"{_safe_filename(t)}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            log.warning("No cached prices available — PE/PB will be empty")
            price_wide = pd.DataFrame()
        else:
            long = pd.concat(frames, ignore_index=True)
            long["date"] = pd.to_datetime(long["date"]).dt.normalize()
            price_wide = long.pivot_table(index="date", columns="symbol", values="close")
    else:
        price_wide = price_data["Close"]

    daily_index = price_wide.index if not price_wide.empty else pd.DatetimeIndex([])
    lag = pd.DateOffset(days=pub_lag_days)

    def _to_daily(report_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Pivot per-quarter values to daily ticker matrix with publication lag.

        Reindex-then-ffill (NOT reindex-with-ffill): the latter is row-wise
        and copies NaN cells from neighbor tickers' report-date rows, leaving
        the target ticker NaN on every other ticker's report date. The
        two-step form ffills per column instead, so a ticker's value
        propagates daily until its next own report.
        """
        if report_df.empty or daily_index.empty:
            return pd.DataFrame()
        df = report_df[["symbol", "date", value_col]].dropna(subset=[value_col]).copy()
        df["date"] = pd.to_datetime(df["date"]) + lag
        df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
        wide = df.pivot_table(index="date", columns="symbol", values=value_col)
        return wide.reindex(daily_index).ffill()

    # FMP serves quarterly statements; industry research / SimFin path use TTM. Roll up
    # net income and gross profit to trailing-4-quarter sums per ticker.
    income = income.sort_values(["symbol", "date"]).copy()
    income["_netIncome_ttm"] = (
        income.groupby("symbol")["netIncome"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    )
    income["_grossProfit_ttm"] = (
        income.groupby("symbol")["grossProfit"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    )
    # EBIT-TTM — prefer FMP's `ebit` field; fall back to `operatingIncome`. Computed
    # unconditionally (not gated on `interestExpense`) so EVEBIT and GBROC can
    # consume it without depending on the EBIT_COVER branch below.
    _ebit_src = "ebit" if "ebit" in income.columns else (
        "operatingIncome" if "operatingIncome" in income.columns else None
    )
    if _ebit_src is not None:
        income["_ebit_ttm"] = (
            income.groupby("symbol")[_ebit_src].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        )
    # EPS-TTM = TTM net income / latest diluted shares (uses the share count from the same row;
    # share count is point-in-time and shouldn't be summed).
    income["_eps_ttm"] = income["_netIncome_ttm"] / income["weightedAverageShsOutDil"].replace(0, np.nan)

    # PE
    eps_daily = _to_daily(income, "_eps_ttm")
    pe = price_wide / eps_daily.replace(0, np.nan) if not eps_daily.empty else pd.DataFrame()

    # BVPS = totalEquity / weightedAverageShsOutDil at the same date.
    income_keyed = income.set_index(["symbol", "date"])
    balance_keyed = balance.set_index(["symbol", "date"])
    common_idx = income_keyed.index.intersection(balance_keyed.index)
    shares = income_keyed.loc[common_idx, "weightedAverageShsOutDil"].replace(0, np.nan)
    bvps_q = pd.DataFrame(index=common_idx)
    bvps_q["_bvps"] = balance_keyed.loc[common_idx, "totalEquity"] / shares
    bvps_q = bvps_q.reset_index()
    bvps_daily = _to_daily(bvps_q, "_bvps")
    pb = price_wide / bvps_daily.replace(0, np.nan) if not bvps_daily.empty else pd.DataFrame()

    # ROE-TTM = TTM NetIncome / TotalEquity (point-in-time)
    ni_ttm = income_keyed.loc[common_idx, "_netIncome_ttm"]
    equity = balance_keyed.loc[common_idx, "totalEquity"].replace(0, np.nan)
    roe_q = pd.DataFrame(index=common_idx)
    roe_q["_roe"] = ni_ttm / equity
    roe_q = roe_q.reset_index()
    roe = _to_daily(roe_q, "_roe")

    # GPOA-TTM = TTM GrossProfit / TotalAssets (point-in-time)
    gp_ttm = income_keyed.loc[common_idx, "_grossProfit_ttm"]
    assets = balance_keyed.loc[common_idx, "totalAssets"].replace(0, np.nan)
    gpoa_q = pd.DataFrame(index=common_idx)
    gpoa_q["_gpoa"] = gp_ttm / assets
    gpoa_q = gpoa_q.reset_index()
    gpoa = _to_daily(gpoa_q, "_gpoa")

    # --- Phase-2 expansion factors -----------------------------------------
    # All built from columns already present in FMP /stable/income-statement
    # and /stable/balance-sheet-statement — no extra API calls. Each is
    # guarded with a column-presence check so a missing field never aborts
    # the whole fundamentals path; absent columns yield empty frames that
    # `build_feature_matrix` simply skips.
    def _col(df_keyed: pd.DataFrame, name: str) -> pd.Series | None:
        if name in df_keyed.columns:
            return df_keyed[name].replace(0, np.nan)
        return None

    extras: dict[str, pd.DataFrame] = {}

    # DE = totalDebt / totalEquity
    total_debt = _col(balance_keyed, "totalDebt")
    if total_debt is None:
        # Fall back to short + long if totalDebt isn't surfaced
        std = _col(balance_keyed, "shortTermDebt")
        ltd = _col(balance_keyed, "longTermDebt")
        if std is not None and ltd is not None:
            total_debt = std.fillna(0) + ltd.fillna(0)
    if total_debt is not None:
        de_q = (total_debt / equity).reset_index()
        de_q.columns = ["symbol", "date", "_de"]
        extras["DE"] = _to_daily(de_q, "_de")

    # NDE = (totalDebt - cash) / totalEquity. Captures net leverage.
    cash = _col(balance_keyed, "cashAndCashEquivalents")
    if total_debt is not None and cash is not None:
        nde_q = ((total_debt.fillna(0) - cash.fillna(0)) / equity).reset_index()
        nde_q.columns = ["symbol", "date", "_nde"]
        extras["NDE"] = _to_daily(nde_q, "_nde")

    # EBIT_COVER = TTM EBIT / TTM Interest Expense. Interest coverage ratio.
    # Higher is safer. NaN'd where interest expense is zero/missing. Consumes
    # `_ebit_ttm` from the unconditional rollup above.
    if "_ebit_ttm" in income.columns and "interestExpense" in income.columns:
        income["_intexp_ttm"] = (
            income.groupby("symbol")["interestExpense"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        )
        income["_ebit_cover"] = income["_ebit_ttm"] / income["_intexp_ttm"].replace(0, np.nan)
        extras["EBIT_COVER"] = _to_daily(income, "_ebit_cover")

    # SALES_GROWTH_YOY = revenue / revenue 4-quarters-ago - 1.
    # Simple and reliable; doesn't need 5y of history like the paper's
    # SALES_5Y_GROWTH does. Use as the growth factor for Phase 2.
    if "revenue" in income.columns:
        income = income.sort_values(["symbol", "date"]).copy()
        income["_rev_yoy"] = (
            income.groupby("symbol")["revenue"].pct_change(periods=4)
        )
        extras["SALES_GROWTH_YOY"] = _to_daily(income, "_rev_yoy")

    # ACCRUALS — Sloan-style approximation:
    #   accruals = Δ(currentAssets - cash) - Δ(currentLiabilities - shortTermDebt) - depreciation
    #   / totalAssets
    # We omit depreciation when not present and use a simplified version.
    # High accruals → low future return (anomaly per Sloan 1996).
    have_acc = all(
        c in balance.columns
        for c in ("totalCurrentAssets", "totalCurrentLiabilities", "cashAndCashEquivalents")
    )
    if have_acc:
        bal = balance.sort_values(["symbol", "date"]).copy()
        bal["_nca"] = bal["totalCurrentAssets"].fillna(0) - bal["cashAndCashEquivalents"].fillna(0)
        bal["_ncl"] = bal["totalCurrentLiabilities"].fillna(0) - bal.get(
            "shortTermDebt", pd.Series(0, index=bal.index)
        ).fillna(0)
        bal["_d_nca"] = bal.groupby("symbol")["_nca"].diff()
        bal["_d_ncl"] = bal.groupby("symbol")["_ncl"].diff()
        bal["_accruals"] = (bal["_d_nca"] - bal["_d_ncl"]) / bal["totalAssets"].replace(0, np.nan)
        extras["ACCRUALS"] = _to_daily(bal, "_accruals")

    # ----- Tier-2 paper factors (page 13–14) ----------------------------
    # Strategy: roll up TTM versions of all income/cashflow line items first,
    # then form the ratios. Every factor is guarded by a column-presence check
    # so partial data never aborts the pipeline.
    income = income.sort_values(["symbol", "date"]).copy()

    def _ttm(df: pd.DataFrame, col: str) -> pd.Series | None:
        """4-quarter rolling sum of `col` per symbol; None if column missing."""
        if col not in df.columns:
            return None
        return (
            df.groupby("symbol")[col]
            .rolling(4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # TTM rollups from income statement (revenue/EBITDA/R&D — net income and
    # gross profit already rolled up above as _netIncome_ttm and _grossProfit_ttm;
    # EBIT already rolled up above as _ebit_ttm).
    rev_ttm = _ttm(income, "revenue")
    if rev_ttm is not None:
        income["_revenue_ttm"] = rev_ttm
    ebitda_ttm = _ttm(income, "ebitda")
    if ebitda_ttm is not None:
        income["_ebitda_ttm"] = ebitda_ttm
    rd_ttm = _ttm(income, "researchAndDevelopmentExpenses")
    if rd_ttm is not None:
        income["_rd_ttm"] = rd_ttm

    income_keyed = income.set_index(["symbol", "date"])
    balance_keyed = balance.set_index(["symbol", "date"])

    # Helper: a daily-aligned series for a column on the income frame.
    def _income_daily(col: str) -> pd.DataFrame:
        if col not in income.columns:
            return pd.DataFrame()
        return _to_daily(income, col)

    def _balance_daily(col: str) -> pd.DataFrame:
        if col not in balance.columns:
            return pd.DataFrame()
        return _to_daily(balance, col)

    # Daily-aligned shares outstanding (diluted). Drives MktCap and SIZE.
    shares_daily = _income_daily("weightedAverageShsOutDil")

    # Market cap (native currency). Used as the denominator for the Value group
    # (PSALES, POCF, PFCF, PTB, STKP) and as a building block for EV.
    # Use `where(... > 0, np.nan)` so the result keeps float dtype AND
    # bypasses an issue where the multiplication can leave object-dtype
    # intermediate cells (e.g., when shares_daily comes in as object after
    # `_to_daily` reindex+ffill). np.log needs float input.
    if not shares_daily.empty and not price_wide.empty:
        mktcap = price_wide.reindex_like(shares_daily) * shares_daily
        mktcap = mktcap.where(mktcap > 0, np.nan)
    else:
        mktcap = pd.DataFrame()

    # SIZE = ln(market cap). Paper Low-Risk group (Universe z-score).
    if not mktcap.empty:
        extras["SIZE"] = np.log(mktcap)

    # Enterprise Value = MktCap + TotalDebt − Cash.
    total_debt_daily = _balance_daily("totalDebt")
    if total_debt_daily.empty:
        # Fall back to ST + LT debt
        std_daily = _balance_daily("shortTermDebt")
        ltd_daily = _balance_daily("longTermDebt")
        if not std_daily.empty and not ltd_daily.empty:
            total_debt_daily = std_daily.fillna(0).add(ltd_daily.fillna(0), fill_value=0)
    cash_daily = _balance_daily("cashAndCashEquivalents")
    if not mktcap.empty and not total_debt_daily.empty and not cash_daily.empty:
        ev = (
            mktcap.fillna(0)
            .add(total_debt_daily.fillna(0), fill_value=0)
            .sub(cash_daily.fillna(0), fill_value=0)
        )
        ev = ev.replace(0, np.nan)
    else:
        ev = pd.DataFrame()

    # Daily-aligned TTM line items (in same shape as price_wide).
    sales_daily = _income_daily("_revenue_ttm")
    gp_ttm_daily = _income_daily("_grossProfit_ttm")
    ebit_ttm_daily = _income_daily("_ebit_ttm") if "_ebit_ttm" in income.columns else pd.DataFrame()
    ebitda_ttm_daily = _income_daily("_ebitda_ttm")
    rd_ttm_daily = _income_daily("_rd_ttm")
    ni_ttm_daily = _income_daily("_netIncome_ttm") if "_netIncome_ttm" in income.columns else pd.DataFrame()

    # Daily-aligned balance-sheet items.
    assets_daily = _balance_daily("totalAssets")
    equity_daily = _balance_daily("totalEquity")
    lt_debt_daily = _balance_daily("longTermDebt")
    intangibles_daily = _balance_daily("intangibleAssets")
    ppe_daily = _balance_daily("propertyPlantEquipmentNet")
    ca_daily = _balance_daily("totalCurrentAssets")
    cl_daily = _balance_daily("totalCurrentLiabilities")

    # Capital Invested = Equity + Long-Term Debt (paper formula).
    if not equity_daily.empty and not lt_debt_daily.empty:
        cap_invested = equity_daily.fillna(0).add(lt_debt_daily.fillna(0), fill_value=0)
        cap_invested = cap_invested.replace(0, np.nan)
    else:
        cap_invested = pd.DataFrame()

    # Net Working Capital = Current Assets − Current Liabilities.
    if not ca_daily.empty and not cl_daily.empty:
        nwc = ca_daily.fillna(0).sub(cl_daily.fillna(0), fill_value=0)
    else:
        nwc = pd.DataFrame()

    # --- Cash-flow TTM rollups (gracefully empty if cashflow data is missing) ---
    if not cashflow.empty:
        cashflow = cashflow.sort_values(["symbol", "date"]).copy()
        # FMP /stable/cash-flow-statement field names. The OCF line shows up as
        # `operatingCashFlow` (preferred) or `netCashProvidedByOperatingActivities`.
        ocf_col = None
        for c in ("operatingCashFlow", "netCashProvidedByOperatingActivities"):
            if c in cashflow.columns:
                ocf_col = c
                break
        if ocf_col is not None:
            cashflow["_ocf_ttm"] = (
                cashflow.groupby("symbol")[ocf_col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
            )
        if "freeCashFlow" in cashflow.columns:
            cashflow["_fcf_ttm"] = (
                cashflow.groupby("symbol")["freeCashFlow"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
            )
        if "capitalExpenditure" in cashflow.columns:
            cashflow["_capex_ttm"] = (
                cashflow.groupby("symbol")["capitalExpenditure"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
            )
        # Net stock issuance: FMP exposes both gross issued and repurchased.
        for c in ("commonStockIssued", "commonStockRepurchased"):
            if c in cashflow.columns:
                cashflow[f"_{c}_ttm"] = (
                    cashflow.groupby("symbol")[c].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
                )
        # Daily-aligned series for ratio formation. Use cashflow's own _to_daily.
        def _cf_daily(col: str) -> pd.DataFrame:
            if col not in cashflow.columns:
                return pd.DataFrame()
            return _to_daily(cashflow, col)

        ocf_daily = _cf_daily("_ocf_ttm")
        fcf_daily = _cf_daily("_fcf_ttm")
        capex_daily = _cf_daily("_capex_ttm")
        stock_issued_daily = _cf_daily("_commonStockIssued_ttm")
        stock_repurchased_daily = _cf_daily("_commonStockRepurchased_ttm")
    else:
        ocf_daily = fcf_daily = capex_daily = pd.DataFrame()
        stock_issued_daily = stock_repurchased_daily = pd.DataFrame()

    # --- Value (additional) -----------------------------------------------
    if not mktcap.empty and not sales_daily.empty:
        extras["PSALES"] = sales_daily / mktcap
    if not mktcap.empty and not ocf_daily.empty:
        extras["POCF"] = ocf_daily / mktcap
    if not mktcap.empty and not fcf_daily.empty:
        extras["PFCF"] = fcf_daily / mktcap
    if not ev.empty and not sales_daily.empty:
        extras["EVSALES"] = sales_daily / ev
    if not ev.empty and not ebit_ttm_daily.empty:
        extras["EVEBIT"] = ebit_ttm_daily / ev
    if not ev.empty and not ebitda_ttm_daily.empty:
        extras["EVEBITDA"] = ebitda_ttm_daily / ev
    if not ev.empty and not ocf_daily.empty:
        extras["EVOCF"] = ocf_daily / ev
    if not ev.empty and not fcf_daily.empty:
        extras["EVFCF"] = fcf_daily / ev
    if not mktcap.empty and not equity_daily.empty and not intangibles_daily.empty:
        tangible_equity = equity_daily.sub(intangibles_daily.fillna(0), fill_value=0)
        extras["PTB"] = tangible_equity / mktcap

    # --- Profitability (additional) ---------------------------------------
    if not ni_ttm_daily.empty and not assets_daily.empty:
        extras["ROA"] = ni_ttm_daily / assets_daily.replace(0, np.nan)
    if not gp_ttm_daily.empty and not sales_daily.empty:
        extras["GROSSMARGIN"] = gp_ttm_daily / sales_daily.replace(0, np.nan)
    if not ni_ttm_daily.empty and not cap_invested.empty:
        extras["ROIC"] = ni_ttm_daily / cap_invested
    if not assets_daily.empty and not ebitda_ttm_daily.empty:
        ebitda_plus_rd = ebitda_ttm_daily.add(
            rd_ttm_daily.fillna(0) if not rd_ttm_daily.empty else 0, fill_value=0
        )
        extras["OPRDA"] = ebitda_plus_rd / assets_daily.replace(0, np.nan)
        if not cap_invested.empty:
            extras["OPRDIC"] = ebitda_plus_rd / cap_invested
    if not fcf_daily.empty and not cap_invested.empty:
        extras["FCFROIC"] = fcf_daily / cap_invested
    if not ebit_ttm_daily.empty and not ppe_daily.empty and not nwc.empty:
        # Greenblatt's RoC: EBIT / (Net Fixed Assets + Net Working Capital)
        denom = ppe_daily.fillna(0).add(nwc.fillna(0), fill_value=0).replace(0, np.nan)
        extras["GBROC"] = ebit_ttm_daily / denom

    # --- Leverage (additional) --------------------------------------------
    if not total_debt_daily.empty and not ebitda_ttm_daily.empty:
        extras["DEBITDA"] = total_debt_daily / ebitda_ttm_daily.replace(0, np.nan)
    if not lt_debt_daily.empty and not equity_daily.empty:
        extras["LTDE"] = lt_debt_daily / equity_daily.replace(0, np.nan)
    if not lt_debt_daily.empty and not ebitda_ttm_daily.empty:
        extras["LTDEBITDA"] = lt_debt_daily / ebitda_ttm_daily.replace(0, np.nan)
    if not total_debt_daily.empty and not cash_daily.empty and not ebitda_ttm_daily.empty:
        net_debt = total_debt_daily.fillna(0).sub(cash_daily.fillna(0), fill_value=0)
        extras["NDEBITDA"] = net_debt / ebitda_ttm_daily.replace(0, np.nan)

    # --- Capital Allocation -----------------------------------------------
    if not capex_daily.empty and not ppe_daily.empty:
        extras["CAPEXPPE"] = capex_daily / ppe_daily.replace(0, np.nan)
    # STKP = TTM net stock issuance / MktCap. FMP `commonStockIssued` is gross
    # issuance proceeds (positive); `commonStockRepurchased` is buybacks (negative
    # in cash-flow convention). Net = issued + repurchased.
    if not mktcap.empty and not stock_issued_daily.empty:
        net_iss = stock_issued_daily.fillna(0)
        if not stock_repurchased_daily.empty:
            net_iss = net_iss.add(stock_repurchased_daily.fillna(0), fill_value=0)
        extras["STKP"] = net_iss / mktcap
    # EXTFA = (net equity issuance + net debt issuance) / assets. Skipped when
    # debt-issuance fields aren't both present — FMP exposes `debtIssuance`/
    # `debtRepayment` inconsistently across vintages; we don't fabricate this.

    return {"PE": pe, "PB": pb, "ROE": roe, "GPOA": gpoa, **extras}
