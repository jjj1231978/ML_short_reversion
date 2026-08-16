"""FMP endpoint wrappers for the per-name deep dive, with a TTL disk cache.

Builds on the request primitives already in `src/data/fmp.py` (`_fmp_url` +
`_fetch_with_429_retry` + `_throttle_sleep` are the whole client) rather than
hand-rolling another HTTP path.

Deliberately does NOT use `fmp.py:_merge_and_write`. That merge is
existing-rows-win by design, to preserve point-in-time vintages for the factor
data. Everything here is mutable — an `epsActual` lands days after the calendar
row, consensus shifts weekly — so it uses a plain TTL cache instead: refetch
once the cached copy is older than the endpoint's TTL.

Regional coverage, verified against the live API (2026-08-16):

    endpoint                          US    UK .L   CA .TO
    profile                           yes   yes     yes
    earnings                          yes   yes     yes
    analyst-estimates                 yes   yes     yes
    key-metrics-ttm / ratios-ttm      yes   yes     yes
    technical-indicators/rsi          yes   yes     yes
    news/stock                        yes   EMPTY   EMPTY
    grades-consensus                  yes   EMPTY   EMPTY
    price-target-summary              yes   EMPTY   EMPTY

An empty response for a non-US listing is therefore expected, not a bug, and is
reported as such rather than as a fetch failure. Note `earnings-calendar`
silently ignores its `symbol` parameter — use `earnings` instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.fmp import (
    _fetch_with_429_retry,
    _fmp_url,
    _resolve_api_key,
    _resolve_cache_dir,
    _safe_filename,
    _throttle_sleep,
)

log = logging.getLogger(__name__)

# How long a cached response stays fresh, per endpoint. Static company facts sit
# for a month; anything that moves with the tape is refetched daily.
DEFAULT_TTL_DAYS = {
    "profile": 30.0,
    "key-metrics-ttm": 7.0,
    "ratios-ttm": 7.0,
    "analyst-estimates": 7.0,
    "grades-consensus": 7.0,
    "price-target-summary": 7.0,
    "earnings": 1.0,
    "news/stock": 1.0,
    "technical-indicators/rsi": 0.0,   # 0 = same-day only
}

# Endpoints FMP only populates for US listings. Hitting them for .L/.TO names
# burns a call to get `[]` back, so skip and report the reason instead.
US_ONLY = ("news/stock", "grades-consensus", "price-target-summary")

_NON_US_REASON = "FMP does not publish this endpoint for non-US listings"


class CallStats:
    """Mutable counters for one batch run (fetches, cache hits, failures)."""

    def __init__(self) -> None:
        self.fetched = 0
        self.cache_hits = 0
        self.skipped_non_us = 0
        self.failures = 0

    def as_dict(self) -> dict:
        return {
            "n_fmp_calls": self.fetched,
            "n_cache_hits": self.cache_hits,
            "n_skipped_non_us": self.skipped_non_us,
            "n_failures": self.failures,
        }


def is_us_listing(ticker: str) -> bool:
    """US names carry no exchange suffix; UK is `.L`, Canada `.TO`."""
    return "." not in (ticker or "")


def _deepdive_cache_dir(cache_dir: Path | None = None) -> Path:
    return _resolve_cache_dir(cache_dir) / "deepdive"


def _cache_path(path: str, params: dict, cache_dir: Path | None) -> Path:
    """One file per (endpoint, symbol, non-symbol params)."""
    slug = path.strip("/").replace("/", "_")
    symbol = params.get("symbol") or params.get("symbols") or "_all"
    extra = {k: v for k, v in sorted(params.items()) if k not in ("symbol", "symbols")}
    suffix = ""
    if extra:
        digest = hashlib.md5(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
        suffix = f"__{digest}"
    return _deepdive_cache_dir(cache_dir) / slug / f"{_safe_filename(str(symbol))}{suffix}.json"


def _read_cache(fp: Path, ttl_days: float) -> object | None:
    """Return the cached payload if it is still within TTL, else None.

    ttl_days == 0 means "same calendar day", which is what price bars want: a
    run this afternoon should reuse this morning's bars but never yesterday's.
    """
    if not fp.exists():
        return None
    try:
        blob = json.loads(fp.read_text())
        fetched = datetime.fromisoformat(blob["fetched_at"])
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    fresh = (
        fetched.date() == now.date()
        if ttl_days == 0
        else now - fetched < timedelta(days=ttl_days)
    )
    return blob.get("payload") if fresh else None


def _write_cache(fp: Path, payload: object) -> None:
    """Atomic write so an interrupted run never leaves a truncated cache file."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "payload": payload})
    )
    tmp.replace(fp)


def fetch_json(
    path: str,
    *,
    ttl_days: float | None = None,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    rate_limit_per_min: int = 280,
    force: bool = False,
    stats: CallStats | None = None,
    **params,
) -> tuple[object | None, str | None]:
    """GET an FMP stable endpoint through the TTL cache.

    Returns `(payload, error)`. Never raises: a transport failure, an empty
    response and a US-only endpoint on a UK listing are all reported as an
    error string so the caller can record it on the dossier and carry on.
    """
    symbol = params.get("symbol") or params.get("symbols") or ""
    if path in US_ONLY and symbol and not is_us_listing(str(symbol)):
        if stats:
            stats.skipped_non_us += 1
        return None, f"{path}: {_NON_US_REASON}"

    if ttl_days is None:
        ttl_days = DEFAULT_TTL_DAYS.get(path, 1.0)
    fp = _cache_path(path, params, cache_dir)

    if not force:
        cached = _read_cache(fp, ttl_days)
        if cached is not None:
            if stats:
                stats.cache_hits += 1
            return cached, None

    try:
        key = _resolve_api_key(api_key)
        payload = _fetch_with_429_retry(_fmp_url(path, key, **params))
        time.sleep(_throttle_sleep(rate_limit_per_min))
    except Exception as exc:  # noqa: BLE001 — one bad endpoint must not kill the run
        if stats:
            stats.failures += 1
        log.warning(f"{path} {symbol}: {exc}")
        return None, f"{path}: {type(exc).__name__}: {exc}"

    if stats:
        stats.fetched += 1
    _write_cache(fp, payload)

    if payload in ([], {}, None):
        reason = _NON_US_REASON if not is_us_listing(str(symbol)) else "no data returned"
        return None, f"{path}: empty ({reason})"
    return payload, None


def _first(payload: object) -> dict | None:
    """FMP returns single-record endpoints as a one-element list."""
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload if isinstance(payload, dict) else None


# --- per-endpoint wrappers ----------------------------------------------------
# Each returns (value, error). `error` is non-None whenever `value` is None.


def get_profile(symbol: str, **kw) -> tuple[dict | None, str | None]:
    payload, err = fetch_json("profile", symbol=symbol, **kw)
    return _first(payload), err


def get_earnings(symbol: str, limit: int = 8, **kw) -> tuple[list | None, str | None]:
    """Past and upcoming rows in one response — this is why `earnings` is used
    rather than `earnings-calendar`, whose symbol filter is silently ignored."""
    payload, err = fetch_json("earnings", symbol=symbol, limit=limit, **kw)
    return (payload if isinstance(payload, list) else None), err


def get_estimates(symbol: str, period: str = "annual", limit: int = 4, **kw):
    payload, err = fetch_json(
        "analyst-estimates", symbol=symbol, period=period, limit=limit, **kw
    )
    return (payload if isinstance(payload, list) else None), err


def get_key_metrics_ttm(symbol: str, **kw) -> tuple[dict | None, str | None]:
    payload, err = fetch_json("key-metrics-ttm", symbol=symbol, **kw)
    return _first(payload), err


def get_ratios_ttm(symbol: str, **kw) -> tuple[dict | None, str | None]:
    payload, err = fetch_json("ratios-ttm", symbol=symbol, **kw)
    return _first(payload), err


def get_consensus(symbol: str, **kw) -> tuple[dict | None, str | None]:
    payload, err = fetch_json("grades-consensus", symbol=symbol, **kw)
    return _first(payload), err


def get_price_target(symbol: str, **kw) -> tuple[dict | None, str | None]:
    payload, err = fetch_json("price-target-summary", symbol=symbol, **kw)
    return _first(payload), err


def get_fmp_news(symbol: str, limit: int = 20, **kw) -> tuple[list | None, str | None]:
    payload, err = fetch_json("news/stock", symbols=symbol, limit=limit, **kw)
    return (payload if isinstance(payload, list) else None), err


def get_daily_bars_with_rsi(
    symbol: str, lookback_days: int = 420, rsi_period: int = 14, asof: str | None = None, **kw
) -> tuple[pd.DataFrame | None, str | None]:
    """~290 daily OHLCV bars *and* RSI in a single call.

    The whole technical layer rides on this one request: SMA20/50/200, ATR, the
    52-week range and the trailing returns are all derived locally from these
    bars. 420 calendar days gives ~290 trading days — enough headroom for SMA200
    plus a full 52-week window. The shared price cache under
    `~/data_lake/fmp/prices/` is not used here because it only advances when the
    backtest runs and lags the live forecast by weeks.
    """
    end = pd.Timestamp(asof) if asof else pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    payload, err = fetch_json(
        "technical-indicators/rsi",
        symbol=symbol,
        periodLength=rsi_period,
        timeframe="1day",
        **{"from": start.date().isoformat(), "to": end.date().isoformat()},
        **kw,
    )
    if not isinstance(payload, list) or not payload:
        return None, err or "technical-indicators/rsi: empty"

    from src.research.technicals import normalize_bars

    bars = normalize_bars(payload)
    return (bars if not bars.empty else None), (None if not bars.empty else "bars: unparseable")
