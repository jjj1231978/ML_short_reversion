"""Macro/market-state time series fetched via FMP.

Pure data layer — fetches and caches; no transforms. Cached parquets land in
the same FMP cache tree used by prices and benchmarks
(`~/data_lake/fmp/prices/by_symbol/{SYMBOL}.parquet`), since the FMP
`/stable/historical-price-eod/full` endpoint serves indexes, commodities, FX
and yield-indexes from the same path as equities. Cache reuse means no new
auth, no new throttle, no new layout — and a single ticker's series is shared
between this module and any other consumer (e.g. benchmark fetch).

Feature transforms (returns, deltas, rolling z-scores, regime flags, rolling
betas of stocks vs macros) live in `src/features/factors.py` and consume
this module's output.

FMP symbol conventions worth knowing:
    Yields (CBOE indexes; values are yield × 10, e.g. ^TNX=42.5 → 4.25%):
        ^IRX = 13W bill, ^FVX = 5Y, ^TNX = 10Y, ^TYX = 30Y.
        Scale-invariant for downstream change/z-score/beta factors.
    Commodities:
        CLUSD = WTI crude, BZUSD = Brent, GCUSD = gold, SIUSD = silver,
        HGUSD = copper, NGUSD = natural gas.
    FX:
        EURUSD, USDJPY, GBPUSD, USDCAD, AUDUSD. DXY is `DX-Y.NYB`.
    Indexes:
        ^GSPC, ^NDX, ^DJI, ^RUT, ^VIX, ^FTSE, ^GSPTSE, ^N225.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# Reasonable defaults across the four families. Override via configs/default.yaml
# under `macro.symbols: {NAME: FMP_SYMBOL}` if a symbol isn't on your FMP plan.
#
# Symbol routing (see fetch_macro_data):
#   - "treasury:<tenor>" → FMP /stable/treasury-rates curve column (on Premium).
#   - anything else      → equity EOD endpoint /stable/historical-price-eod/full.
#
# FMP's dedicated index/commodity namespaces (^NDX, CLUSD, HGUSD, DX-Y.NYB, and
# the CBOE yield indexes ^TNX/^IRX/^TYX) are gated above Premium and 402 on the
# equity endpoint. We substitute liquid ETF proxies that the equity endpoint
# DOES serve, and pull real yields from treasury-rates. Proxies track their
# underlyings well on weekly returns (USO/UUP carry mild roll/level drift —
# fine for return/z-score/beta regime factors, not for absolute levels).
DEFAULT_MACROS: dict[str, str] = {
    # Stock indexes (^GSPC/^RUT/^VIX are served on the equity endpoint; ^NDX is
    # gated → Nasdaq-100 ETF QQQ proxy).
    "SPX": "^GSPC",
    "NDX": "QQQ",
    "RUT": "^RUT",
    "VIX": "^VIX",
    # Treasury par yields (real %, full curve) — see fetch_treasury_rates_fmp.
    "US10Y": "treasury:year10",
    "US30Y": "treasury:year30",
    "US3M": "treasury:month3",
    # Commodities (GCUSD gold is served; WTI/copper gated → USO/CPER ETF proxies).
    "OIL": "USO",
    "GOLD": "GCUSD",
    "COPPER": "CPER",
    # FX (EURUSD/USDJPY served directly; DXY index gated → UUP dollar-ETF proxy).
    "DXY": "UUP",
    "EURUSD": "EURUSD",
    "USDJPY": "USDJPY",
}

# Prefix marking a symbol as a treasury-rates curve tenor rather than an EOD ticker.
_TREASURY_PREFIX = "treasury:"


def fetch_macro_data(
    symbols: dict[str, str] | None = None,
    start: str = "2006-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    rate_limit_per_min: int = 240,
) -> pd.DataFrame:
    """Fetch macro/market-state series via FMP, one symbol per logical name.

    Each (name, fmp_symbol) lands in the shared FMP per-symbol price cache —
    same incremental refresh semantics as equities and benchmarks (trailing
    7-day re-pull, older rows immutable, atomic rename). Failed symbols are
    skipped with a warning rather than aborting the run; FMP plan gating
    varies per ticker (^TNX, ^IRX, DX-Y.NYB are commonly restricted).

    Args:
        symbols: {macro_name: fmp_symbol}. Falls back to `DEFAULT_MACROS`.
        start, end: ISO 8601 date strings. `end=None` → today.
        cache_dir: override for the FMP cache root.
        api_key: FMP API key. Falls back to FMP_API_KEY env var.
        rate_limit_per_min: passed through to the FMP fetcher.

    Returns:
        Wide DataFrame indexed by tz-naive midnight-normalized date, one
        column per `macro_name`, forward-filled across business-day gaps
        (macro state is persistent across weekends/holidays).
    """
    from src.data.fmp import fetch_market_close_fmp, fetch_treasury_rates_fmp

    if symbols is None:
        symbols = dict(DEFAULT_MACROS)
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Split treasury-curve tenors (one shared endpoint call) from per-symbol EOD.
    tsy_syms = {n: s[len(_TREASURY_PREFIX):] for n, s in symbols.items() if s.startswith(_TREASURY_PREFIX)}
    eod_syms = {n: s for n, s in symbols.items() if not s.startswith(_TREASURY_PREFIX)}

    def _normalize_index(idx: pd.Index) -> pd.Index:
        idx = pd.to_datetime(idx)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        return idx.normalize()

    series: dict[str, pd.Series] = {}
    failed: list[tuple[str, str]] = []
    for name, sym in eod_syms.items():
        try:
            s = fetch_market_close_fmp(
                sym, start, end,
                cache_dir=cache_dir,
                api_key=api_key,
                rate_limit_per_min=rate_limit_per_min,
            )
            s.index = _normalize_index(s.index)
            s.name = name  # label by logical name, not FMP symbol
            series[name] = s
        except Exception as e:
            log.warning(f"Macro fetch failed for {name}={sym!r}: {type(e).__name__}: {e}")
            failed.append((name, sym))

    # Treasury par yields: one curve fetch, then slice the requested tenors.
    if tsy_syms:
        try:
            curve = fetch_treasury_rates_fmp(start, end, cache_dir=cache_dir, api_key=api_key)
            curve.index = _normalize_index(curve.index)
            for name, tenor in tsy_syms.items():
                if tenor in curve.columns:
                    s = curve[tenor].dropna()
                    s.name = name
                    series[name] = s
                else:
                    log.warning(f"Macro fetch failed for {name}: tenor {tenor!r} not in treasury curve {list(curve.columns)}")
                    failed.append((name, f"{_TREASURY_PREFIX}{tenor}"))
        except Exception as e:
            log.warning(f"Treasury-rates fetch failed: {type(e).__name__}: {e}")
            failed.extend((n, f"{_TREASURY_PREFIX}{t}") for n, t in tsy_syms.items())

    if not series:
        raise RuntimeError(f"All macro fetches failed. Tried: {list(symbols.items())}")

    wide = pd.concat(series, axis=1).sort_index().ffill()

    if failed:
        log.warning(
            f"Macro data: got {len(series)} of {len(symbols)} series; "
            f"missing: {[n for n, _ in failed]}"
        )
    else:
        log.info(f"Macro data: fetched {len(series)} series, {wide.shape[0]} dates")

    return wide


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import load_config

    cfg = load_config()
    syms = cfg.get("macro", {}).get("symbols") or DEFAULT_MACROS
    df = fetch_macro_data(
        symbols=syms,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
    )
    print(df.tail())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nCoverage:")
    for c in df.columns:
        print(f"  {c}: {df[c].notna().sum()} non-NaN ({df[c].first_valid_index()} → {df[c].last_valid_index()})")
