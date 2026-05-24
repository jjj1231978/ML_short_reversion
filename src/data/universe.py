"""Universe construction: index constituents, sector filters, ADV filters."""

import io

import pandas as pd
import requests


def get_sp500_constituents() -> pd.DataFrame:
    """Fetch current S&P 500 constituents with GICS sector info from Wikipedia.

    Returns DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry.

    NOTE: This gives current membership only. For survivorship-bias-free
    backtesting, use point-in-time constituent lists from a data vendor.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "ML_short_reversion/1.0"})
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 ticker symbols."""
    return get_sp500_constituents()["Symbol"].tolist()


def get_gics_sector_map() -> dict[str, str]:
    """Return {ticker: GICS Sector} mapping for S&P 500.

    Sector names match Wikipedia, e.g. 'Financials', 'Information Technology'.
    """
    df = get_sp500_constituents()
    return dict(zip(df["Symbol"], df["GICS Sector"]))


def filter_by_gics_sector(
    tickers: list[str],
    sector_map: dict[str, str],
    exclude_sectors: list[str],
) -> list[str]:
    """Remove tickers belonging to excluded GICS sectors.

    Args:
        tickers: List of ticker symbols.
        sector_map: {ticker: sector_name} from get_gics_sector_map().
        exclude_sectors: Sector names to exclude, e.g. ["Financials"].
    """
    return [t for t in tickers if sector_map.get(t) not in exclude_sectors]


def filter_by_adv(
    volume_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_adv: float | dict[str, float],
    window: int = 126,
    region_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return boolean mask: True where 6-month rolling ADV >= per-ticker threshold.

    ADV = rolling mean of (close * volume).

    Args:
        volume_df, close_df: (date × ticker) DataFrames.
        min_adv: scalar (legacy single-region) OR dict {region_code: native_threshold}
            for per-region thresholds. UK uses GBp pence; CA uses CAD; US uses USD.
        region_map: {ticker: region_code}. Required when min_adv is a dict.

    Returns:
        Boolean DataFrame (date × ticker), broadcast against per-ticker thresholds.
    """
    dollar_volume = close_df * volume_df
    rolling_adv = dollar_volume.rolling(window=window, min_periods=window // 2).mean()

    if isinstance(min_adv, dict):
        if region_map is None:
            raise ValueError("region_map required when min_adv is a per-region dict")
        # Build per-ticker threshold series (NaN if ticker has no region — those fail filter)
        thresholds = pd.Series(
            {t: min_adv.get(region_map.get(t), float("inf")) for t in close_df.columns},
            dtype=float,
        )
        return rolling_adv.ge(thresholds, axis=1)

    return rolling_adv >= min_adv


# ---------------------------------------------------------------------------
# FMP-backed point-in-time universe
# ---------------------------------------------------------------------------


def get_universe_pit(
    date: str | pd.Timestamp,
    regions: tuple[str, ...] = ("US",),
) -> pd.DataFrame:
    """Return tickers active on `date` across the given regions.

    Reads cached membership parquets written by `src.data.fmp.fetch_index_membership_fmp`.
    For US (SP500), `date_added` and `date_removed` reflect actual historical
    events. For UK (FTSE100) and Canada (TSX60), FMP exposes no historical
    constituent stream — those universes are current-snapshot only and
    survivorship-biased (accepted limitation; see `.specify/001-overview.md`).

    Returns DataFrame with columns: symbol, sector, sub_industry, region, exchange.
    """
    from src.data.fmp import _resolve_cache_dir

    date = pd.Timestamp(date)
    cache_root = _resolve_cache_dir(None) / "constituents"

    region_to_file = {
        "US": "sp500_membership.parquet",
        "UK": "ftse100_current.parquet",
        "CA": "tsx60_current.parquet",
    }

    frames = []
    for region in regions:
        fname = region_to_file.get(region.upper())
        if not fname:
            raise ValueError(f"Unknown region: {region!r}. Use US, UK, or CA.")
        path = cache_root / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Membership cache missing for {region}: {path}. "
                f"Run `python -m src.data backfill --regions {region}` first."
            )
        df = pd.read_parquet(path)
        active = df[
            (df["date_added"].isna() | (df["date_added"] <= date))
            & (df["date_removed"].isna() | (df["date_removed"] > date))
        ]
        frames.append(active[["symbol", "sector", "sub_industry", "region", "exchange"]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "sector", "sub_industry", "region", "exchange"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])


def build_eligibility_mask(
    weekly_close: pd.DataFrame,
    weekly_adv: pd.DataFrame,
    region_map: dict[str, str],
    min_price_by_region: dict[str, float],
    min_adv_by_region: dict[str, float],
    pit_membership_us: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-week eligibility mask combining price floor, ADV floor, and PIT membership.

    Returns a boolean (date × ticker) DataFrame. True means ticker is tradeable
    that week. Predictions for ineligible cells should be masked to NaN before
    quintile assignment.

    For US tickers, `pit_membership_us` is the SP500 history with date_added /
    date_removed; we mask weeks outside [date_added, date_removed). UK/CA
    tickers from snapshot regions are always-active (survivorship-biased).
    """
    # Per-ticker thresholds aligned to columns
    price_thr = pd.Series(
        {t: min_price_by_region.get(region_map.get(t), 0.0) for t in weekly_close.columns},
        dtype=float,
    )
    adv_thr = pd.Series(
        {t: min_adv_by_region.get(region_map.get(t), float("inf")) for t in weekly_close.columns},
        dtype=float,
    )

    price_ok = weekly_close.ge(price_thr, axis=1).fillna(False)
    adv_ok = weekly_adv.ge(adv_thr, axis=1).fillna(False)
    eligible = price_ok & adv_ok

    # PIT membership for US tickers
    if pit_membership_us is not None and not pit_membership_us.empty:
        us_tickers = [t for t, r in region_map.items() if r == "US" and t in eligible.columns]
        if us_tickers:
            pit_mask = pd.DataFrame(False, index=eligible.index, columns=us_tickers)
            for sym, grp in pit_membership_us.groupby("symbol"):
                if sym not in pit_mask.columns:
                    continue
                col = pd.Series(False, index=eligible.index)
                for _, row in grp.iterrows():
                    start = row["date_added"] if pd.notna(row["date_added"]) else eligible.index.min()
                    end = row["date_removed"] if pd.notna(row["date_removed"]) else eligible.index.max() + pd.Timedelta(days=1)
                    col.loc[(eligible.index >= start) & (eligible.index < end)] = True
                pit_mask[sym] = col
            # Apply: eligible AND active in PIT (only for US tickers)
            for t in us_tickers:
                eligible[t] = eligible[t] & pit_mask[t]

    return eligible


def get_region_for_ticker(symbol: str) -> str | None:
    """Return region (US/UK/CA) for a ticker by inspecting cached membership.

    Heuristic: matches the first hit across SP500, FTSE100, TSX60 caches.
    Returns None if not in any cached universe.
    """
    from src.data.fmp import _resolve_cache_dir

    cache_root = _resolve_cache_dir(None) / "constituents"
    for region, fname in [
        ("US", "sp500_membership.parquet"),
        ("UK", "ftse100_current.parquet"),
        ("CA", "tsx60_current.parquet"),
    ]:
        path = cache_root / fname
        if path.exists():
            df = pd.read_parquet(path)
            if symbol in set(df["symbol"].astype(str)):
                return region
    return None
