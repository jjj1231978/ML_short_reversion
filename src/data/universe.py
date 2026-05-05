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
    min_adv: float,
    window: int = 126,
) -> pd.DataFrame:
    """Return boolean mask: True where 6-month rolling ADV >= min_adv.

    ADV = rolling mean of (close * volume).
    """
    dollar_volume = close_df * volume_df
    rolling_adv = dollar_volume.rolling(window=window, min_periods=window // 2).mean()
    return rolling_adv >= min_adv
