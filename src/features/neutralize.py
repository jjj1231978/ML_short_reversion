"""Factor neutralization: winsorization, z-scoring, industry neutralization."""

import numpy as np
import pandas as pd
from scipy.stats import mstats


def winsorize(
    df: pd.DataFrame, lower: float = 0.02, upper: float = 0.98
) -> pd.DataFrame:
    """Winsorize each column cross-sectionally at the given percentiles."""
    result = df.copy()
    for col in result.columns:
        vals = result[col].dropna()
        if len(vals) == 0:
            continue
        lo, hi = np.nanpercentile(vals, [lower * 100, upper * 100])
        result[col] = result[col].clip(lo, hi)
    return result


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x - mean) / std for each row."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def iterative_zscore(df: pd.DataFrame, n_iter: int = 10) -> pd.DataFrame:
    """Repeat cross-sectional z-scoring `n_iter` times (SG methodology)."""
    result = df.copy()
    for _ in range(n_iter):
        result = zscore(result)
    return result


def neutralize_by_industry(
    df: pd.DataFrame, industry_map: dict[str, str]
) -> pd.DataFrame:
    """Subtract the median of each stock's industry group.

    Args:
        df: Factor values, columns = tickers.
        industry_map: {ticker: gics_industry_name}.
    """
    result = df.copy()
    # Build industry groups from column names
    groups: dict[str, list[str]] = {}
    for ticker in result.columns:
        ind = industry_map.get(ticker, "Unknown")
        groups.setdefault(ind, []).append(ticker)

    for _, tickers in groups.items():
        cols = [t for t in tickers if t in result.columns]
        if len(cols) > 1:
            industry_median = result[cols].median(axis=1)
            result[cols] = result[cols].sub(industry_median, axis=0)

    return result


def full_neutralization(
    df: pd.DataFrame,
    industry_map: dict[str, str],
    winsorize_lower: float = 0.02,
    winsorize_upper: float = 0.98,
    zscore_iterations: int = 10,
) -> pd.DataFrame:
    """Complete neutralization pipeline: winsorize -> iterative z-score -> industry neutralize."""
    df = winsorize(df, winsorize_lower, winsorize_upper)
    df = iterative_zscore(df, zscore_iterations)
    df = neutralize_by_industry(df, industry_map)
    return df
