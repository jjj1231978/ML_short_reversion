"""Factor neutralization: winsorization, z-scoring, industry neutralization.

Two APIs:
  * Wide-format helpers (`winsorize`, `zscore`, `iterative_zscore`,
    `neutralize_by_industry`, `full_neutralization`): operate on a single
    factor at a time as a (date × ticker) DataFrame. Kept for backwards
    compatibility and unit tests.
  * Stacked-format pipeline (`neutralize_stacked`): operates on the
    MultiIndex (date, ticker) × factor matrix produced by
    `build_feature_matrix`, applying the full industry-research recipe correctly:
    cross-sectional winsorize per (date, factor) → iterative cross-sectional
    z-score → cap at ±3 → subtract industry median → drop high-missing
    tickers, zero-fill remainder. This is the function `main.py` calls.
"""

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
    """Repeat cross-sectional z-scoring `n_iter` times (industry-research methodology)."""
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
    """Wide-format pipeline: winsorize -> iterative z-score -> industry neutralize.

    Note: `winsorize` here is per-column (per-ticker temporal), not cross-sectional.
    For the paper-correct cross-sectional pipeline use `neutralize_stacked`.
    """
    df = winsorize(df, winsorize_lower, winsorize_upper)
    df = iterative_zscore(df, zscore_iterations)
    df = neutralize_by_industry(df, industry_map)
    return df


# ---------------------------------------------------------------------------
# Stacked-format pipeline (the one main.py calls)
# ---------------------------------------------------------------------------

def neutralize_stacked(
    features: pd.DataFrame,
    industry_map: dict[str, str],
    winsorize_lower: float = 0.02,
    winsorize_upper: float = 0.98,
    zscore_iterations: int = 10,
    zscore_cap: float = 3.0,
    max_missing_factors: int = 10,
    passthrough_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Cross-sectional neutralization on the stacked feature matrix.

    Implements the industry-research recipe (p.13):

      1. Winsorize per (date, factor) at [winsorize_lower, winsorize_upper].
      2. Iterative cross-sectional z-score per (date, factor), `zscore_iterations` times.
      3. Cap z-scores at ±`zscore_cap`.
      4. Subtract median within (date, industry) per factor.
      5. Drop tickers with > `max_missing_factors` missing values that week,
         zero-fill the rest.

    Args:
        features: MultiIndex (date, ticker) × factor columns.
        industry_map: {ticker: GICS industry/sub-industry name}.
        passthrough_cols: columns to exclude from steps 1–4 because they are
            constant across the cross-section per date (broadcast macro
            values, regime flags, HMM posteriors). Z-scoring a constant
            series collapses it to NaN (std=0), so these MUST be passthrough
            or they vanish. They still participate in step 5: their NaN
            count is summed into the missing-factor budget and surviving
            rows are zero-filled.

    Returns:
        DataFrame of the same column shape with row count <= input
        (high-missing rows dropped).
    """
    if features.empty:
        return features.copy()
    if list(features.index.names) != ["date", "ticker"]:
        raise ValueError(
            f"Expected MultiIndex with names ['date', 'ticker']; got {features.index.names}"
        )

    passthrough = set(passthrough_cols or [])
    unknown_pass = passthrough - set(features.columns)
    if unknown_pass:
        # Not fatal — caller may pass a superset that includes columns the
        # builder didn't emit. Just log so it's visible.
        import logging
        logging.getLogger(__name__).warning(
            f"neutralize_stacked: passthrough_cols not in features: {sorted(unknown_pass)}"
        )
    cs_cols = [c for c in features.columns if c not in passthrough]
    pt_cols = [c for c in features.columns if c in passthrough]

    df_cs = features[cs_cols].copy()
    df_pt = features[pt_cols].copy() if pt_cols else None

    # Step 1: cross-sectional winsorize per (date, factor)
    def _winsorize_block(g: pd.DataFrame) -> pd.DataFrame:
        lo = g.quantile(winsorize_lower)
        hi = g.quantile(winsorize_upper)
        return g.clip(lower=lo, upper=hi, axis=1)

    if cs_cols:
        df_cs = df_cs.groupby(level="date", group_keys=False).apply(_winsorize_block)

        # Step 2: iterative cross-sectional z-score per (date, factor)
        for _ in range(zscore_iterations):
            gb = df_cs.groupby(level="date")
            means = gb.transform("mean")
            stds = gb.transform("std").replace(0, np.nan)
            df_cs = (df_cs - means) / stds

        # Step 3: cap at ±zscore_cap
        df_cs = df_cs.clip(lower=-zscore_cap, upper=zscore_cap)

        # Step 4: subtract industry median per (date, industry, factor)
        industries = (
            pd.Series(df_cs.index.get_level_values("ticker"), index=df_cs.index)
            .map(industry_map)
            .fillna("Unknown")
        )
        grouper = [df_cs.index.get_level_values("date"), industries]
        df_cs = df_cs - df_cs.groupby(grouper).transform("median")

    # Recombine: cross-sectional + passthrough side-by-side, original col order
    if df_pt is not None:
        df = pd.concat([df_cs, df_pt.reindex(df_cs.index)], axis=1)[features.columns.tolist()]
    else:
        df = df_cs

    # Step 5: drop high-missing tickers per date, then zero-fill rest
    keep_mask = df.isna().sum(axis=1) <= max_missing_factors
    df = df.loc[keep_mask].fillna(0.0)

    return df
