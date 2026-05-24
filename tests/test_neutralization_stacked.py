"""Tests for the stacked-format neutralization pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.neutralize import neutralize_stacked


@pytest.fixture
def sample_features():
    """Synthetic stacked feature matrix: 8 weeks x 30 tickers x 4 factors."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-03", periods=8, freq="W-WED")
    tickers = [f"T{i:02d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n) * 5 + 3,        # mean 3, std 5
            "f2": np.random.randn(n) * 100 - 50,     # mean -50, std 100
            "f3": np.random.randn(n) * 0.01,         # tiny scale
            "f4": np.random.randn(n) * 1.0,          # already standardized
        },
        index=idx,
    )
    # Inject a few outliers
    df.iloc[0, 0] = 1e6
    df.iloc[5, 1] = -1e6
    return df


@pytest.fixture
def industry_map():
    return {f"T{i:02d}": ("ind_A" if i % 3 == 0 else ("ind_B" if i % 3 == 1 else "ind_C"))
            for i in range(30)}


class TestNeutralizeStacked:
    def test_outputs_are_capped(self, sample_features, industry_map):
        out = neutralize_stacked(sample_features, industry_map, zscore_cap=3.0)
        # Allow a small slack because industry-median subtraction happens after the cap
        assert out.abs().max().max() < 4.0

    def test_per_date_means_near_zero(self, sample_features, industry_map):
        out = neutralize_stacked(sample_features, industry_map)
        per_date_mean = out.groupby(level="date").mean().abs().max().max()
        # Industry-median subtraction can shift means slightly off zero
        assert per_date_mean < 0.5

    def test_no_nan_in_output(self, sample_features, industry_map):
        # Inject NaNs in 2 factors for one ticker on one date — should still survive (≤ 10)
        df = sample_features.copy()
        df.iloc[0, :2] = np.nan
        out = neutralize_stacked(df, industry_map, max_missing_factors=10)
        assert not out.isna().any().any()

    def test_drops_rows_with_too_many_missing(self, sample_features, industry_map):
        df = sample_features.copy()
        # Make one ticker's row 4 NaNs (= all 4 factors NaN), exceeds threshold of 1
        df.iloc[3, :] = np.nan
        out = neutralize_stacked(df, industry_map, max_missing_factors=1)
        # That row should have been dropped
        assert len(out) < len(df)

    def test_industry_median_subtraction(self, sample_features, industry_map):
        out = neutralize_stacked(sample_features, industry_map)
        # Industry median per (date, industry, factor) should be ~0
        ticker_idx = out.index.get_level_values("ticker")
        industries = pd.Series(ticker_idx, index=out.index).map(industry_map)
        med = out.groupby([out.index.get_level_values("date"), industries]).median()
        # With small samples the median isn't exactly 0 after the cap, but should be small
        assert med.abs().max().max() < 1.0
