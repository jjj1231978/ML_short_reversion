"""Tests for feature computation and neutralization."""

import numpy as np
import pandas as pd
import pytest

from src.features.factors import r1w, rsi, momentum, rolling_volatility
from src.features.neutralize import winsorize, zscore, iterative_zscore


@pytest.fixture
def sample_close():
    """Generate synthetic daily close prices for 10 stocks over 300 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = [f"STOCK_{i}" for i in range(10)]
    prices = pd.DataFrame(
        100 * np.exp(np.random.randn(300, 10).cumsum(axis=0) * 0.01),
        index=dates,
        columns=tickers,
    )
    return prices


class TestFactors:
    def test_r1w_shape(self, sample_close):
        result = r1w(sample_close)
        assert result.shape == sample_close.shape

    def test_r1w_first_values_nan(self, sample_close):
        result = r1w(sample_close)
        assert result.iloc[:5].isna().all().all()

    def test_rsi_range(self, sample_close):
        result = rsi(sample_close, window=14)
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 100).all().all()

    def test_momentum_shape(self, sample_close):
        result = momentum(sample_close, lookback=63, skip=21)
        assert result.shape == sample_close.shape

    def test_rolling_volatility_positive(self, sample_close):
        result = rolling_volatility(sample_close, window=126)
        valid = result.dropna()
        assert (valid >= 0).all().all()


class TestNeutralization:
    def test_winsorize_bounds(self):
        data = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        result = winsorize(data, lower=0.05, upper=0.95)
        assert result["a"].min() >= 4  # ~5th percentile
        assert result["a"].max() <= 95  # ~95th percentile

    def test_zscore_mean_zero(self):
        np.random.seed(0)
        data = pd.DataFrame(np.random.randn(50, 10))
        result = zscore(data)
        # Row means should be ~0
        row_means = result.mean(axis=1)
        assert np.abs(row_means).max() < 1e-10

    def test_zscore_std_one(self):
        np.random.seed(0)
        data = pd.DataFrame(np.random.randn(50, 10))
        result = zscore(data)
        # Row stds should be ~1
        row_stds = result.std(axis=1)
        assert np.abs(row_stds - 1).max() < 1e-10

    def test_iterative_zscore_convergence(self):
        np.random.seed(0)
        data = pd.DataFrame(np.random.randn(50, 10) * 5 + 3)
        result = iterative_zscore(data, n_iter=10)
        # After 10 iterations, should still have mean~0, std~1
        row_means = result.mean(axis=1)
        assert np.abs(row_means).max() < 1e-10
