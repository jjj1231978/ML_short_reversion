"""Tests for portfolio construction and backtesting."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import assign_quintiles, compute_performance_metrics


@pytest.fixture
def sample_predictions():
    """Synthetic predictions with (date, ticker) MultiIndex."""
    dates = pd.date_range("2020-01-01", periods=10, freq="W-WED")
    tickers = [f"STOCK_{i}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    np.random.seed(42)
    return pd.Series(np.random.randn(len(idx)), index=idx, name="prediction")


class TestQuintiles:
    def test_quintile_values(self, sample_predictions):
        q = assign_quintiles(sample_predictions)
        assert set(q.dropna().unique()).issubset({1, 2, 3, 4, 5})

    def test_quintile_balanced(self, sample_predictions):
        q = assign_quintiles(sample_predictions)
        # Each week should have roughly equal quintile sizes
        for date in sample_predictions.index.get_level_values("date").unique():
            counts = q.loc[date].value_counts()
            assert counts.max() - counts.min() <= 2  # allow small imbalance


class TestMetrics:
    def test_metrics_keys(self):
        portfolio = pd.DataFrame({
            "ls_return_net": np.random.randn(100) * 0.01,
            "turnover": np.random.uniform(0.1, 0.5, 100),
        })
        metrics = compute_performance_metrics(portfolio)
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "information_ratio",
            "max_drawdown",
            "avg_weekly_turnover",
            "num_weeks",
        }
        assert set(metrics.keys()) == expected_keys

    def test_max_drawdown_negative(self):
        portfolio = pd.DataFrame({
            "ls_return_net": [-0.01] * 50 + [0.01] * 50,
            "turnover": [0.2] * 100,
        })
        metrics = compute_performance_metrics(portfolio)
        assert metrics["max_drawdown"] < 0
