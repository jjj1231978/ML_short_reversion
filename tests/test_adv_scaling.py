"""Tests for ADV-based position scaling in the portfolio module."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import _basket_weights, build_long_short_portfolio


def test_basket_weights_equal_when_no_adv():
    w = _basket_weights({"A", "B", "C"}, adv_today=None, adv_threshold=20e6)
    assert pytest.approx(w.sum(), abs=1e-9) == 1.0
    assert (w == 1 / 3).all()


def test_basket_weights_adv_scaled():
    # A and B above threshold (full weight 1.0); C and D below
    adv = pd.Series({"A": 100e6, "B": 50e6, "C": 5e6, "D": 2e6})
    w = _basket_weights({"A", "B", "C", "D"}, adv, adv_threshold=20e6)
    assert pytest.approx(w.sum(), abs=1e-9) == 1.0
    # C/A weight ratio = 0.25 / 1.0 = 0.25
    assert pytest.approx(w["C"] / w["A"], rel=1e-6) == 0.25
    # D/A weight ratio = 0.10 / 1.0
    assert pytest.approx(w["D"] / w["A"], rel=1e-6) == 0.10


def test_basket_weights_handles_zero_adv():
    adv = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    w = _basket_weights({"A", "B", "C"}, adv, adv_threshold=20e6)
    # All zero → fallback to equal-weight
    assert pytest.approx(w.sum(), abs=1e-9) == 1.0


def test_portfolio_differs_with_adv_scaling():
    """End-to-end: ADV-scaled portfolio should differ from equal-weight."""
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=8, freq="W-WED")
    tickers = [f"T{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    preds = pd.Series(np.random.randn(len(idx)), index=idx, name="prediction")
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.02, index=dates, columns=tickers
    )
    adv = pd.DataFrame(
        np.random.uniform(1e6, 100e6, (len(dates), len(tickers))),
        index=dates, columns=tickers,
    )
    cfg = {
        "backtest": {
            "trading_cost_bps": 1.5, "execution_lag_days": 1,
            "long_quintile": 5, "short_quintile": 1,
            "weighting": "equal", "rebalance_day": "Wednesday",
        },
        "universe": {"adv_scaling_threshold": 20_000_000},
    }
    p_eq = build_long_short_portfolio(preds, returns, adv=None, cfg=cfg)
    p_adv = build_long_short_portfolio(preds, returns, adv=adv, cfg=cfg)
    assert not np.allclose(
        p_eq["ls_return"].fillna(0).values,
        p_adv["ls_return"].fillna(0).values,
    )
