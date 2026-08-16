"""Unit tests for the LightGBM random-hyperparameter ensemble (200→50 bag)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.train import (
    LGBBag,
    _rank_ic,
    _sample_lgb_params,
    _fit_lgb,
    _predict_lgb,
)


def _panel(n_dates=12, n_names=40, seed=0):
    """Synthetic cross-sectional panel with a real signal: y depends on f0/f1
    so rank-IC is meaningful and some candidates beat others."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-05", periods=n_dates, freq="W-WED")
    idx = pd.MultiIndex.from_product([dates, range(n_names)], names=["date", "ticker"])
    X = pd.DataFrame(
        rng.normal(size=(len(idx), 4)),
        index=idx,
        columns=[f"f{i}" for i in range(4)],
    )
    y = (1.5 * X["f0"] - 0.8 * X["f1"] + 0.3 * rng.normal(size=len(idx)))
    y.name = "target"
    return X, y


def test_sample_lgb_params_respects_ranges_and_types():
    base = {"objective": "regression", "num_leaves": 31}
    space = {
        "num_leaves": [15, 63],
        "learning_rate": [0.01, 0.08],
        "reg_lambda": [0.0, 5.0],
    }
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = _sample_lgb_params(base, space, rng)
        assert p["objective"] == "regression"          # base passed through
        assert 15 <= p["num_leaves"] <= 63
        assert isinstance(p["num_leaves"], int)         # int-rounded
        assert 0.01 <= p["learning_rate"] <= 0.08
        assert not isinstance(p["reg_lambda"], int) or p["reg_lambda"] in (0, 5)
        assert 0.0 <= p["reg_lambda"] <= 5.0


def test_rank_ic_sign_and_perfect_correlation():
    dates = np.repeat(pd.date_range("2022-01-01", periods=3), 10)
    y = np.tile(np.arange(10.0), 3)
    assert _rank_ic(y, y, dates) == pytest.approx(1.0)        # perfect
    assert _rank_ic(-y, y, dates) == pytest.approx(-1.0)      # inverted
    # Too-few-names weeks are skipped → NaN when nothing qualifies.
    assert np.isnan(_rank_ic(np.arange(3.0), np.arange(3.0), [0, 0, 0]))


def test_fit_lgb_returns_bag_of_selected_size_and_predicts():
    X, y = _panel()
    member_cfg = {
        "params": {"objective": "regression", "verbose": -1, "n_estimators": 50},
        "early_stopping_rounds": 10,
        "random_ensemble": {
            "enabled": True,
            "n_candidates": 12,
            "n_select": 4,
            "seed": 7,
            "param_space": {
                "num_leaves": [8, 31],
                "learning_rate": [0.02, 0.1],
                "colsample_bytree": [0.5, 0.9],
            },
        },
    }
    # Use the full panel as both train and val for the unit test (we only check
    # plumbing: bag size, ordering, prediction shape — not generalization).
    model = _fit_lgb(member_cfg, X, y, X, y)
    assert isinstance(model, LGBBag)
    assert len(model.boosters) == 4
    assert len(model.val_ics) == 4
    # Selected boosters are ordered best-IC first.
    assert model.val_ics == sorted(model.val_ics, reverse=True)

    preds = _predict_lgb(model, X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
    # The bagged prediction has real cross-sectional signal vs the target.
    assert _rank_ic(preds, y.to_numpy(), X.index.get_level_values("date")) > 0.3


def test_disabled_random_ensemble_returns_single_model():
    X, y = _panel()
    member_cfg = {
        "params": {"objective": "regression", "verbose": -1, "n_estimators": 50},
        "early_stopping_rounds": 10,
        "random_ensemble": {"enabled": False},
    }
    model = _fit_lgb(member_cfg, X, y, X, y)
    assert not isinstance(model, LGBBag)            # plain Booster
    preds = _predict_lgb(model, X)
    assert preds.shape == (len(X),)
