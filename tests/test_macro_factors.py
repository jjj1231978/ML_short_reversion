"""Tests for the macro/regime additions:
  - macro factor functions (transforms, flags, betas, corrs, broadcast)
  - neutralize_stacked passthrough_cols
  - HMM regime utility determinism and canonical state ordering
  - MLP fit/predict round-trip
"""

import numpy as np
import pandas as pd
import pytest

from src.features import factors
from src.features.neutralize import neutralize_stacked


# -----------------------------------------------------------------------------
# Macro factor primitives
# -----------------------------------------------------------------------------


def _macro_panel(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "VIX": 20 + rng.standard_normal(n).cumsum() * 0.5,
            "OIL": 70 + rng.standard_normal(n).cumsum() * 0.3,
            "DXY": 100 + rng.standard_normal(n).cumsum() * 0.1,
        },
        index=idx,
    )


def _stock_panel(macro_index, n_tickers=5, seed=1):
    rng = np.random.default_rng(seed)
    cols = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(
        rng.standard_normal((len(macro_index), n_tickers)).cumsum(axis=0) + 100,
        index=macro_index,
        columns=cols,
    )


def test_macro_changes_and_returns_shapes():
    m = _macro_panel()
    chg = factors.macro_changes(m, periods=5)
    ret = factors.macro_returns(m, periods=5)
    assert chg.shape == m.shape
    assert ret.shape == m.shape
    # First 5 rows must be NaN (diff/pct_change look-back)
    assert chg.iloc[:5].isna().all().all()
    assert ret.iloc[:5].isna().all().all()


def test_macro_zscore_is_causal():
    """A value at date t must depend only on dates <= t."""
    m = _macro_panel(n=300, seed=0)
    z_full = factors.macro_rolling_zscore(m, window=60)
    # Truncate then z-score; values up to truncation point must match.
    cut = 200
    z_partial = factors.macro_rolling_zscore(m.iloc[:cut], window=60)
    pd.testing.assert_frame_equal(
        z_full.iloc[:cut],
        z_partial,
        check_names=False,
    )


def test_macro_regime_flags_are_zero_or_one():
    m = _macro_panel()
    flags = factors.macro_regime_flags(m, window=60, hi_threshold=1.0)
    # Output: 2 flags per macro
    assert flags.shape[1] == 2 * m.shape[1]
    # Drop NaN warmup rows and check binary
    vals = flags.dropna().values
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_stock_macro_beta_varies_cross_sectionally():
    m = _macro_panel(n=300)
    stocks = _stock_panel(m.index, n_tickers=8)
    b = factors.stock_macro_beta(stocks, m["OIL"], window=60)
    assert b.shape == stocks.shape
    # On at least one date past warmup, betas should differ across tickers.
    row = b.iloc[-1].dropna()
    assert row.std() > 0


def test_broadcast_helper_replicates_across_tickers():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
    out = factors._broadcast_to_tickers(s, ["A", "B", "C"])
    assert out.shape == (3, 3)
    # Each row must be constant across tickers
    for _, row in out.iterrows():
        assert row.nunique() == 1


# -----------------------------------------------------------------------------
# Neutralization passthrough
# -----------------------------------------------------------------------------


def _make_stacked(n_dates=10, n_tickers=20, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factor_cols = [f"F{i}" for i in range(n_factors)]
    df = pd.DataFrame(
        rng.standard_normal((len(idx), n_factors)),
        index=idx,
        columns=factor_cols,
    )
    return df, tickers


def test_passthrough_columns_survive_with_constant_per_date():
    """A broadcast macro column should NOT be z-scored to 0 when passthrough."""
    df, tickers = _make_stacked(n_dates=8, n_tickers=15, n_factors=2)
    # Add a synthetic broadcast column: same value for every ticker per date
    dates = df.index.get_level_values("date")
    df["VIX_HI"] = pd.Series(
        (dates.factorize()[0] % 2).astype(float),  # 0,1,0,1,... per date
        index=df.index,
    )
    industry_map = {t: "Indust" for t in tickers}

    out_no_pass = neutralize_stacked(df.copy(), industry_map=industry_map)
    out_pass = neutralize_stacked(df.copy(), industry_map=industry_map, passthrough_cols=["VIX_HI"])

    # Without passthrough, VIX_HI was z-scored across the constant column → NaN/0
    # After fillna(0) in the pipeline, the column should be all zero
    assert (out_no_pass["VIX_HI"] == 0).all()

    # With passthrough, the original 0/1 values survive (one row's worth at least)
    assert out_pass["VIX_HI"].unique().tolist() in ([0.0, 1.0], [1.0, 0.0])


def test_passthrough_unknown_col_warns_not_raises(caplog):
    df, tickers = _make_stacked()
    industry_map = {t: "Indust" for t in tickers}
    with caplog.at_level("WARNING"):
        out = neutralize_stacked(
            df.copy(),
            industry_map=industry_map,
            passthrough_cols=["DOES_NOT_EXIST"],
        )
    assert "passthrough_cols not in features" in caplog.text
    assert out.shape == df.shape


# -----------------------------------------------------------------------------
# HMM utility — determinism + canonical state order
# -----------------------------------------------------------------------------


def test_hmm_fit_is_deterministic_for_same_seed():
    pytest.importorskip("hmmlearn")
    from src.lib.regime_hmm import fit_regimes

    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=800)
    s = pd.Series(rng.standard_normal(800).cumsum() + 100, index=idx, name="X")
    df = s.to_frame()

    r1 = fit_regimes(df, n_states=2, random_state=42, fit_mode="full")
    r2 = fit_regimes(df, n_states=2, random_state=42, fit_mode="full")
    pd.testing.assert_frame_equal(r1, r2)


def test_hmm_walk_forward_is_strictly_causal():
    """Output at date t must depend only on data observed up to t-1 (where
    the fit happened at the most recent refit). Truncating the input at t and
    rerunning must yield identical posteriors for all rows <= t.
    """
    pytest.importorskip("hmmlearn")
    from src.lib.regime_hmm import fit_regimes

    rng = np.random.default_rng(0)
    n = 600
    obs = rng.standard_normal(n).cumsum() + 100
    idx = pd.bdate_range("2018-01-01", periods=n)
    df = pd.Series(obs, index=idx, name="X").to_frame()

    full = fit_regimes(
        df, n_states=2, fit_mode="walk_forward",
        min_obs=120, refit_every=60, random_state=7,
    )
    # Truncate after row 400 and rerun. Decoded rows in [min_obs, 400] must
    # be identical regardless of whether dates beyond 400 exist or not.
    cut = 400
    truncated = fit_regimes(
        df.iloc[:cut], n_states=2, fit_mode="walk_forward",
        min_obs=120, refit_every=60, random_state=7,
    )
    # Compare on the truncated index. Emission (pct_change) drops the first
    # row, so the output series has length len(input)-1 — using `loc[...]`
    # rather than iloc keeps the alignment honest.
    pd.testing.assert_frame_equal(
        full.loc[truncated.index],
        truncated,
        check_names=False,
    )


def test_hmm_walk_forward_warmup_is_nan():
    """Rows before `min_obs` must be NaN (no causal posterior available)."""
    pytest.importorskip("hmmlearn")
    from src.lib.regime_hmm import fit_regimes

    rng = np.random.default_rng(0)
    n = 500
    obs = rng.standard_normal(n).cumsum() + 100
    idx = pd.bdate_range("2018-01-01", periods=n)
    df = pd.Series(obs, index=idx, name="X").to_frame()

    out = fit_regimes(
        df, n_states=2, fit_mode="walk_forward",
        min_obs=120, refit_every=60, random_state=42,
    )
    # The first `min_obs` rows of the *emitted* series should be NaN.
    # Note: input has one initial NaN from pct_change, so the emitted index
    # has length n - 1. The first min_obs rows of that index are warmup.
    s0 = out["X_HMM_S0_P"]
    assert s0.iloc[:120].isna().all(), "warmup rows must be NaN (no causal posterior)"
    assert s0.iloc[120:].notna().any(), "post-warmup rows must have values"


def test_hmm_state_ordering_canonical():
    """State 0 must have lower posterior-weighted variance than state 1
    after canonicalization, regardless of init seed."""
    pytest.importorskip("hmmlearn")
    from src.lib.regime_hmm import fit_regimes

    rng = np.random.default_rng(0)
    # Mixture of low-vol and high-vol regimes — synthetic, with clear separation
    n = 1000
    high = rng.standard_normal(n // 2) * 3.0
    low = rng.standard_normal(n // 2) * 0.5
    # Interleave so the series has obvious state switches
    series = np.empty(n)
    series[0::2] = low
    series[1::2] = high
    cum = np.cumsum(series) + 100
    idx = pd.bdate_range("2018-01-01", periods=n)
    df = pd.Series(cum, index=idx, name="MIX").to_frame()

    out = fit_regimes(df, n_states=2, random_state=7, fit_mode="full")
    # Posterior-weighted variance of S1 should exceed that of S0
    obs = df["MIX"].pct_change().dropna().values
    p0 = out["MIX_HMM_S0_P"].dropna().values[: len(obs)]
    p1 = out["MIX_HMM_S1_P"].dropna().values[: len(obs)]
    var0 = ((obs - obs.mean()) ** 2 * p0).sum() / max(p0.sum(), 1e-9)
    var1 = ((obs - obs.mean()) ** 2 * p1).sum() / max(p1.sum(), 1e-9)
    assert var1 >= var0


# -----------------------------------------------------------------------------
# MLP — fit + predict round-trip
# -----------------------------------------------------------------------------


def test_mlp_fit_and_predict_round_trip():
    pytest.importorskip("torch")
    from src.model.mlp import MLPConfig, fit_mlp

    rng = np.random.default_rng(0)
    n_train, n_val, n_features = 500, 100, 8
    cols = [f"f{i}" for i in range(n_features)]
    X_train = pd.DataFrame(rng.standard_normal((n_train, n_features)), columns=cols)
    X_val = pd.DataFrame(rng.standard_normal((n_val, n_features)), columns=cols)
    # Make target a linear function so the MLP can actually fit
    w = rng.standard_normal(n_features)
    y_train = pd.Series(X_train.values @ w + 0.1 * rng.standard_normal(n_train))
    y_val = pd.Series(X_val.values @ w + 0.1 * rng.standard_normal(n_val))

    cfg = MLPConfig(
        hidden_dims=(16, 8),
        dropout=0.0,
        batchnorm=False,
        max_epochs=20,
        early_stopping_patience=5,
        batch_size=64,
        device="cpu",
    )
    model = fit_mlp(cfg, X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == (n_val,)
    # Should beat predicting the mean
    baseline = y_val.var()
    mse = float(((preds - y_val.values) ** 2).mean())
    assert mse < baseline, f"MLP MSE {mse} should be below baseline variance {baseline}"


def test_mlp_predict_handles_dataframe_reordered_columns():
    """The wrapper must reindex columns to `feature_names` before scaling."""
    pytest.importorskip("torch")
    from src.model.mlp import MLPConfig, fit_mlp

    rng = np.random.default_rng(1)
    cols = ["a", "b", "c"]
    X_train = pd.DataFrame(rng.standard_normal((200, 3)), columns=cols)
    y_train = pd.Series(X_train["a"] + X_train["b"] * 0.5)
    X_val = X_train.iloc[:30]
    y_val = y_train.iloc[:30]
    cfg = MLPConfig(hidden_dims=(8,), batchnorm=False, max_epochs=10, device="cpu")
    model = fit_mlp(cfg, X_train, y_train, X_val, y_val)

    # Predict with shuffled columns — must still match
    X_shuffled = X_val[["c", "a", "b"]]
    p1 = model.predict(X_val)
    p2 = model.predict(X_shuffled)
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)
