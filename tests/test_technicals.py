"""Tests for the deterministic technical layer (src/research/technicals.py).

No network: every case is built from a synthetic price path.
"""

import numpy as np
import pandas as pd
import pytest

from src.research import technicals as tech


def make_bars(closes, start="2025-01-01", rsi=50.0, volume=1_000_000):
    """Daily OHLCV frame in the shape `normalize_bars` produces."""
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), volume, dtype=float),
            "rsi": np.full(len(closes), rsi, dtype=float),
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


def flat(n=260, level=100.0):
    return [level] * n


# --- normalization ------------------------------------------------------------


def test_normalize_bars_sorts_ascending_and_dedupes():
    raw = [  # FMP returns newest-first
        {"date": "2026-08-14 00:00:00", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 1, "rsi": 55},
        {"date": "2026-08-13 00:00:00", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1, "rsi": 54},
        {"date": "2026-08-13 00:00:00", "open": 9, "high": 9, "low": 9, "close": 9, "volume": 1, "rsi": 54},
    ]
    bars = tech.normalize_bars(raw)
    assert list(bars.index) == [pd.Timestamp("2026-08-13"), pd.Timestamp("2026-08-14")]
    assert bars["close"].iloc[-1] == 3.0
    assert bars["close"].iloc[0] == 9.0  # last duplicate wins


def test_normalize_bars_empty():
    assert tech.normalize_bars([]).empty


# --- indicators ---------------------------------------------------------------


def test_flat_series_has_zero_stretch_and_no_z():
    ind = tech.compute_indicators(make_bars(flat()))
    assert ind["last_close"] == 100.0
    assert ind["px_vs_sma20_pct"] == 0.0
    assert ind["z20"] is None  # zero stdev -> undefined, not inf
    assert ind["ret_1w_pct"] == 0.0


def test_indicators_are_json_safe():
    ind = tech.compute_indicators(make_bars(flat()))
    import json

    json.dumps(ind)  # must not raise on NaN/inf
    assert all(not isinstance(v, float) or np.isfinite(v) for v in ind.values() if v is not None)


def test_52w_range_position():
    closes = list(np.linspace(50, 150, 260))
    ind = tech.compute_indicators(make_bars(closes))
    assert ind["pct_of_52w_range"] == pytest.approx(1.0, abs=0.02)
    # The window is the trailing 252 bars of a 260-bar ramp, so the low is the
    # level ~8 bars in, not the series minimum.
    assert 52 < ind["range_52w_low"] < 54


def test_move_since_signal_measured_from_the_signal_close():
    bars = make_bars(flat(60) + [110.0] * 5)
    as_of = bars.index[-6]
    ind = tech.compute_indicators(bars, as_of=as_of)
    assert ind["move_since_signal_pct"] == pytest.approx(10.0, abs=0.01)


def test_short_history_leaves_long_averages_none_without_crashing():
    ind = tech.compute_indicators(make_bars(flat(30)))
    assert ind["sma20"] is not None
    assert ind["sma50"] is None
    assert ind["sma200"] is None
    verdict, _, _ = tech.classify(ind)
    assert verdict in tech.VERDICT_LABELS


# --- verdicts -----------------------------------------------------------------


def test_insufficient_data():
    verdict, label, reasons = tech.classify(tech.compute_indicators(make_bars(flat(5))))
    assert verdict == "INSUFFICIENT_DATA"
    assert "bars" in reasons[0]


def test_empty_bars_classify_as_insufficient():
    assert tech.compute_indicators(pd.DataFrame())["verdict"] == "INSUFFICIENT_DATA"
    assert tech.compute_indicators(None)["verdict"] == "INSUFFICIENT_DATA"


def test_overbought_pullback_on_a_spike_up():
    bars = make_bars(flat(250) + [102, 104, 107, 110, 114], rsi=78.0)
    ind = tech.compute_indicators(bars)
    verdict, label, reasons = tech.classify(ind)
    assert verdict == "OVERBOUGHT_PULLBACK"
    assert "reversal" in label
    assert any("RSI14" in r for r in reasons)


def test_oversold_bounce_on_a_spike_down():
    bars = make_bars(flat(250) + [98, 96, 93, 90, 86], rsi=22.0)
    verdict, _, reasons = tech.classify(tech.compute_indicators(bars))
    assert verdict == "OVERSOLD_BOUNCE"
    assert any("below SMA20" in r or "RSI14" in r for r in reasons)


def test_rsi_boundary_is_inclusive():
    """70.0 is overbought, 69.9 is not.

    Classified from a hand-built indicator dict so the RSI boundary is isolated:
    on a real price path the z20 arm of the same `or` fires too, and would mask
    the thing under test.
    """
    base = {"n_bars": 260, "z20": 0.4, "ret_1w_pct": 1.0, "trend": "SIDEWAYS"}
    assert tech.classify({**base, "rsi14": 70.0})[0] == "OVERBOUGHT_PULLBACK"
    assert tech.classify({**base, "rsi14": 69.9})[0] == "RANGEBOUND"
    assert tech.classify({**base, "rsi14": 30.0, "ret_1w_pct": -1.0})[0] == "OVERSOLD_BOUNCE"
    assert tech.classify({**base, "rsi14": 30.1, "ret_1w_pct": -1.0})[0] == "RANGEBOUND"


def test_stretch_requires_the_move_to_agree():
    """RSI can be hot while the week was down — that is not a pullback setup,
    so it must not be labelled one."""
    base = {"n_bars": 260, "z20": 0.4, "trend": "SIDEWAYS", "rsi14": 80.0}
    assert tech.classify({**base, "ret_1w_pct": -1.0})[0] == "RANGEBOUND"


def test_stretch_beats_trend():
    """A name can be both trending up and 3 sigma extended; on a one-week
    reversion horizon the stretch is what matters, so it must win."""
    closes = list(np.linspace(80, 120, 255)) + [128, 136, 145, 155, 166]
    ind = tech.compute_indicators(make_bars(closes, rsi=85.0))
    assert ind["trend"] == "UP"
    assert tech.classify(ind)[0] == "OVERBOUGHT_PULLBACK"


def test_uptrend_pullback():
    closes = list(np.linspace(80, 120, 255)) + [119.5, 119, 118.5, 118, 117.5]
    ind = tech.compute_indicators(make_bars(closes, rsi=48.0))
    assert ind["trend"] == "UP"
    assert ind["ret_1w_pct"] < 0
    assert tech.classify(ind)[0] == "UPTREND_PULLBACK"


def test_downtrend_rally():
    closes = list(np.linspace(120, 80, 255)) + [80.5, 81, 81.5, 82, 82.5]
    ind = tech.compute_indicators(make_bars(closes, rsi=45.0))
    assert ind["trend"] == "DOWN"
    assert tech.classify(ind)[0] == "DOWNTREND_RALLY"


def test_steady_trend_is_not_called_rangebound():
    """A name grinding higher sits ~1-1.5 sigma above its own SMA20 the whole
    way up. That is an uptrend, not an absence of one."""
    up = tech.compute_indicators(make_bars(list(np.linspace(80, 120, 260)), rsi=55.0))
    assert up["trend"] == "UP"
    assert 0.5 < up["z20"] < 2.0  # the band that used to fall through
    assert tech.classify(up)[0] == "UPTREND_INTACT"

    down = tech.compute_indicators(make_bars(list(np.linspace(120, 80, 260)), rsi=45.0))
    assert tech.classify(down)[0] == "DOWNTREND_INTACT"


def test_flat_series_is_rangebound():
    """RANGEBOUND means no trend, not merely 'no rule matched'."""
    ind = tech.compute_indicators(make_bars(flat()))
    assert ind["trend"] == "SIDEWAYS"
    assert tech.classify(ind)[0] == "RANGEBOUND"


def test_every_verdict_has_a_label():
    for verdict in tech.VERDICT_LABELS:
        assert tech.VERDICT_LABELS[verdict]


def test_classify_is_deterministic():
    ind = tech.compute_indicators(make_bars(flat(250) + [102, 104, 107, 110, 114], rsi=78.0))
    assert tech.classify(ind) == tech.classify(ind)


# --- flags --------------------------------------------------------------------


def test_at_52w_high_and_low_flags():
    rising = tech.compute_indicators(make_bars(list(np.linspace(50, 150, 260))))
    assert "at_52w_high" in tech.flags(rising)
    falling = tech.compute_indicators(make_bars(list(np.linspace(150, 50, 260))))
    assert "at_52w_low" in tech.flags(falling)


def test_volume_spike_flag():
    bars = make_bars(flat())
    bars.loc[bars.index[-1], "volume"] = 5_000_000
    assert "volume_spike" in tech.flags(tech.compute_indicators(bars))


def test_gapped_since_signal_flag():
    rng = np.random.default_rng(0)
    noisy = 100 + np.cumsum(rng.normal(0, 0.5, 250))
    bars = make_bars(list(noisy) + [float(noisy[-1] * 1.25)] * 3)
    ind = tech.compute_indicators(bars, as_of=bars.index[-4])
    assert "gapped_since_signal" in tech.flags(ind)


def test_no_flags_on_a_quiet_series():
    assert tech.flags(tech.compute_indicators(make_bars(flat()))) == []


# --- alignment ----------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict,long_expected,short_expected",
    [
        ("OVERSOLD_BOUNCE", "aligned", "conflict"),
        ("UPTREND_PULLBACK", "aligned", "conflict"),
        ("OVERBOUGHT_PULLBACK", "conflict", "aligned"),
        ("DOWNTREND_RALLY", "conflict", "aligned"),
        ("UPTREND_INTACT", "neutral", "neutral"),
        ("DOWNTREND_INTACT", "neutral", "neutral"),
        ("RANGEBOUND", "neutral", "neutral"),
        ("INSUFFICIENT_DATA", "neutral", "neutral"),
    ],
)
def test_alignment_matrix(verdict, long_expected, short_expected):
    assert tech.alignment("LONG", verdict) == long_expected
    assert tech.alignment("SHORT", verdict) == short_expected


def test_alignment_is_case_insensitive_and_safe_on_junk():
    assert tech.alignment("long", "OVERSOLD_BOUNCE") == "aligned"
    assert tech.alignment("", "OVERSOLD_BOUNCE") == "neutral"
    assert tech.alignment("LONG", "NOT_A_VERDICT") == "neutral"


def test_build_technicals_bundles_everything():
    bars = make_bars(flat(250) + [102, 104, 107, 110, 114], rsi=78.0)
    out = tech.build_technicals(bars, "LONG", as_of=bars.index[-3])
    assert out["verdict"] == "OVERBOUGHT_PULLBACK"
    # A long on a name making stretched new highs is the model fighting the tape.
    assert out["alignment"] == "conflict"
    assert out["verdict_reasons"] and isinstance(out["flags"], list)
