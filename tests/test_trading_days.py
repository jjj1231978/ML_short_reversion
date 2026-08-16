"""Tests for observed-session detection and holiday fallback.

The scenario throughout is US Thanksgiving 2025 (Thu 27 Nov), a US market
closure that the LSE and TSX trade straight through — the case a single global
calendar cannot represent.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.forecast_picks_log import evaluate_picks
from src.lib.trading_days import resolve_session, sessions, sessions_by_group

US = ["US1", "US2", "US3"]
UK = ["UK1", "UK2", "UK3"]

# Mon 24 Nov .. Fri 28 Nov 2025. Thu 27th is Thanksgiving.
WEEK = pd.to_datetime(
    ["2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27", "2025-11-28"]
)
THANKSGIVING = pd.Timestamp("2025-11-27")


@pytest.fixture
def panel():
    """Daily closes where US names are absent on Thanksgiving but UK names print."""
    df = pd.DataFrame(100.0, index=WEEK, columns=US + UK)
    df.loc[THANKSGIVING, US] = np.nan
    # Give every name a distinct move so returns are not degenerate.
    for i, col in enumerate(df.columns):
        df[col] = df[col] + np.arange(len(WEEK)) * (i + 1)
    df.loc[THANKSGIVING, US] = np.nan
    return df


def test_sessions_flags_a_full_closure(panel):
    """A date where nothing prints is not a session."""
    us_only = panel[US]
    got = sessions(us_only)
    assert THANKSGIVING not in got
    assert len(got) == 4


def test_sessions_by_group_gives_each_region_its_own_calendar(panel):
    """The whole point: US shut, UK open, same date."""
    per_region = sessions_by_group(panel, {t: "US" for t in US} | {t: "UK" for t in UK})
    assert THANKSGIVING not in per_region["US"]
    assert THANKSGIVING in per_region["UK"]


def test_sessions_tolerates_thin_early_coverage():
    """Early history where most names are unlisted must still read as sessions.

    Guards the rolling-median reference: a flat fraction-of-all-columns rule
    would call these closures.
    """
    dates = pd.date_range("2008-01-02", periods=40, freq="B")
    df = pd.DataFrame(np.nan, index=dates, columns=[f"T{i}" for i in range(100)])
    df.iloc[:, :3] = 100.0  # only 3 of 100 names listed, every day
    assert len(sessions(df)) == len(dates)


def test_resolve_session_prefers_the_day_then_falls_back(panel):
    us_days = sessions(panel[US])
    # Thanksgiving resolves back to Wednesday, not forward to Friday.
    assert resolve_session(THANKSGIVING, us_days) == pd.Timestamp("2025-11-26")
    # An open day resolves to itself.
    assert resolve_session(pd.Timestamp("2025-11-25"), us_days) == pd.Timestamp("2025-11-25")


def test_resolve_session_falls_forward_only_when_nothing_precedes():
    """With no prior session in range, the next one is taken."""
    days = pd.DatetimeIndex(["2025-11-28"])
    assert resolve_session(THANKSGIVING, days) == pd.Timestamp("2025-11-28")


def test_resolve_session_returns_none_beyond_the_window():
    days = pd.DatetimeIndex(["2025-12-15"])
    assert resolve_session(THANKSGIVING, days) is None


def _picks(target, tickers, scope="US"):
    return pd.DataFrame(
        {
            "signal_day": "THU",
            "as_of_date": "2025-11-20",
            "target_date": target,
            "scope": scope,
            "side": "LONG",
            "rank": range(1, len(tickers) + 1),
            "ticker": tickers,
            "realized_ret": np.nan,
            "signed_ret": np.nan,
            "correct": pd.NA,
            "evaluable": 0,
        }
    )


def test_us_picks_evaluate_through_a_us_holiday(panel):
    """The regression: US names were unevaluable when the target was a US holiday.

    The target row exists (UK names print), so the old whole-row lookup returned
    all-NaN for US tickers and scored the week evaluable=0.
    """
    close = pd.concat([pd.DataFrame(90.0, index=[pd.Timestamp("2025-11-20")],
                                    columns=panel.columns), panel])
    out = evaluate_picks(_picks("2025-11-27", US), close)
    assert (out["evaluable"] == 1).all()
    assert out["realized_ret"].notna().all()


def test_full_closure_target_still_resolves():
    """When every exchange is shut on the target there is no row for it at all."""
    idx = pd.to_datetime(["2025-11-20", "2025-11-26", "2025-11-28"])
    close = pd.DataFrame({"US1": [90.0, 100.0, 110.0], "US2": [50.0, 55.0, 60.0]}, index=idx)
    out = evaluate_picks(_picks("2025-11-27", ["US1", "US2"]), close)
    assert (out["evaluable"] == 1).all()
    # Priced off Wednesday the 26th, the last session on/before the target.
    assert out.loc[out["ticker"] == "US1", "realized_ret"].iloc[0] == pytest.approx(100 / 90 - 1)


def test_week_not_yet_reached_stays_pending():
    """A target beyond the panel must not be scored early."""
    idx = pd.to_datetime(["2025-11-20", "2025-11-24"])
    close = pd.DataFrame({"US1": [90.0, 95.0]}, index=idx)
    out = evaluate_picks(_picks("2025-11-27", ["US1"]), close)
    assert (out["evaluable"] == 0).all()
    assert out["realized_ret"].isna().all()


def test_repeated_ticker_across_scopes_does_not_fan_out(panel):
    """A name held in both a region and the pooled book must yield one row each."""
    close = pd.concat([pd.DataFrame(90.0, index=[pd.Timestamp("2025-11-20")],
                                    columns=panel.columns), panel])
    picks = pd.concat(
        [_picks("2025-11-27", ["US1"], scope="US"),
         _picks("2025-11-27", ["US1"], scope="ALL")],
        ignore_index=True,
    )
    out = evaluate_picks(picks, close)
    assert len(out) == 2
    assert out["realized_ret"].notna().all()


def test_stale_price_is_not_carried_indefinitely():
    """A name that stopped printing long before the target must not resolve."""
    idx = pd.to_datetime(["2025-11-20"] ) .append(pd.date_range("2025-12-01", periods=10, freq="B"))
    close = pd.DataFrame(100.0, index=idx, columns=["US1", "US2"])
    close.loc[idx[1:], "US2"] = np.nan  # US2 delists right after as-of
    out = evaluate_picks(_picks("2025-12-11", ["US1", "US2"]), close)
    assert out.loc[out["ticker"] == "US1", "evaluable"].iloc[0] == 1
    assert out.loc[out["ticker"] == "US2", "evaluable"].iloc[0] == 0
