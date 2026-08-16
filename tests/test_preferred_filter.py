"""Preferred-share exclusion (src/data/universe.py).

The risk this guards against is not missing a preferred — it is silently
dropping legitimate dual-class COMMON shares, which share the `-<LETTER>`
ticker convention and are real universe members.
"""

from __future__ import annotations

import pytest

from src.data.universe import filter_preferred_shares, is_preferred_share

# The 12 preferred lines actually present in the 2026-08-20 forecast universe,
# all verified as ~C$25-par / 5.7-7.6%-yield instruments.
PREFERRED = [
    "AQN-PA.TO", "BBD-PC.TO", "BCE-PY.TO", "BPO-PI.TO", "EMA-PC.TO",
    "EMA-PH.TO", "ENB-PN.TO", "ENB-PT.TO", "PPL-PA.TO", "PPL-PC.TO",
    "TA-PH.TO", "WN-PA.TO",
]

# Must survive: dual-class commons, trust/REIT units, and plain tickers.
COMMON = [
    "BBD-A.TO", "BBD-B.TO",      # Bombardier Class A/B common
    "BF-B",                       # Brown-Forman Class B
    "CTC-A.TO", "GIB-A.TO",       # Canadian Tire / CGI Class A
    "CCL-B.TO", "EMP-A.TO", "QBR-B.TO",
    "BT-A.L",                     # BT Group
    "ACO-X.TO",                   # Atco Class I non-voting
    "BEP-UN.TO", "CAR-UN.TO", "CHP-UN.TO", "CSH-UN.TO", "GRT-UN.TO",  # units
    "AAPL", "MSFT", "ENB.TO", "BCE.TO", "HMSO.L", "TSLA",
]


@pytest.mark.parametrize("sym", PREFERRED)
def test_preferred_detected(sym):
    assert is_preferred_share(sym), f"{sym} should be flagged preferred"


@pytest.mark.parametrize("sym", COMMON)
def test_common_not_flagged(sym):
    assert not is_preferred_share(sym), (
        f"{sym} is common equity / a trust unit and must NOT be filtered out"
    )


def test_filter_removes_only_preferred():
    out = filter_preferred_shares(PREFERRED + COMMON)
    assert out == COMMON, "filter changed order or dropped a legitimate name"
    assert not any(is_preferred_share(s) for s in out)


def test_filter_is_order_preserving_and_total():
    mixed = ["AAPL", "ENB-PN.TO", "BF-B", "BCE-PY.TO", "TSLA"]
    assert filter_preferred_shares(mixed) == ["AAPL", "BF-B", "TSLA"]


def test_handles_whitespace_and_case():
    assert is_preferred_share("  enb-pn.to  ")
    assert not is_preferred_share("  bbd-b.to  ")
