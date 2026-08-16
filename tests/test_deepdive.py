"""Tests for universe selection and dossier assembly (src/research/deepdive.py).

All offline: the FMP and Tavily fetchers are monkeypatched, so these cover the
extraction and degradation logic rather than the vendors.
"""

import json

import pandas as pd
import pytest

from src.research import deepdive as dd


def make_forecast(n_us=60, n_uk=30, n_ca=20, ineligible=5):
    """A forecast frame shaped like data/processed/forecasts/{DAY}/{date}.parquet."""
    rows = []
    for region, n, suffix in (("US", n_us, ""), ("UK", n_uk, ".L"), ("CA", n_ca, ".TO")):
        for i in range(n):
            rows.append(
                {
                    "ticker": f"{region}{i:03d}{suffix}",
                    "prediction": 1.0 - i / (n + 1),
                    "eligible": True,
                    "region": region,
                    "industry": "Widgets",
                    "commentary": "R1W high (pushes score down)",
                    "as_of_date": pd.Timestamp("2026-08-13"),
                    "target_date": pd.Timestamp("2026-08-20"),
                }
            )
    for i in range(ineligible):
        rows.append(
            {
                "ticker": f"BAD{i}", "prediction": 0.99, "eligible": False, "region": "US",
                "industry": None, "commentary": "",
                "as_of_date": pd.Timestamp("2026-08-13"), "target_date": pd.Timestamp("2026-08-20"),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


# --- selection ----------------------------------------------------------------


def test_selects_top_and_bottom_per_region():
    sel = dd.select_universe(make_forecast(), top_n=5)
    assert len(sel) == 5 * 2 * 3
    us_long = sel[(sel.region == "US") & (sel.side == "LONG")].sort_values("rank")
    assert list(us_long.ticker) == [f"US{i:03d}" for i in range(5)]
    # Rank 1 short is the *weakest* name, i.e. the tail read backwards.
    us_short = sel[(sel.region == "US") & (sel.side == "SHORT")].sort_values("rank")
    assert us_short.ticker.iloc[0] == "US059"


def test_ineligible_names_are_never_selected():
    sel = dd.select_universe(make_forecast(), top_n=5)
    assert not sel.ticker.str.startswith("BAD").any()


def test_longs_and_shorts_never_overlap_on_a_thin_cross_section():
    """CA has ~126 eligible names live; with a small region the basket must
    halve rather than let one ticker be both a long and a short."""
    sel = dd.select_universe(make_forecast(n_ca=6), top_n=20, regions=["CA"])
    assert len(sel) == 6           # k = min(20, 6 // 2) = 3, both sides
    assert sel.ticker.nunique() == 6


def test_single_side_can_use_the_whole_region():
    sel = dd.select_universe(make_forecast(n_ca=6), top_n=5, sides=["long"], regions=["CA"])
    assert len(sel) == 5
    assert set(sel.side) == {"LONG"}


def test_region_with_too_few_names_is_skipped_not_fatal():
    fc = make_forecast(n_us=1, n_uk=0, n_ca=0)
    sel = dd.select_universe(fc, top_n=5, regions=["US", "UK", "CA"])
    assert sel.empty or set(sel.region) <= {"US"}


def test_regions_filter_is_honoured():
    sel = dd.select_universe(make_forecast(), top_n=3, regions=["UK"])
    assert set(sel.region) == {"UK"}
    assert sel.ticker.str.endswith(".L").all()


def test_selection_carries_the_model_context_through():
    sel = dd.select_universe(make_forecast(), top_n=2, regions=["US"])
    assert {"prediction", "industry", "model_commentary"} <= set(sel.columns)
    assert sel.model_commentary.iloc[0].startswith("R1W")


def test_top_n_above_the_cap_is_rejected():
    with pytest.raises(ValueError, match="max_top_n"):
        dd.select_universe(make_forecast(), top_n=25, max_top_n=20)


def test_top_n_below_one_is_rejected():
    with pytest.raises(ValueError):
        dd.select_universe(make_forecast(), top_n=0)


def test_empty_forecast_returns_empty_selection():
    empty = make_forecast(0, 0, 0, ineligible=3)
    assert dd.select_universe(empty, top_n=5).empty


# --- earnings extraction ------------------------------------------------------

AS_OF = pd.Timestamp("2026-08-13")
TARGET = pd.Timestamp("2026-08-20")

EARNINGS = [
    {"date": "2026-11-04", "epsActual": None, "epsEstimated": 1.8,
     "revenueActual": None, "revenueEstimated": 2296304000},
    {"date": "2026-08-12", "epsActual": 1.74, "epsEstimated": 1.62,
     "revenueActual": 2045500000, "revenueEstimated": 1984608000},
    {"date": "2026-05-06", "epsActual": 1.41, "epsEstimated": 1.40,
     "revenueActual": 1_500_000, "revenueEstimated": 1_490_000},
]


def test_last_earnings_is_the_most_recent_reported():
    last, _ = dd._extract_earnings(EARNINGS, AS_OF, TARGET)
    assert last["date"] == "2026-08-12"
    assert last["eps_surprise_pct"] == pytest.approx(7.41, abs=0.01)
    assert last["revenue_surprise_pct"] == pytest.approx(3.07, abs=0.01)


def test_next_earnings_is_the_first_scheduled_after_the_signal():
    _, nxt = dd._extract_earnings(EARNINGS, AS_OF, TARGET)
    assert nxt["date"] == "2026-11-04"
    assert nxt["days_until"] == 83
    assert nxt["in_holding_week"] is False


def test_earnings_inside_the_holding_week_is_flagged():
    rows = EARNINGS + [{"date": "2026-08-18", "epsActual": None, "epsEstimated": 0.5,
                        "revenueActual": None, "revenueEstimated": 100}]
    _, nxt = dd._extract_earnings(rows, AS_OF, TARGET)
    assert nxt["date"] == "2026-08-18"
    assert nxt["in_holding_week"] is True


def test_earnings_on_the_target_date_itself_counts_as_inside():
    rows = [{"date": "2026-08-20", "epsActual": None, "epsEstimated": 1.0,
             "revenueActual": None, "revenueEstimated": 1}]
    _, nxt = dd._extract_earnings(rows, AS_OF, TARGET)
    assert nxt["in_holding_week"] is True


def test_missing_earnings_degrades_to_empty_dicts():
    assert dd._extract_earnings(None, AS_OF, TARGET) == ({}, {})
    assert dd._extract_earnings([], AS_OF, TARGET) == ({}, {})


def test_surprise_is_undefined_against_a_non_positive_estimate():
    """A percentage beat off a zero or negative estimate is meaningless or
    sign-flipped, so it must be None rather than a misleading number."""
    assert dd._surprise_pct(1.0, 0) is None
    assert dd._surprise_pct(1.0, -0.5) is None
    assert dd._surprise_pct(None, 1.0) is None
    assert dd._surprise_pct(1.1, 1.0) == pytest.approx(10.0)


# --- valuation / consensus ----------------------------------------------------


def test_valuation_pulls_roe_from_key_metrics_not_ratios():
    val = dd._extract_valuation(
        {"returnOnEquityTTM": 0.183, "evToEBITDATTM": 12.5},
        {"priceToEarningsRatioTTM": 30.1, "grossProfitMarginTTM": 0.352},
    )
    assert val["roe_ttm"] == pytest.approx(18.3)
    assert val["gross_margin_ttm"] == pytest.approx(35.2)
    assert val["pe_ttm"] == 30.1 and val["ev_ebitda_ttm"] == 12.5


def test_valuation_on_empty_inputs_is_all_none():
    val = dd._extract_valuation(None, None)
    assert set(val.values()) == {None}


def test_consensus_absent_states_a_display_ready_reason():
    """The reason is rendered verbatim by the page, so it must be a finished
    sentence rather than a raw endpoint error."""
    uk = dd._extract_consensus(None, None, 100.0, "KLR.L")
    assert uk["available"] is False
    assert "US listings only" in uk["reason"]
    assert "grades-consensus" not in uk["reason"]
    assert uk["reason"].endswith(".")

    us = dd._extract_consensus(None, None, 100.0, "COHR")
    assert us["available"] is False
    assert "no analyst consensus" in us["reason"].lower()


def test_consensus_present_computes_target_upside():
    cons = dd._extract_consensus(
        {"strongBuy": 1, "buy": 24, "hold": 6, "sell": 0, "strongSell": 0, "consensus": "Buy"},
        {"lastMonthAvgPriceTarget": 427.0, "lastMonthCount": 4},
        324.5, "COHR",
    )
    assert cons["available"] is True and cons["label"] == "Buy"
    assert cons["pt_vs_price_pct"] == pytest.approx(31.6, abs=0.1)


def test_estimates_prefers_a_forward_fiscal_year():
    rows = [{"date": "2020-12-31", "epsAvg": 1.0}, {"date": "2029-12-31", "epsAvg": 3.5,
                                                    "numAnalystsEps": 7}]
    est = dd._extract_estimates(rows)
    assert est["date"] == "2029-12-31" and est["eps_avg"] == 3.5 and est["n_analysts"] == 7


def test_estimates_on_empty_input():
    assert dd._extract_estimates(None) == {}


# --- dossier ------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """Every network fetcher returns 'nothing available'."""
    from src.research import fmp_deep, tavily_client

    def _empty(name):
        return lambda *a, **k: (None, f"{name}: empty")

    for name in ("get_profile", "get_key_metrics_ttm", "get_ratios_ttm", "get_consensus",
                 "get_price_target", "get_earnings", "get_estimates", "get_fmp_news",
                 "get_daily_bars_with_rsi"):
        monkeypatch.setattr(fmp_deep, name, _empty(name))
    monkeypatch.setattr(tavily_client, "search_news", lambda *a, **k: {
        "source": "tavily", "query": "q", "window_days": 30,
        "answer": None, "articles": [], "error": "TAVILY_API_KEY not set",
    })


def test_dossier_survives_every_source_being_empty(offline):
    rec = dd.build_dossier(
        "KLR.L", {"region": "UK", "side": "LONG", "rank": 1}, dd.deepdive_config({}),
        AS_OF, TARGET,
    )
    assert rec["technicals"]["verdict"] == "INSUFFICIENT_DATA"
    assert rec["fundamentals"]["consensus"]["available"] is False
    assert rec["narrative"] is None
    assert len(rec["errors"]) >= 5
    json.dumps(rec)  # the artifact must stay serializable


def test_dossier_records_the_position_it_is_researching(offline):
    meta = {"region": "CA", "side": "SHORT", "rank": 3, "prediction": 0.02}
    rec = dd.build_dossier("TWM.TO", meta, dd.deepdive_config({}), AS_OF, TARGET)
    assert rec["meta"]["side"] == "SHORT" and rec["meta"]["rank"] == 3


def test_dossier_uses_live_sources_when_available(monkeypatch):
    from src.research import fmp_deep, tavily_client
    from tests.test_technicals import make_bars

    bars = make_bars([100.0] * 250 + [102, 104, 107, 110, 114], rsi=78.0)
    monkeypatch.setattr(fmp_deep, "get_profile", lambda *a, **k: (
        {"companyName": "Coherent Corp", "currency": "USD", "sector": "Technology",
         "marketCap": 5e10, "beta": 1.4}, None))
    monkeypatch.setattr(fmp_deep, "get_earnings", lambda *a, **k: (EARNINGS, None))
    monkeypatch.setattr(fmp_deep, "get_estimates", lambda *a, **k: (None, "empty"))
    monkeypatch.setattr(fmp_deep, "get_key_metrics_ttm",
                        lambda *a, **k: ({"evToEBITDATTM": 12.5, "returnOnEquityTTM": 0.18}, None))
    monkeypatch.setattr(fmp_deep, "get_ratios_ttm",
                        lambda *a, **k: ({"priceToEarningsRatioTTM": 30.1}, None))
    monkeypatch.setattr(fmp_deep, "get_consensus", lambda *a, **k: (
        {"strongBuy": 1, "buy": 24, "hold": 6, "sell": 0, "strongSell": 0, "consensus": "Buy"}, None))
    monkeypatch.setattr(fmp_deep, "get_price_target", lambda *a, **k: (None, "empty"))
    monkeypatch.setattr(fmp_deep, "get_fmp_news", lambda *a, **k: ([], None))
    monkeypatch.setattr(fmp_deep, "get_daily_bars_with_rsi", lambda *a, **k: (bars, None))
    monkeypatch.setattr(tavily_client, "search_news", lambda *a, **k: {
        "source": "tavily", "query": "q", "window_days": 30, "answer": "AI optics demand.",
        "articles": [{"title": "Coherent beats", "url": "https://x/a", "published": "2026-08-12",
                      "publisher": "x", "score": 0.8, "snippet": "beat"}],
        "error": None,
    })

    rec = dd.build_dossier("COHR", {"region": "US", "side": "LONG", "rank": 1},
                           dd.deepdive_config({}), AS_OF, TARGET)

    assert rec["meta"]["company_name"] == "Coherent Corp"
    assert rec["fundamentals"]["last_earnings"]["eps_surprise_pct"] == pytest.approx(7.41, abs=0.01)
    assert rec["fundamentals"]["valuation"]["roe_ttm"] == pytest.approx(18.0)
    # A long on a name stretched to new highs is the model fighting the tape.
    assert rec["technicals"]["verdict"] == "OVERBOUGHT_PULLBACK"
    assert rec["technicals"]["alignment"] == "conflict"
    assert rec["news"]["articles"][0]["url"] == "https://x/a"
    assert rec["errors"] == ["get_estimates: empty", "get_price_target: empty"] or rec["errors"]


def test_config_defaults_fill_every_key_the_pipeline_reads():
    cfg = dd.deepdive_config({})
    for key in ("top_n", "max_top_n", "sides", "regions", "news_days", "llm_backend",
                "llm_model", "max_budget_usd", "max_workers", "rsi_period",
                "output_dir", "rate_limit_per_min"):
        assert key in cfg


def test_shipped_config_caps_top_n_at_twenty():
    from src.config import load_config

    cfg = dd.deepdive_config(load_config())
    assert cfg["max_top_n"] == 20
    assert cfg["top_n"] <= cfg["max_top_n"]


def test_artifact_paths_are_namespaced_by_signal_day():
    p = dd.artifact_path("thu", "2026-08-20")
    assert p.name == "2026-08-20.json" and p.parent.name == "THU"
    assert dd.factsheet_path("THU", "2026-08-20").name == "2026-08-20.factsheets.md"
