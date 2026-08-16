"""Orchestrator: forecast picks -> one JSON dossier per (signal day, target date).

Everything the page renders comes out of `build_artifact`. Per name that is 6
FMP calls (9 for US, which additionally has vendor news, consensus and price
targets) plus one Tavily search; the narrative layer is applied afterwards, by
whichever backend `--llm-backend` selects.

Nothing here raises on a missing datum. FMP genuinely does not publish some of
these fields for LSE and TSX listings, so a dossier records what it could not
get in `errors` and carries on — a half-populated name is still worth showing,
and the fact sheet tells the narrative layer exactly what was unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, load_config
from src.research import fmp_deep, tavily_client
from src.research.technicals import build_technicals

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "deepdive"
FORECASTS_ROOT = PROCESSED_DIR / "forecasts"


def deepdive_config(cfg: dict | None = None) -> dict:
    """The `deepdive:` block, with defaults for every key the pipeline reads."""
    cfg = cfg if cfg is not None else load_config()
    dd = dict(cfg.get("deepdive") or {})
    dd.setdefault("signal_day", "THU")
    dd.setdefault("top_n", 20)
    dd.setdefault("max_top_n", 20)
    dd.setdefault("sides", ["long", "short"])
    dd.setdefault("regions", ["US", "UK", "CA"])
    dd.setdefault("news_days", 30)
    dd.setdefault("news_max_results", 6)
    dd.setdefault("llm_backend", "session")
    dd.setdefault("llm_model", "claude-haiku-4-5")
    dd.setdefault("max_budget_usd", 1.00)
    dd.setdefault("max_workers", 4)
    dd.setdefault("rsi_period", 14)
    dd.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))
    dd.setdefault("rate_limit_per_min", (cfg.get("data") or {}).get("fmp_rate_limit_per_min", 280))
    return dd


# --- artifact I/O -------------------------------------------------------------


def artifact_path(signal_day: str, target_date: str, output_dir: str | Path | None = None) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    return base / signal_day.upper() / f"{target_date}.json"


def factsheet_path(signal_day: str, target_date: str, output_dir: str | Path | None = None) -> Path:
    return artifact_path(signal_day, target_date, output_dir).with_suffix(".factsheets.md")


def save_artifact(artifact: dict, path: Path) -> Path:
    """Atomic write — an interrupted run must not leave a half-written artifact
    that the page would then try to render."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=1, sort_keys=False, default=str))
    tmp.replace(path)
    return path


def load_artifact(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        log.error(f"unreadable artifact {path}: {exc}")
        return None


def list_forecasts(signal_day: str) -> list[Path]:
    d = FORECASTS_ROOT / signal_day.upper()
    return sorted(d.glob("*.parquet"), reverse=True) if d.exists() else []


# --- universe selection -------------------------------------------------------


def select_universe(
    fc: pd.DataFrame,
    top_n: int,
    sides: list[str] | tuple[str, ...] = ("long", "short"),
    regions: list[str] | tuple[str, ...] = ("US", "UK", "CA"),
    max_top_n: int = 20,
) -> pd.DataFrame:
    """The names to research: top/bottom `top_n` per region, per side.

    `k = min(top_n, len(region) // 2)` mirrors the guard in
    `src/backtest/forecast_picks_log.py` so that on a thin cross-section — CA
    has only ~126 eligible names — the long and short baskets can never claim
    the same ticker.
    """
    if top_n > max_top_n:
        raise ValueError(f"top_n={top_n} exceeds deepdive.max_top_n={max_top_n}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    sides = [s.upper() for s in sides]
    eligible = fc[fc["eligible"]].copy() if "eligible" in fc.columns else fc.copy()
    rows = []

    for region in regions:
        grp = eligible[eligible["region"] == region].sort_values("prediction", ascending=False)
        if grp.empty:
            continue
        k = min(top_n, len(grp) // 2) if len(sides) > 1 else min(top_n, len(grp))
        if k < 1:
            log.warning(f"{region}: only {len(grp)} eligible names — skipped")
            continue
        if "LONG" in sides:
            rows += [(t, region, "LONG", i + 1) for i, t in enumerate(grp.head(k).index)]
        if "SHORT" in sides:
            rows += [(t, region, "SHORT", i + 1) for i, t in enumerate(grp.tail(k).index[::-1])]

    if not rows:
        return pd.DataFrame(columns=["ticker", "region", "side", "rank"])

    sel = pd.DataFrame(rows, columns=["ticker", "region", "side", "rank"])
    # `target_trade_date` carries the per-region holiday resolution: a shut
    # Thursday moves that region's close to Wednesday, which shifts where the
    # holding week actually ends. Absent on forecasts written before that
    # existed, in which case the nominal target stands.
    extra = [c for c in ("prediction", "industry", "commentary", "target_trade_date")
             if c in fc.columns]
    return sel.join(fc[extra], on="ticker").rename(columns={"commentary": "model_commentary"})


# --- field extraction ---------------------------------------------------------


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _surprise_pct(actual, estimate) -> float | None:
    """Percent beat/miss. Undefined against a zero or negative estimate, where
    the percentage would be meaningless or sign-flipped."""
    a, e = _num(actual), _num(estimate)
    if a is None or e is None or e <= 0:
        return None
    return round((a / e - 1.0) * 100.0, 2)


def _extract_earnings(rows: list | None, as_of: pd.Timestamp, target: pd.Timestamp) -> tuple[dict, dict]:
    """Split the `earnings` response into the last reported and the next scheduled."""
    last, nxt = {}, {}
    if not rows:
        return last, nxt
    df = pd.DataFrame(rows)
    if "date" not in df:
        return last, nxt
    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"]).sort_values("_d")

    reported = df[(df["_d"] <= as_of) & df["epsActual"].notna()] if "epsActual" in df else df[df["_d"] <= as_of]
    if not reported.empty:
        r = reported.iloc[-1]
        last = {
            "date": r["_d"].date().isoformat(),
            "eps_actual": _num(r.get("epsActual")),
            "eps_estimated": _num(r.get("epsEstimated")),
            "eps_surprise_pct": _surprise_pct(r.get("epsActual"), r.get("epsEstimated")),
            "revenue_actual": _num(r.get("revenueActual")),
            "revenue_estimated": _num(r.get("revenueEstimated")),
            "revenue_surprise_pct": _surprise_pct(r.get("revenueActual"), r.get("revenueEstimated")),
        }

    upcoming = df[df["_d"] > as_of]
    if not upcoming.empty:
        r = upcoming.iloc[0]
        d = r["_d"]
        nxt = {
            "date": d.date().isoformat(),
            "eps_estimated": _num(r.get("epsEstimated")),
            "revenue_estimated": _num(r.get("revenueEstimated")),
            "days_until": int((d - as_of).days),
            # Event risk the weekly model does not price: a print inside the
            # holding week can swamp a 1-week reversion edge entirely.
            "in_holding_week": bool(as_of < d <= target),
        }
    return last, nxt


def _extract_valuation(km: dict | None, ratios: dict | None) -> dict:
    """TTM multiples. Note ROE lives in key-metrics, not ratios."""
    km, ratios = km or {}, ratios or {}

    def pct(v):
        n = _num(v)
        return None if n is None else round(n * 100.0, 2)

    return {
        "pe_ttm": _num(ratios.get("priceToEarningsRatioTTM")),
        "pb_ttm": _num(ratios.get("priceToBookRatioTTM")),
        "ps_ttm": _num(ratios.get("priceToSalesRatioTTM")),
        "ev_ebitda_ttm": _num(km.get("evToEBITDATTM")),
        "ev_sales_ttm": _num(km.get("evToSalesTTM")),
        "ev_fcf_ttm": _num(km.get("evToFreeCashFlowTTM")),
        "net_debt_ebitda_ttm": _num(km.get("netDebtToEBITDATTM")),
        "debt_to_equity_ttm": _num(ratios.get("debtToEquityRatioTTM")),
        "gross_margin_ttm": pct(ratios.get("grossProfitMarginTTM")),
        "net_margin_ttm": pct(ratios.get("netProfitMarginTTM")),
        "roe_ttm": pct(km.get("returnOnEquityTTM")),
        "roic_ttm": pct(km.get("returnOnInvestedCapitalTTM")),
        "fcf_yield_ttm": pct(km.get("freeCashFlowYieldTTM")),
    }


def _extract_estimates(rows: list | None) -> dict:
    """Nearest forward fiscal year from `analyst-estimates` (returned descending)."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if "date" not in df:
        return {}
    df["_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_d"]).sort_values("_d")
    fwd = df[df["_d"] > pd.Timestamp.utcnow().tz_localize(None)]
    r = (fwd if not fwd.empty else df).iloc[0]
    return {
        "date": r["_d"].date().isoformat(),
        "eps_avg": _num(r.get("epsAvg")),
        "eps_low": _num(r.get("epsLow")),
        "eps_high": _num(r.get("epsHigh")),
        "revenue_avg": _num(r.get("revenueAvg")),
        "n_analysts": _num(r.get("numAnalystsEps")) or _num(r.get("numAnalystsRevenue")),
    }


def _extract_consensus(cons: dict | None, pt: dict | None, last_close: float | None,
                       ticker: str = "") -> dict:
    """Ratings + price target. US-only on FMP; elsewhere say so explicitly
    rather than rendering an empty panel.

    `reason` is written as one finished sentence the page can display verbatim —
    the raw endpoint error stays on the dossier's `errors` list instead, so the
    UI never ends up concatenating "grades-consensus: ..." with its own prose.
    """
    if not cons and not pt:
        reason = (
            "FMP publishes analyst ratings and price targets for US listings only, "
            "so this LSE/TSX line has no consensus panel. It is a data-coverage "
            "gap, not a signal about the company."
            if ticker and not fmp_deep.is_us_listing(ticker)
            else "FMP returned no analyst consensus for this symbol."
        )
        return {"available": False, "reason": reason}
    out = {"available": True}
    if cons:
        out.update(
            strong_buy=cons.get("strongBuy"), buy=cons.get("buy"), hold=cons.get("hold"),
            sell=cons.get("sell"), strong_sell=cons.get("strongSell"), label=cons.get("consensus"),
        )
    if pt:
        avg = _num(pt.get("lastMonthAvgPriceTarget")) or _num(pt.get("lastQuarterAvgPriceTarget"))
        out["avg_pt_last_month"] = avg
        out["n_targets_last_month"] = pt.get("lastMonthCount")
        if avg and last_close:
            out["pt_vs_price_pct"] = round((avg / last_close - 1.0) * 100.0, 2)
    return out


# --- dossier ------------------------------------------------------------------


def build_dossier(
    ticker: str,
    meta: dict,
    dd: dict,
    as_of: pd.Timestamp,
    target: pd.Timestamp,
    *,
    stats: fmp_deep.CallStats | None = None,
    force: bool = False,
) -> dict:
    """All three angles for one name. Never raises."""
    errors: list[str] = []
    kw = {"rate_limit_per_min": dd["rate_limit_per_min"], "force": force, "stats": stats}

    def _rec(err):
        if err:
            errors.append(err)

    profile, err = fmp_deep.get_profile(ticker, **kw); _rec(err)
    profile = profile or {}

    earnings_rows, err = fmp_deep.get_earnings(ticker, **kw); _rec(err)
    estimates_rows, err = fmp_deep.get_estimates(ticker, **kw); _rec(err)
    km, err = fmp_deep.get_key_metrics_ttm(ticker, **kw); _rec(err)
    ratios, err = fmp_deep.get_ratios_ttm(ticker, **kw); _rec(err)
    cons, err = fmp_deep.get_consensus(ticker, **kw); _rec(err)
    pt, err = fmp_deep.get_price_target(ticker, **kw); _rec(err)

    bars, err = fmp_deep.get_daily_bars_with_rsi(
        ticker, rsi_period=dd["rsi_period"], **kw
    ); _rec(err)
    tech = build_technicals(bars, meta.get("side", ""), as_of=as_of)

    last_earn, next_earn = _extract_earnings(earnings_rows, as_of, target)
    if next_earn.get("in_holding_week"):
        tech.setdefault("flags", []).append("earnings_in_holding_week")

    company_name = profile.get("companyName")
    news = tavily_client.search_news(
        ticker, company_name, meta.get("region", "US"),
        days=dd["news_days"], max_results=dd["news_max_results"], force=force,
    )
    if news.get("error"):
        errors.append(news["error"])
    if fmp_deep.is_us_listing(ticker):
        fmp_news, err = fmp_deep.get_fmp_news(ticker, **kw)
        if err:
            errors.append(err)
        news = tavily_client.merge_fmp_news(news, fmp_news)

    return {
        "meta": {
            **meta,
            "company_name": company_name,
            "currency": profile.get("currency"),
            "sector": profile.get("sector"),
            "market_cap": _num(profile.get("marketCap")),
            "beta": _num(profile.get("beta")),
            "exchange": profile.get("exchangeFullName"),
        },
        "fundamentals": {
            "last_earnings": last_earn,
            "next_earnings": next_earn,
            "estimates_next_fy": _extract_estimates(estimates_rows),
            "valuation": _extract_valuation(km, ratios),
            "consensus": _extract_consensus(cons, pt, tech.get("last_close"), ticker),
        },
        "technicals": tech,
        "news": news,
        "narrative": None,
        "errors": errors,
    }


def build_artifact(
    signal_day: str,
    target_date: str | None = None,
    cfg: dict | None = None,
    *,
    top_n: int | None = None,
    sides: list[str] | None = None,
    regions: list[str] | None = None,
    tickers: list[str] | None = None,
    force: bool = False,
    progress=None,
) -> dict:
    """Fetch every angle for the selected names and assemble the artifact."""
    dd = deepdive_config(cfg)
    signal_day = signal_day.upper()

    files = list_forecasts(signal_day)
    if not files:
        raise FileNotFoundError(
            f"No forecasts under {FORECASTS_ROOT / signal_day}. "
            f"Run: python -m src.predict --signal-day {signal_day}"
        )
    path = next((f for f in files if f.stem == target_date), None) if target_date else files[0]
    if path is None:
        raise FileNotFoundError(f"No forecast for {signal_day} target {target_date}")

    fc = pd.read_parquet(path)
    as_of = pd.Timestamp(fc["as_of_date"].iloc[0])
    target = pd.Timestamp(fc["target_date"].iloc[0])

    sel = select_universe(
        fc,
        top_n if top_n is not None else dd["top_n"],
        sides or dd["sides"],
        regions or dd["regions"],
        max_top_n=dd["max_top_n"],
    )
    if tickers:
        sel = sel[sel["ticker"].isin(tickers)]

    stats = fmp_deep.CallStats()
    started = time.time()
    dossiers: dict[str, dict] = {}

    for i, row in enumerate(sel.itertuples(index=False), start=1):
        if progress:
            progress(i, len(sel), row.ticker)
        meta = {
            "region": row.region,
            "side": row.side,
            "rank": int(row.rank),
            "prediction": _num(getattr(row, "prediction", None)),
            "industry": getattr(row, "industry", None),
            "model_commentary": getattr(row, "model_commentary", None),
        }
        # Per-region trading target where the forecast resolved one.
        row_target = getattr(row, "target_trade_date", None)
        ticker_target = pd.Timestamp(row_target) if pd.notna(row_target) else target
        try:
            dossiers[row.ticker] = build_dossier(
                row.ticker, meta, dd, as_of, ticker_target, stats=stats, force=force
            )
        except Exception as exc:  # noqa: BLE001 — one bad name must not lose the run
            log.error(f"{row.ticker}: {exc}")
            dossiers[row.ticker] = {
                "meta": meta, "fundamentals": {}, "technicals": {}, "news": {},
                "narrative": None, "errors": [f"dossier failed: {type(exc).__name__}: {exc}"],
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "signal_day": signal_day,
        "as_of_date": as_of.date().isoformat(),
        "target_date": target.date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forecast_file": str(path.relative_to(PROCESSED_DIR.parent)),
        "config": {
            "top_n": top_n if top_n is not None else dd["top_n"],
            "sides": [s.upper() for s in (sides or dd["sides"])],
            "regions": list(regions or dd["regions"]),
            "news_days": dd["news_days"],
            "llm_backend": dd["llm_backend"],
        },
        "provenance": {
            **stats.as_dict(),
            "elapsed_seconds": round(time.time() - started, 1),
            "n_tickers": len(dossiers),
            "n_narratives": 0,
            "news_source_by_region": {"US": "fmp+tavily", "UK": "tavily", "CA": "tavily"},
        },
        "tickers": dossiers,
    }
