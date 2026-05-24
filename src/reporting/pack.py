"""Assemble the research pack: structured JSON facts fed to all three agents.

Reads whatever pipeline outputs are available in `data/processed/` (predictions,
portfolio_returns, shap_values, alpha_decay, weekday_effect, metrics_summary)
and combines them with hard-coded paper reference values plus excerpts from
`.specify/004-constraints.md` so agents have the validation context.

Robust to missing files — flags them as `not_yet_produced` rather than failing,
so the report can be smoke-tested before the full pipeline runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

PAPER_PDF = PROJECT_ROOT / "ML_Mean_Reversion.pdf"
PAPER_TEXT_CACHE = PROCESSED_DIR / "_cache" / "paper_text.txt"


def _extract_paper_text() -> str:
    """Extract text from the industry research PDF, cached after first extraction.

    Returns the full text of the source paper so all three agents can use it
    as a template (section structure) and as the reference to compare results
    against. Skips silently if the PDF or pypdf is unavailable.
    """
    if PAPER_TEXT_CACHE.exists():
        return PAPER_TEXT_CACHE.read_text()
    if not PAPER_PDF.exists():
        log.warning(f"Paper PDF not found at {PAPER_PDF}; agents will rely only on hardcoded reference values.")
        return ""
    try:
        import pypdf
    except ImportError:
        log.warning("pypdf not installed; cannot extract paper text. Install with: pip install pypdf")
        return ""
    # pypdf emits hundreds of "Ignoring wrong pointing object" warnings on this
    # PDF; silence them so they don't drown the run log.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = pypdf.PdfReader(str(PAPER_PDF))
    parts = []
    for i, page in enumerate(reader.pages):
        parts.append(f"\n========== PAGE {i+1} ==========\n")
        parts.append(page.extract_text() or "")
    text = "\n".join(parts)
    PAPER_TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TEXT_CACHE.write_text(text)
    log.info(f"Extracted {len(text):,} chars from paper PDF; cached at {PAPER_TEXT_CACHE}")
    return text


# Hardcoded reference numbers from industry research "Can ML help us design a better
# stock mean reversion strategy?", 2 April 2025. Page references in comments.
# Authoritative Phase 1 factor list. The pipeline computes these; agents must
# reference them by these exact names rather than inventing plausible-sounding
# alternatives. Sourced from src/features/build.py and configs/default.yaml.
PHASE1_FACTORS = [
    {"name": "R1W", "group": "Price reversal", "definition": "5-day total return"},
    {"name": "IREV1W", "group": "Price reversal", "definition": "1-week beta-adjusted residual return: stock_5d_ret − BETA6M × market_5d_ret. Implemented in src/features/factors.py:irev1w."},
    {"name": "RSI5D", "group": "Price reversal", "definition": "5-day Relative Strength Index"},
    {"name": "RSI14", "group": "Price reversal", "definition": "14-day Relative Strength Index"},
    {"name": "R3M1M", "group": "Price momentum", "definition": "3-month total return lagged 1 month"},
    {"name": "R12M1M", "group": "Price momentum", "definition": "12-month total return lagged 1 month"},
    {"name": "VOL6M", "group": "Low risk", "definition": "6-month annualized volatility of daily returns"},
    {"name": "BETA6M", "group": "Low risk", "definition": "6-month rolling beta vs SPY (proxy for SPX; main.py fetches SPY close via fetch_market_close and passes it to build_feature_matrix). Has ~25 weeks of warm-up NaN at the start of the backtest window."},
    {"name": "PE", "group": "Value", "definition": "Trailing 12-month price/earnings (from SimFin TTM)"},
    {"name": "PB", "group": "Value", "definition": "Price/book (from SimFin balance + market value)"},
    {"name": "ROE", "group": "Profitability", "definition": "Trailing 12-month return on equity"},
    {"name": "GPOA", "group": "Profitability", "definition": "Trailing 12-month gross profit / total assets"},
    {"name": "UPDOWN1W_RATINGS", "group": "Earnings momentum (proxy)", "definition": "7-day rolling (analyst rating upgrades − downgrades) / total rating actions per ticker. Sourced from FMP /stable/grades. PROXY for the paper's UPDOWN1W (which uses EPS revisions from IBES); same formula structure but uses analyst rating revisions instead. Honest distinct name keeps SHAP comparisons against the paper unambiguous."},
]


PAPER_REFERENCE = {
    "source": "Industry research, Global Style Counselling, 2 April 2025",
    "us_net_of_costs_2006_2025": {  # paper p.10
        "annualized_return": 0.086,
        "annualized_volatility": 0.085,
        "information_ratio": 1.0,
        "max_drawdown": -0.180,
    },
    "global_net_of_costs_2006_2025": {  # paper p.10
        "annualized_return": 0.097,
        "annualized_volatility": 0.063,
        "information_ratio": 1.6,
        "max_drawdown": -0.123,
    },
    "alpha_decay_global": {  # paper p.7-8
        0: 0.149,
        1: 0.116,
        2: 0.095,
        3: 0.081,
        4: 0.068,
    },
    "weekday_signals_global_pct_data_release": {  # paper p.9
        "Monday": 0.08,
        "Tuesday": 0.17,
        "Wednesday": 0.23,
        "Thursday": 0.28,
        "Friday": 0.24,
    },
    "two_way_weekly_turnover": {  # paper p.9
        "ML": 2.12,
        "Basic Price Reversals": 3.17,
        "Earnings-Filtered Price Reversals": 3.41,
    },
    "top_shap_features": [  # paper p.7
        "R1W (1-week return) — most important",
        "UPDOWN1W (7-day EPS upward minus downward revisions) — second most important",
    ],
    "validation_checklist": [
        "Top SHAP feature should be R1W",
        "Second SHAP feature should be UPDOWN1W (Phase 2 only — needs earnings revisions)",
        "Alpha decay should be monotone over 0d→4d execution lag",
        "Thursday signal should outperform Monday signal",
        "Quintile spread Q1→Q5 should be monotonic",
        "ML should beat plain R1W reversal baseline net of costs",
        "ML should beat earnings-filtered R1W reversal baseline net of costs",
        "ML 2-way weekly turnover should be lower than basic R1W reversal (paper: 212% vs 317%)",
    ],
}


# Known gaps in the codebase relative to the paper, sourced from
# .specify/007-implementation-plan.md. Surfaced to agents so they don't have
# to discover these from the data alone.
KNOWN_GAPS = [
    "BETA6M and several momentum factors have a multi-week warm-up period at the start of the backtest window (rolling lookback >= 126 days for BETA6M / VOL6M, 252 days for R12M1M); the missing-data rule zero-fills these so the model can train, but training quality on the early weeks is reduced.",
    "Paper-correct earnings-revision factors (UPDOWN1W, SUE1W, SUE3, SUE6) require historical EPS-estimate revision data (IBES/Refinitiv); not available on FMP Starter. UPDOWN1W_RATINGS uses analyst rating revisions as a proxy — same formula structure but a different underlying signal. SUE factors remain unimplemented.",
    "Universe membership is current Wikipedia S&P 500 constituents; survivorship-biased. Phase 2 needs point-in-time membership.",
    "Quintiles formed across the full universe; paper forms within region × GICS industry peer groups. Phase 2 work.",
    "Backtest window is ~3 years (Databento EQUS.MINI history starts 2023-03-28) vs paper's 2006-2025. The narrow regime (post-COVID rate cycle) likely inflates IR vs the 19-year sample.",
    "Train / val window is 78 / 26 weeks vs paper's 520 / 104 — forced by data history; can be relaxed in Phase 2 with a longer-history price source.",
    "Cost model is 1.5 bps per side with no borrow cost on shorts and no market-impact term beyond the bps charge.",
    "Universe is US-only (S&P 500 ex-financials). Paper covers global developed (US + Europe + Japan).",
]


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log.warning(f"Failed to read {path}: {e}")
        return None


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read {path}: {e}")
        return None


def _summarize_portfolio(df: pd.DataFrame) -> dict:
    """Headline metrics from a portfolio_returns parquet."""
    if df is None or df.empty:
        return {"status": "not_yet_produced"}
    out: dict[str, Any] = {
        "status": "available",
        "num_weeks": int(len(df)),
        "date_first": str(df.index.min()) if df.index.size else None,
        "date_last": str(df.index.max()) if df.index.size else None,
    }
    if "ls_return_net" in df.columns:
        ls = df["ls_return_net"].dropna()
        if len(ls):
            ann_ret = float(ls.mean() * 52)
            ann_vol = float(ls.std() * (52 ** 0.5))
            cum = (1 + ls).cumprod()
            max_dd = float((cum / cum.cummax() - 1).min())
            out["annualized_return"] = round(ann_ret, 4)
            out["annualized_volatility"] = round(ann_vol, 4)
            out["information_ratio"] = round(ann_ret / ann_vol, 3) if ann_vol else None
            out["max_drawdown"] = round(max_dd, 4)
    if "turnover" in df.columns:
        out["avg_weekly_turnover"] = round(float(df["turnover"].mean()), 4)
    return out


def _summarize_predictions(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "not_yet_produced"}
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    return {
        "status": "available",
        "num_observations": int(len(s)),
        "num_unique_dates": int(s.index.get_level_values("date").nunique())
        if hasattr(s.index, "get_level_values") and "date" in s.index.names
        else None,
    }


def _summarize_shap(importance_df: pd.DataFrame, top_n: int = 10) -> dict:
    if importance_df is None or importance_df.empty:
        return {"status": "not_yet_produced"}
    s = importance_df.iloc[:, 0] if isinstance(importance_df, pd.DataFrame) else importance_df
    s_sorted = s.sort_values(ascending=False)
    return {
        "status": "available",
        "top_features": [
            {"feature": str(idx), "mean_abs_shap": float(val)}
            for idx, val in s_sorted.head(top_n).items()
        ],
    }


def _summarize_alpha_decay(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "not_yet_produced"}
    return {
        "status": "available",
        "by_lag": df.to_dict(orient="index"),
    }


def _summarize_weekday(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"status": "not_yet_produced"}
    return {
        "status": "available",
        "by_signal_day": df.to_dict(orient="index"),
    }


def build_research_pack(processed_dir: Path | None = None) -> dict:
    """Assemble the JSON-serializable research pack."""
    cfg = load_config()
    p_dir = processed_dir or PROCESSED_DIR

    portfolio = _safe_read_parquet(p_dir / "portfolio_returns.parquet")
    baseline = _safe_read_parquet(p_dir / "baseline_r1w_returns.parquet")
    predictions = _safe_read_parquet(p_dir / "predictions.parquet")
    importance = _safe_read_parquet(p_dir / "feature_importance.parquet")
    alpha_decay = _safe_read_parquet(p_dir / "alpha_decay.parquet")
    weekday = _safe_read_parquet(p_dir / "weekday_effect.parquet")
    metrics = _safe_read_json(p_dir / "metrics_summary.json")
    model_comparison = _safe_read_json(p_dir / "model_comparison.json")

    paper_text = _extract_paper_text()

    # Build run_metadata that accurately reflects the current pipeline setup.
    # Agents must reason from this — not from any assumed Phase 1 / Phase 2 defaults.
    regions = cfg["universe"].get("regions", [cfg["universe"].get("index", "SP500")])
    universe_method = {}
    for r in regions:
        universe_method[r] = (
            "FMP point-in-time membership (date_added / date_removed)"
            if r.upper() == "US"
            else "FMP current snapshot (survivorship-biased)"
        )
    eligibility_filter_active = bool(
        cfg["universe"].get("min_price_by_region")
        or cfg["universe"].get("min_adv_6m_by_region")
    )

    # Model: surface ensemble members + per-member hyperparameters so the
    # methodology section can write them verbatim and the agent doesn't
    # invent "XGBoost-only" framing.
    m_cfg = cfg["model"]
    model_class = m_cfg.get("type", "xgboost")
    if model_class == "ensemble":
        ensemble_members = list(m_cfg["ensemble"]["members"])
    else:
        ensemble_members = [model_class]
    model_hyperparams = {
        member: m_cfg.get(member, {}) for member in ensemble_members
    }
    # Per-member SHAP top features, alongside the canonical (XGBoost) one.
    shap_per_member = {}
    for member in ensemble_members:
        suffix = "" if member == "xgboost" else f"_{member}"
        imp_df = _safe_read_parquet(p_dir / f"feature_importance{suffix}.parquet")
        shap_per_member[member] = _summarize_shap(imp_df)

    pack = {
        "schema_version": 3,
        "source_paper_full_text": paper_text or "(paper PDF not available; agents must rely on hardcoded reference values)",
        "run_metadata": {
            "regions": regions,
            "universe_method": universe_method,
            "config_universe_index": cfg["universe"].get("index"),
            "config_date_window": {
                "start": cfg["data"].get("start_date"),
                "end": cfg["data"].get("end_date"),
            },
            "factor_count": len(PHASE1_FACTORS),
            "config_factor_count_phase1": len(PHASE1_FACTORS),  # back-compat
            "config_train_weeks": cfg["model"].get("train_weeks"),
            "config_val_weeks": cfg["model"].get("val_weeks"),
            "config_retrain_freq": cfg["model"].get("retrain_freq"),
            "trading_cost_bps": cfg["backtest"].get("trading_cost_bps"),
            "execution_lag_days": cfg["backtest"].get("execution_lag_days"),
            "weekly_return_cap": cfg["backtest"].get("weekly_return_cap"),
            "eligibility_filter_active": eligibility_filter_active,
            "eligibility_filter_components": [
                "per-region price floor (US/CA: $3 native; UK: 100 GBp)",
                "per-region time-varying ADV floor (US: $3M USD; UK: 5M GBp; CA: $100k CAD)",
                "US point-in-time SP500 membership (date_added / date_removed); UK/CA snapshot is always-active",
            ] if eligibility_filter_active else [],
            "min_adv_6m_by_region": cfg["universe"].get("min_adv_6m_by_region"),
            "min_price_by_region": cfg["universe"].get("min_price_by_region"),
            # Data sourcing — explicit so agents do not invent vendor names.
            "price_source": cfg["data"].get("price_source"),
            "fundamental_source": cfg["data"].get("fundamental_source"),
            "universe_source": cfg["data"].get("universe_source"),
            "earnings_revisions_source": "FMP /stable/grades (analyst rating revisions; PROXY for paper's IBES EPS revisions)",
            "data_vendor_summary": (
                "FMP exclusive for prices, fundamentals, index membership, and analyst grades. "
                "History from 2006-01-01. No Databento, SimFin, IBES, Alpha Vantage, Tiingo, "
                "Wikipedia, or other vendors are used in this run — do not reference them."
            ),
            "benchmarks": cfg["data"].get("benchmarks"),
            # Model — explicit so agents do not call this "XGBoost-only".
            "model_class": model_class,
            "ensemble_members": ensemble_members,
            "ensemble_combine_method": m_cfg.get("ensemble", {}).get("combine"),
            "model_hyperparams": model_hyperparams,
        },
        "factor_inventory": PHASE1_FACTORS,
        "phase1_factor_inventory": PHASE1_FACTORS,  # back-compat alias
        "headline_metrics": metrics or _summarize_portfolio(portfolio),
        "portfolio_summary": _summarize_portfolio(portfolio),
        "baseline_r1w_summary": _summarize_portfolio(baseline),
        "predictions_summary": _summarize_predictions(predictions),
        "shap_top_features": _summarize_shap(importance),
        "shap_top_features_per_model": shap_per_member,
        "model_comparison": model_comparison or {"status": "not_yet_produced"},
        "alpha_decay": _summarize_alpha_decay(alpha_decay),
        "weekday_effect": _summarize_weekday(weekday),
        "paper_reference": PAPER_REFERENCE,
        "known_implementation_gaps": KNOWN_GAPS,
        "spec_pointer": ".specify/ — see 001-overview.md, 004-constraints.md, 007-implementation-plan.md",
    }

    has_any_run = portfolio is not None or metrics is not None
    pack["pipeline_run_status"] = "completed" if has_any_run else "no_run_yet"

    return pack


def render_pack_markdown(pack: dict) -> str:
    """Render the research pack as a markdown string (for use in agent prompts)."""
    return "```json\n" + json.dumps(pack, indent=2, default=str) + "\n```"


if __name__ == "__main__":
    pack = build_research_pack()
    print(json.dumps(pack, indent=2, default=str))
