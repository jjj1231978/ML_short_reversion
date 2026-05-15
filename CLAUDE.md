# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-based short-horizon equity reversion strategy, replicating and extending the approach from Societe Generale's research on machine learning for weekly mean reversion in developed markets. The model predicts forward 1-week stock returns (Wednesday close to Wednesday close) using an ensemble of XGBoost, LightGBM, RandomForest, and MLP, with SHAP-based interpretability per ensemble member.

**Core hypothesis**: ML models naturally discover that short-term price reversal (R1W) and earnings momentum (EPS revisions) are the dominant drivers of weekly alpha.

**Source paper**: `ML_Mean_Reversion.pdf` (SG Cross Asset Research, 2 Apr 2025).
**Spec**: see [`.specify/README.md`](.specify/README.md) for the full breakdown of objective, requirements, procedures, constraints, expected outputs, planned Streamlit app, and the prioritized implementation plan with paper-vs-code gaps called out.

## Architecture

```
data/           — raw and processed data caches
src/
  data/         — data acquisition (FMP for prices/fundamentals/grades/macro, Databento for high-freq prices, simfin legacy)
  features/     — factor computation, neutralization pipeline, HMM macro regimes
  model/        — rolling-window ensemble training (xgb / lgbm / rf / mlp)
  backtest/     — portfolio construction, ADV-scaled basket weights, cost model, R1W baseline
  diagnostics/  — SHAP fan-out, alpha decay, weekday effect
  reporting/    — multi-agent (briefing → critique → synthesize) report generation
app/            — Streamlit viewer (7 pages: overview, predictions, SHAP, alpha decay, weekday, param explorer, report)
notebooks/      — exploratory analysis; figures/ holds the persisted SHAP PNGs
configs/        — YAML configs for universe, model params, backtest settings
tests/          — unit and integration tests
```

## Key Design Decisions

- **Universe**: multi-region — US S&P 500 (point-in-time via FMP, back to 1957), UK FTSE 100, CA TSX 60 (current-snapshot only on FMP; survivorship-biased — accepted limitation). Financials excluded. Per-region ADV thresholds in native currency (USD / GBp / CAD).
- **Rebalance cadence**: Weekly on Wednesdays
- **Rolling window**: paper uses 520-week train / 104-week validation / 1-week test, retrain every 12 weeks. Current `configs/default.yaml` uses **78 / 26 / 1** as a held-over Phase-1 default; the data now spans 2008-2026 so the paper window fits — bump tracked in `.specify/007-implementation-plan.md` task H.10.
- **Neutralization**: cross-sectional per (date, factor) — winsorize 2%/98% → iterative z-score (10x) → cap at ±3 → subtract industry median → drop tickers with >10 missing factors and zero-fill the rest. Wired via `neutralize_stacked` in `src/main.py`. Macro/regime columns bypass the cross-section as passthrough (would otherwise collapse to NaN with std=0 across the cross-section).
- **Target**: Cross-sectionally z-scored forward 1-week returns
- **Cost model**: 1.5 bps per side, 1-day execution lag (enforced — `main.py` refuses lag<1), ADV-based position scaling (`min(1, ADV/threshold)` then renormalize), per-region thresholds.

## Factor Groups

Currently 24 factors live in the model (vs paper's 86):

| Group | Factors | Data Source |
|---|---|---|
| Price Reversals | R1W, IREV1W (beta-adjusted residual), RSI5D, RSI14, RSI30 | FMP / Databento prices |
| Price Momentum | R1M, R3M1M, R6M1M, R12M1M | FMP / Databento prices |
| Low Risk | VOL6M, VOL12M, BETA6M | FMP / Databento prices + market index |
| Value | PE, PB, EVEBIT, EVEBITDA, EVSALES | SimFin / FMP fundamentals |
| Profitability / Quality | ROE, ROA, GPOA, OPRDIC, GBROC, DE, ACCRUALS, SALES_GROWTH_YOY | SimFin / FMP fundamentals |
| Analyst (UPDOWN1W proxy) | UPDOWN1W_RATINGS | FMP `/grades` (rating-change events, *not* EPS-revision counts) |
| Macro / Regime (passthrough) | macro returns + HMM-regime posteriors | FMP series + fitted HMM in `src/features/regime.py` |

**Gap vs paper**: true UPDOWN1W (EPS-estimate-revision count) and SUE1W/3/6 are
absent — gated on an IBES / Refinitiv / FMP-estimates sourcing decision (see
`.specify/007-implementation-plan.md` block E). The FMP `/grades` proxy is named
`UPDOWN1W_RATINGS` to keep the SHAP comparison against the paper honest.

## Build and Run Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys — FMP is the primary source. .env at repo root is loaded automatically.
export FMP_API_KEY="..."
export DATABENTO_API_KEY="db-..."        # optional, for price_source: databento
export ANTHROPIC_API_KEY="sk-ant-..."    # reporting layer (briefing/critique/synthesize)
export OPENAI_API_KEY="sk-..."           # alternative provider for reporting

# Run full pipeline (data → features → ensemble → backtest → SHAP → diagnostics → save)
python -m src.main

# Streamlit viewer over the cached parquet outputs
streamlit run app/streamlit_app.py

# Multi-agent report from the latest run's artifacts
python -m src.reporting.run_report --profile cheap

# Tests
pytest tests/
pytest tests/test_neutralization_stacked.py  # single test
```

## Critical Implementation Notes

- **Survivorship bias**: Use point-in-time constituent lists, not current index membership
- **Look-ahead bias**: Fundamental data must use a 60-90 day publication lag from fiscal quarter end
- **EPS revision timing**: Analyst estimate timestamps must be strictly point-in-time
- **Neutralization ordering**: Winsorize -> z-score (iterate 10x) -> industry neutralize. Skipping this produces a model that learns sector bets, not stock-level alpha
- **Wednesday-to-Wednesday returns**: The target variable and rebalancing both key off Wednesday closes specifically

## Validation Checklist (paper's predictions vs current results)

Paper claims to verify; current state in parentheses:

- R1W should emerge as the top SHAP feature organically — **not confirmed** (currently #4; VOL12M and VOL6M lead)
- 7-day EPS revision should rank second — **untestable** (true UPDOWN1W not yet sourced; the FMP-grades proxy `UPDOWN1W_RATINGS` ranks #23)
- Alpha decay test: plot annualized IR vs 0-4 day execution lags — **confirmed** (monotone 12.7% → 4.7%)
- Weekday effect: Thursday signals should outperform Monday signals — **confirmed and stronger** (monotone Mon→Fri ramp, 22% → 35%)
- Quintile spread: monotonic returns from Q1 (short) to Q5 (long) — TODO surface in Streamlit
- Compare ML model vs R1W-only baseline — done (ensemble IR 1.00 vs baseline ~1.10; ML's edge is in turnover/drawdown, not return)
- vs earnings-filtered reversal baseline — gated on Phase 2 UPDOWN1W

## Backtest Scope

Phase 1 (shipped 2026-05-15): multi-region US/UK/CA (S&P 500 ex-financials PIT, FTSE 100 + TSX 60 snapshot), 2008-01-30 → 2026-05-13 (955 weeks), 24 factors, ensemble of XGB + LGB + RF + MLP. Ensemble IR (net of 1.5 bps/side, 1-day lag) = 1.00; best member (RF) = 1.13; paper net Global L/S IR = 1.6. R1W reversal baseline IR ≈ 1.10. See [`.specify/007-implementation-plan.md`](.specify/007-implementation-plan.md) for the full results table.

Phase 2 priorities (ordered by expected information value): true UPDOWN1W from IBES/Refinitiv/FMP-estimates → region × industry quintile peer grouping → factor expansion toward 86 → 520/104 train/val window → EU + JP regions. See [`.specify/001-overview.md`](.specify/001-overview.md) for phase boundaries and [`.specify/007-implementation-plan.md`](.specify/007-implementation-plan.md) for the ordered work list.
