# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-based short-horizon equity reversion strategy, replicating and extending the approach from Societe Generale's research on machine learning for weekly mean reversion in developed markets. The model predicts forward 1-week stock returns (Wednesday close to Wednesday close) using gradient boosted trees (XGBoost/LightGBM) with SHAP-based interpretability.

**Core hypothesis**: ML models naturally discover that short-term price reversal (R1W) and earnings momentum (EPS revisions) are the dominant drivers of weekly alpha.

## Architecture

```
data/           — raw and processed data caches
src/
  data/         — data acquisition (Databento for prices, simfin for fundamentals, optionally WRDS/IBES)
  features/     — factor computation and neutralization pipeline
  model/        — training loop with rolling window, hyperparameter tuning
  backtest/     — portfolio construction, rebalancing, cost model
  diagnostics/  — SHAP analysis, alpha decay, weekday effects
notebooks/      — exploratory analysis and result visualization
configs/        — YAML configs for universe, model params, backtest settings
tests/          — unit and integration tests
```

## Key Design Decisions

- **Universe**: S&P 500 (or Russell 1000), excluding GICS sector 40 (financials), minimum $3M 6-month rolling ADV
- **Rebalance cadence**: Weekly on Wednesdays
- **Rolling window**: 520-week train / 104-week validation / 1-week test, retrain every 12 weeks
- **Neutralization**: Winsorize 2%/98%, iterative z-score (10x), subtract GICS industry median — this step is critical and must precede model training
- **Target**: Cross-sectionally z-scored forward 1-week returns
- **Cost model**: 1.5 bps per side, 1-day execution lag, ADV-based position scaling (threshold $20M)

## Factor Groups

| Group | Factors | Data Source |
|---|---|---|
| Price Reversals | R1W, IREV1W, RSI5D, RSI14 | Price data (Databento or Massive) |
| Price Momentum | R3M1M, R12M1M | Price data (Databento or Massive) |
| Earnings Momentum | UPDOWN1W, SUE1W | IBES/FMP (if available) |
| Low Risk | VOL6M, BETA6M | Price data |
| Value | PE, PB | Fundamentals (simfin) |
| Profitability | ROE, GPOA | Fundamentals (simfin) |

Start with price-based factors only; add earnings revision data once the pipeline is validated.

## Build and Run Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API key for price data (one of these, depending on price_source in config)
export DATABENTO_API_KEY="db-..."        # for price_source: databento
export MASSIVE_API_KEY="..."             # for price_source: massive (formerly Polygon.io)

# Run full pipeline
python -m src.main

# Run individual phases
python -m src.data.fetch          # data acquisition
python -m src.features.build      # feature engineering
python -m src.model.train         # model training
python -m src.backtest.run        # backtesting
python -m src.diagnostics.shap_analysis  # SHAP plots

# Tests
pytest tests/
pytest tests/test_features.py -k "test_neutralization"  # single test
```

## Critical Implementation Notes

- **Survivorship bias**: Use point-in-time constituent lists, not current index membership
- **Look-ahead bias**: Fundamental data must use a 60-90 day publication lag from fiscal quarter end
- **EPS revision timing**: Analyst estimate timestamps must be strictly point-in-time
- **Neutralization ordering**: Winsorize -> z-score (iterate 10x) -> industry neutralize. Skipping this produces a model that learns sector bets, not stock-level alpha
- **Wednesday-to-Wednesday returns**: The target variable and rebalancing both key off Wednesday closes specifically

## Validation Checklist

- R1W should emerge as the top SHAP feature organically
- 7-day EPS revision should rank second (if earnings data available)
- Alpha decay test: plot annualized IR vs 0-4 day execution lags
- Weekday effect: Thursday signals should outperform Monday signals
- Quintile spread: monotonic returns from Q1 (short) to Q5 (long)
- Compare ML model vs R1W-only baseline vs earnings-filtered reversal baseline

## Backtest Scope

Start with a simplified version: US-only (S&P 500), 2019-2024 (5-year window covering SG's live period), 10-15 factors. Expand to full 86-factor cross-regional implementation only after core findings are validated.
