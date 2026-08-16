# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-based short-horizon equity reversion strategy, replicating and extending the approach from industry research on machine learning for weekly mean reversion in developed markets. The model predicts forward 1-week stock returns (Wednesday close to Wednesday close) using an ensemble of XGBoost, LightGBM, RandomForest, and MLP, with SHAP-based interpretability per ensemble member.

**Core hypothesis**: ML models naturally discover that short-term price reversal (R1W) and earnings momentum (EPS revisions) are the dominant drivers of weekly alpha.

**Source paper**: `ML_Mean_Reversion.pdf` (industry research, 2 Apr 2025).
**Spec**: see [`.specify/README.md`](.specify/README.md) for the full breakdown of objective, requirements, procedures, constraints, expected outputs, planned Streamlit app, and the prioritized implementation plan with paper-vs-code gaps called out.

## Architecture

```
data/           — raw and processed data caches
src/
  data/         — data acquisition (FMP for prices/fundamentals/grades/macro, Databento for high-freq prices, simfin legacy)
  features/     — factor computation, neutralization pipeline, HMM macro regimes
  model/        — rolling-window ensemble training (xgb / lgbm / rf / mlp). LightGBM
                  supports a random-hyperparameter ensemble (`model.lightgbm.random_ensemble`):
                  per retrain, train N candidates → keep top-K by validation rank-IC →
                  average (`LGBBag`). Modal CPU fan-out in `model/lgb_modal.py`. Ported
                  from the ML_short_sentiment fork, where it ~2×'d the LGB member's IR
                  (0.509 → 0.983); not yet re-measured on this fork.
  backtest/     — portfolio construction, ADV-scaled basket weights, cost model, R1W baseline
  diagnostics/  — SHAP fan-out, alpha decay, weekday effect
  research/     — per-name deep dive over the live picks: FMP fundamentals/valuation,
                  Tavily news (FMP publishes none for .L/.TO), a deterministic
                  technical verdict, and a written brief per name
  reporting/    — multi-agent (briefing → critique → synthesize) report generation
  main.py       — full backtest pipeline (data → features → ensemble → backtest → SHAP → diagnostics)
  predict.py    — live forecast for the next rebalance, per signal day (WED/THU/FRI); writes forecasts/{day}/<target>.parquet, picks ledger, model-version history
app/            — Streamlit viewer (10 pages: overview, predictions, SHAP, alpha decay, weekday, param explorer, report, live forecast, track record, deep dive).
                  Launch with `PYTHONPATH=. streamlit run app/streamlit_app.py` — Streamlit
                  restores sys.path around each page run, so without it every page fails
                  with "No module named 'app'". The Dockerfile already sets PYTHONPATH.
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
- **LightGBM member**: a random-hyperparameter bag, not a single booster. Each retrain
  samples `n_candidates` param sets from `random_ensemble.param_space` (learning rate
  log-uniform, leaf/sample counts int-rounded), trains one booster per set, scores each
  by **validation rank-IC** (the cross-sectional objective the strategy actually trades),
  and averages the top `n_select`. `backend: modal` fans the fits across CPU containers —
  required here because this backtest has ~58 retrains (690 weeks / retrain_freq 12), so
  200 serial local fits per retrain will not finish. Needs `pip install modal` +
  `modal setup`; set `backend: local` for the in-process loop.
- **Ensemble combination**: `rank_mean` (default) | `mean` | `ir_weighted` (weight members
  by their validation rank-IC per retrain period) | `stack` (walk-forward ridge on member
  cross-sectional ranks). `ir_weighted`/`stack` degrade to `rank_mean` in live
  single-week scoring, which has no IC history to weight with.
- **Cost model**: 1.5 bps per side, 1-day execution lag (enforced — `main.py` refuses lag<1), ADV-based position scaling (`min(1, ADV/threshold)` then renormalize), per-region thresholds.

## Factor Groups

Currently 106 factors live in the model (vs paper's 86). Representative members per group (not exhaustive):

| Group | Factors | Data Source |
|---|---|---|
| Price Reversals | R1W, IREV1W (beta-adjusted residual), IREV1M, IREVVOL1W, IREVVOL1M, RSI5D, RSI14, RSI30 | FMP / Databento prices |
| Price Momentum | R1M, R3M1M, R6M1M, R12M1M, IMOM12M1M, IMOM3M1M, IMOMVOL12M1M, IMOMVOL3M1M | FMP / Databento prices |
| Size | SIZE (log market cap) | FMP / Databento prices |
| Low Risk | VOL6M, VOL12M, IVOL6M, IVOL12M, DOWNVOL6M, BETA6M, BETA12M, plus macro betas (BETA_VIX, BETA_US10Y, BETA_OIL, BETA_DXY) | FMP / Databento prices + market/macro series |
| Value | PE, PB, PTB, PSALES, EVEBIT, EVEBITDA, EVSALES | SimFin / FMP fundamentals |
| Profitability / Quality / Leverage | ROE, ROA, ROIC, GPOA, GBROC, OPRDIC, OPRDA, GROSSMARGIN, EBIT_COVER, ACCRUALS, SALES_GROWTH_YOY, DE, DEBITDA, NDE, NDEBITDA, LTDE, LTDEBITDA | SimFin / FMP fundamentals |
| Analyst (UPDOWN1W proxy) | UPDOWN1W_RATINGS | FMP `/grades` (rating-change events, *not* EPS-revision counts) |
| Macro / Regime (passthrough) | macro 5-day returns (OIL_R5, GOLD_R5, SPX_R5, DXY_R5, US10Y_D5, VIX_D5…) + per-series HMM-regime posteriors (S0/S1/transition across ~13 series: SPX, NDX, RUT, VIX, OIL, GOLD, COPPER, DXY, EURUSD, USDJPY, US3M/10Y/30Y) + trend/level flags (OIL_TREND_UP, US10Y_HI…) | FMP series + fitted HMM in `src/features/regime.py` |

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

# Incremental data refresh (extends the shared ~/data_lake/fmp cache forward).
# Add --refresh-membership periodically so index adds/drops enter the universe.
python -m src.data backfill --regions US,UK,CA --refresh-membership

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

- R1W should emerge as the top SHAP feature organically — **confirmed** in the 106-factor run (R1W now #1 in the ensemble and in xgb/lgb/rf individually; VOL12M demoted to #9, VOL6M to #28). Was #4 in the 24-factor run — the factor expansion fixed it.
- 7-day EPS revision should rank second — **untestable** (true UPDOWN1W not yet sourced; the FMP-grades proxy `UPDOWN1W_RATINGS` ranks #102 of 106)
- Alpha decay test: plot annualized IR vs 0-4 day execution lags — **confirmed** (monotone 12.7% → 4.7%)
- Weekday effect: Thursday signals should outperform Monday signals — **confirmed and stronger** (monotone Mon→Fri ramp, 22% → 35%)
- Quintile spread: monotonic returns from Q1 (short) to Q5 (long) — TODO surface in Streamlit
- Compare ML model vs R1W-only baseline — done (106-factor run: ensemble IR 1.06 vs baseline 1.09; ML's edge is in turnover/drawdown, not return)
- vs earnings-filtered reversal baseline — gated on Phase 2 UPDOWN1W

## Backtest Scope

Phase 1 (shipped 2026-05-15): multi-region US/UK/CA (S&P 500 ex-financials PIT, FTSE 100 + TSX 60 snapshot), 2008-01-30 → 2026-05-13 (955 weeks), ensemble of XGB + LGB + RF + MLP. Latest run (2026-06-06) expands the factor set to **106 factors** over 690 backtest weeks: ensemble IR (net of 1.5 bps/side, 1-day lag) = 1.06; best member (LGB) = 1.17, RF = 1.14; paper net Global L/S IR = 1.6. R1W reversal baseline IR = 1.09. (The earlier 24-factor run scored ensemble IR 1.00, RF 1.13.) See [`.specify/007-implementation-plan.md`](.specify/007-implementation-plan.md) for the full results table.

Phase 2 priorities (ordered by expected information value): true UPDOWN1W from IBES/Refinitiv/FMP-estimates → region × industry quintile peer grouping → factor expansion toward 86 → 520/104 train/val window → EU + JP regions. See [`.specify/001-overview.md`](.specify/001-overview.md) for phase boundaries and [`.specify/007-implementation-plan.md`](.specify/007-implementation-plan.md) for the ordered work list.

## Live Forecasting

`python -m src.predict` produces the next-rebalance long/short picks. **Thursday is the signal day** — it is the `--signal-day` default, it matches `backtest.signal_day`, and it is the only day whose bundles and forecasts are maintained.

The `--signal-day` flag still accepts the other weekdays, each with its **own trained model bundle** (`data/processed/models/{day}/<as-of>.joblib`) and forecast file (`data/processed/forecasts/{day}/<target>.parquet`), fit on day-specific rolling windows. Those are research-only: the WED (118 bundles) and FRI (2 bundles) namespaces are leftovers from the WED→THU migration, not a live book. Do not regenerate them by routine.

A retrain is ~14 min; pass `--no-refresh-prices --no-refresh-fundamentals --no-refresh-grades --no-refresh-membership` to run cache-only (skips the ~45 min FMP fetch). The Streamlit "Live forecast" and "Track record" pages auto-discover whatever forecast files exist per day.

## Deep Dive Research

`python -m src.research deepdive --signal-day THU` researches the model's top-N
picks per side per region (cap 20, so at most 120 names) and writes one JSON
dossier to `data/processed/deepdive/THU/<target>.json`, which the **Deep dive**
Streamlit page renders. Three angles per name: fundamentals (last earnings vs
estimate, next earnings, TTM valuation), theme and news, and a technical verdict.

**Regional data coverage — verified against the live API.** FMP serves
fundamentals, valuation and technicals for all three regions, but `news/stock`,
`grades-consensus` and `price-target-summary` return empty for `.L` and `.TO`
symbols. **Tavily therefore carries the news and theme layer for every region**
(`TAVILY_API_KEY` in `.env`). Do not use `earnings-calendar` — it silently
ignores its `symbol` parameter; use `earnings` instead, which returns past and
upcoming rows together.

**The technical layer is deterministic** (`src/research/technicals.py`): one
`technical-indicators/rsi` call returns ~290 daily OHLCV bars *plus* RSI, from
which SMA20/50/200, ATR, the 52-week range and trailing returns are computed
locally. A seven-state classifier turns those into a verdict, and `alignment`
compares it to the side the model took — a LONG that is `OVERBOUGHT_PULLBACK`
means the reversion model is fighting the tape, which is the single most useful
column on the page. The narrative layer *narrates* this verdict and never
recomputes it. Note the shared price cache under `~/data_lake/fmp/prices/` is
not used here: it only advances when the backtest runs and lags the live
forecast by weeks.

**Narrative backends** (`deepdive.llm_backend`):

- `session` (default) — the batch job writes `<target>.factsheets.md` and stops.
  The running Claude Code session writes one JSON brief per ticker and merges
  them with `python -m src.research merge-narratives --file <briefs.json>`.
  Free, uses a stronger model than the metered path, but needs a session in the
  loop. Briefs survive a later `deepdive` rerun (carried forward unless
  `--drop-narratives`).
- `api` — one metered call per name via `src/reporting/llm.py`, budget-guarded
  by `deepdive.max_budget_usd`. This is the backend a cron job needs.
- `none` — deterministic layer only.

Responses are TTL-cached under `~/data_lake/fmp/deepdive/` (profile 30d,
valuation/estimates 7d, earnings/news 1d, bars same-day), so a cold 12-name run
takes ~90s and a re-run ~10s. `--force` bypasses it. Budget for a full 120-name
book: ~840 FMP + 120 Tavily calls.

The page is read-only like the rest of the viewer; a per-ticker refresh button
appears only when `DEEPDIVE_ALLOW_REFRESH=1` and the API keys are set.

### Holidays

The target date is a weekly-bin label from calendar arithmetic, so it can land on a market closure. Resolution is **per region** (`src/lib/trading_days.py`) — a US closure must not move the UK or Canadian legs, since the book spans US/UK/CA and they share no calendar:

- Sessions are derived from the observed price bars, not a holiday library — no dependency to keep patched, correct for every venue in the panel, and it catches one-off closures published calendars miss.
- A shut target falls back to the **prior** session (Thursday → Wednesday), and only then forward (→ Friday). Backwards wins because that close is already known when the target arrives.
- The nominal date stays the forecast's identity (filename, ledger primary key); the resolved date rides alongside as the per-row `target_trade_date` column.
- The **backtest** needs no such handling: `resample("W-THU").last()` bins Friday→Thursday and takes the last available bar, so a shut Thursday already resolves to Wednesday's close. It can never fall forward to Friday — that is the next bin. Changing the binning would alter every feature and target while leaving the retrain fingerprint's date index identical, silently mixing vintages across a resumed run.
